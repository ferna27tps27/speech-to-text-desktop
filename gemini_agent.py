import os
import sys
import time
import threading
import queue
import logging
import pyaudio
import pyperclip
import pyautogui
from pynput import keyboard
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Load env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    logging.warning("GEMINI_API_KEY not found in .env. Please ensure it is set.")

# Configuration (load from .env when possible)
TTS_MODEL = os.getenv("TTS_MODEL")
STT_MODEL = os.getenv("STT_MODEL")
VOICE_NAME = os.getenv("VOICE_NAME", "Kore")
WARMUP_TTS = os.getenv("WARMUP_TTS", "false").lower() in ("1", "true", "yes")
TTS_CHUNK_CHARS = int(os.getenv("TTS_CHUNK_CHARS", "2000"))
TTS_COMBINE_CHUNKS = os.getenv("TTS_COMBINE_CHUNKS", "true").lower() in ("1", "true", "yes")
TERMINAL_PASTE_DEBUG = os.getenv("TERMINAL_PASTE_DEBUG", "true").lower() in ("1", "true", "yes")

if not TTS_MODEL:
    logging.warning("TTS_MODEL not set in .env; using default.")
    TTS_MODEL = "gemini-2.5-flash-preview-tts"

if not STT_MODEL:
    logging.warning("STT_MODEL not set in .env; using default.")
    STT_MODEL = "gemini-2.5-flash"  # Capable of audio understanding

# Audio Config
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE_IN = 16000  # Gemini expects 16kHz input for best results
RATE_OUT = 24000 # Gemini TTS outputs 24kHz

import subprocess

# ... (existing imports)

class AudioHandler:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.recording = False
        self.frames = []
        self.stream = None

    def play_system_sound(self, sound_name):
        """Plays a system sound for feedback."""
        try:
            # macOS system sounds
            sounds = {
                "start": "/System/Library/Sounds/Tink.aiff",
                "stop": "/System/Library/Sounds/Pop.aiff",
                "error": "/System/Library/Sounds/Basso.aiff"
            }
            path = sounds.get(sound_name)
            if path and os.path.exists(path):
                subprocess.run(["afplay", path], check=False)
        except Exception:
            pass # Ignore sound errors

    def start_recording(self):
        self.recording = True
        self.frames = []
        try:
            # Re-initialize PyAudio here to be safe across threads/runs
            if not self.p:
                 self.p = pyaudio.PyAudio()
            
            self.stream = self.p.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=RATE_IN,
                                      input=True,
                                      frames_per_buffer=CHUNK)
            logging.info("Microphone input started.")
            self.play_system_sound("start") # Feedback
            
            # Record in a separate thread to not block
            threading.Thread(target=self._record_loop, daemon=True).start()
        except Exception as e:
            logging.error(f"Failed to open microphone: {e}")
            self.play_system_sound("error")
            self.recording = False

    def _record_loop(self):
        while self.recording and self.stream:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                logging.error(f"Error recording stream: {e}")
                break

    def stop_recording(self):
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        logging.info("Microphone input stopped.")
        self.play_system_sound("stop") # Feedback
        return b''.join(self.frames)


    def play_audio_stream(self, response_stream, request_start_time=None):
        """Plays audio chunks as they arrive from the stream."""
        if not response_stream:
            return

        try:
            logging.info("Starting stream playback...")
            # Initialize stream
            stream = self.p.open(format=FORMAT,
                                 channels=CHANNELS,
                                 rate=RATE_OUT,
                                 output=True)
            
            chunk_count = 0
            for chunk in response_stream:
                if chunk.candidates and chunk.candidates[0].content.parts:
                    part = chunk.candidates[0].content.parts[0]
                    if part.inline_data and part.inline_data.data:
                        audio_data = part.inline_data.data
                        stream.write(audio_data)
                        chunk_count += 1
                        if chunk_count == 1:
                            if request_start_time is not None:
                                latency_ms = int((time.monotonic() - request_start_time) * 1000)
                                logging.info(f"First audio chunk played in {latency_ms} ms")
                            else:
                                logging.info("First audio chunk played (Latency check)")
            
            stream.stop_stream()
            stream.close()
            logging.info(f"Stream playback finished. Total chunks: {chunk_count}")
        except Exception as e:
            logging.error(f"Stream playback failed: {e}")

    def play_audio(self, audio_data):
        # Keep old method for backward compatibility if needed, but we prefer stream now
        if not audio_data:
            return
        try:
            logging.info(f"Playing audio ({len(audio_data)} bytes)...")
            stream = self.p.open(format=FORMAT,
                                 channels=CHANNELS,
                                 rate=RATE_OUT,
                                 output=True)
            stream.write(audio_data)
            stream.stop_stream()
            stream.close()
            logging.info("Playback finished.")
        except Exception as e:
            logging.error(f"Failed to play audio: {e}")

class GeminiClient:
    def __init__(self):
        if not API_KEY:
            raise ValueError("API Key is missing")
        self.client = genai.Client(api_key=API_KEY)

    def _add_wav_header(self, pcm_data, rate=16000, channels=1, sampwidth=2):
        import io
        import wave
        
        with io.BytesIO() as wav_file:
            with wave.open(wav_file, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sampwidth)
                wf.setframerate(rate)
                wf.writeframes(pcm_data)
            return wav_file.getvalue()

    def transcribe(self, audio_bytes):
        logging.info("Sending audio to Gemini...")
        try:
            # Wrap in WAV to ensure sample rate is communicated correctly
            wav_data = self._add_wav_header(audio_bytes, rate=RATE_IN)
            
            # Create a user prompt that enforces strict transcription
            response = self.client.models.generate_content(
                model=STT_MODEL,
                contents=[
                    types.Part.from_bytes(data=wav_data, mime_type="audio/wav"),
                    """Transcribe the audio exactly as spoken, but format punctuation commands into their symbols.
                    
                    Rules:
                    - Replace "comma" with ","
                    - Replace "dot" or "period" with "."
                    - Replace "semicolon" with ";"
                    - Replace "colon" with ":"
                    - Replace "question mark" with "?"
                    - Replace "exclamation mark" with "!"
                    - Replace "dash" or "hyphen" with "-"
                    - Replace "new line" or "next line" with a line break
                    - Replace "open parenthesis" with "(" and "close parenthesis" with ")"
                    - Replace "open quote" with '"' and "close quote" with '"'
                    
                    Do not add any other markdown or speaker labels. Just the formatted text."""
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT"],
                    temperature=0.0 # Low temperature for factual transcription
                )
            )
            return response.text
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return None

    def text_to_speech_stream(self, text):
        logging.info("Requesting TTS Stream from Gemini...")
        try:
            # We use stream=True to get chunks of audio as they are generated
            # Note: The 'generate_content_stream' API is used for streaming
            # The model is strictly TTS, so we must be careful with the prompt structure.
            # Passing raw text as the only content is the most reliable way.
            
            response = self.client.models.generate_content_stream(
                model=TTS_MODEL,
                contents=[text], # Ensure list format
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=VOICE_NAME
                            )
                        )
                    )
                )
            )
            return response
        except Exception as e:
            logging.error(f"TTS Stream Request failed: {e}")
            return None

    def text_to_speech(self, text):
        logging.info("Requesting TTS from Gemini...")
        try:
            response = self.client.models.generate_content(
                model=TTS_MODEL,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=VOICE_NAME
                            )
                        )
                    )
                )
            )
            return response
        except Exception as e:
            logging.error(f"TTS Request failed: {e}")
            return None

    def warmup_tts(self):
        """Optional warmup to reduce first-request latency."""
        try:
            response = self.client.models.generate_content_stream(
                model=TTS_MODEL,
                contents=[" "],
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=VOICE_NAME
                            )
                        )
                    )
                )
            )
            for _ in response:
                break
            logging.info("TTS warmup completed.")
        except Exception as e:
            logging.warning(f"TTS warmup failed: {e}")

def _split_text_for_tts(text, max_chars=None):
    max_chars = TTS_CHUNK_CHARS if max_chars is None else max_chars
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        cut = text.rfind('.', start, end)
        if cut == -1:
            cut = text.rfind(' ', start, end)
        if cut == -1 or cut <= start:
            cut = end
        parts.append(text[start:cut].strip())
        start = cut
    return [p for p in parts if p]

def _dedupe_repeated_text(text):
    """Collapse obvious duplicated transcriptions."""
    cleaned = " ".join(text.split())
    if len(cleaned) < 20:
        return text
    half = len(cleaned) // 2
    if len(cleaned) % 2 == 0 and cleaned[:half] == cleaned[half:]:
        return cleaned[:half]
    # Handle exact duplication with a separator
    if cleaned.count(cleaned[:half]) >= 2 and cleaned.startswith(cleaned[:half] + " " + cleaned[:half]):
        return cleaned[:half]
    return text

def _get_frontmost_app_name():
    try:
        applescript = (
            'tell application "System Events" to get name of '
            'first application process whose frontmost is true'
        )
        result = subprocess.run(
            ["osascript", "-e", applescript],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except Exception:
        return None

def _paste_to_frontmost_app():
    try:
        applescript = """
        tell application "System Events"
            keystroke "v" using command down
        end tell
        """
        subprocess.run(["osascript", "-e", applescript], check=False)
    except Exception:
        pass

def _paste_to_terminal(text):
    if not TERMINAL_PASTE_DEBUG:
        return
    try:
        applescript = """
        on run argv
            set theText to item 1 of argv
            tell application "Terminal"
                do script "printf '%s\\n' " & quoted form of theText in front window
            end tell
        end run
        """
        subprocess.run(["osascript", "-e", applescript, text], check=False)
    except Exception:
        pass

# Global Instance
audio = AudioHandler()
gemini = None

def handle_dictation():
    """Toggle recording state."""
    global audio, gemini
    
    if not gemini:
        logging.error("Gemini client not initialized (check API Key).")
        return

    if audio.recording:
        # Stop and Transcribe
        raw_audio = audio.stop_recording()
        
        # Run transcription in a separate thread to prevent blocking
        def process_transcription():
            text = gemini.transcribe(raw_audio)
            if text:
                text = _dedupe_repeated_text(text)
                logging.info(f"Typing: {text}")
                
                # Method 1: Pyperclip + Hotkey (Most reliable for macOS)
                try:
                    pyperclip.copy(text + " ")
                    time.sleep(0.1) # Tiny delay to ensure clipboard is ready
                    
                    # Use AppleScript to paste - this bypasses many permission issues
                    # that pynput/pyautogui might face
                    front_app = _get_frontmost_app_name()
                    if front_app:
                        logging.info(f"Frontmost app: {front_app}")
                    _paste_to_frontmost_app()
                    _paste_to_terminal(text)
                    
                except Exception as e:
                    logging.error(f"Paste failed: {e}")
            else:
                logging.warning("No text returned.")

        threading.Thread(target=process_transcription).start()
        
    else:
        # Start Recording
        audio.start_recording()

def handle_read_aloud():
    """Copy selected text and read it."""
    global gemini, audio
    
    if not gemini:
        logging.error("Gemini client not initialized.")
        return

    logging.info("Simulating copy...")
    
    # Clear clipboard first to ensure we get new text
    pyperclip.copy("") 
    
    # Use pynput to simulate Cmd+C (Hardware level simulation)
    # This bypasses AppleScript/System Events permission issues
    controller = keyboard.Controller()
    with controller.pressed(keyboard.Key.cmd):
        controller.press('c')
        controller.release('c')

    # Wait for clipboard to update (short poll instead of fixed long sleep)
    text = ""
    start = time.monotonic()
    while time.monotonic() - start < 0.6:
        text = pyperclip.paste()
        if text and text.strip():
            break
        time.sleep(0.05)

    if not text or not text.strip():
        logging.warning("No text selected or clipboard empty.")
        return

    text = text.strip()
    if not text:
        logging.warning("Clipboard text was only whitespace.")
        return

    logging.info(f"Reading: {text[:50]}...")
    
    def process_tts():
        # Use non-streaming TTS due to stream instability on TTS model
        chunks = _split_text_for_tts(text)
        if not chunks:
            logging.warning("No non-empty chunks to read aloud.")
            return
        audio_parts = []
        for idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            request_start = time.monotonic()
            response = gemini.text_to_speech(chunk)
            if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                if part.inline_data and part.inline_data.data:
                    elapsed_ms = int((time.monotonic() - request_start) * 1000)
                    if idx == 0:
                        logging.info(f"TTS response received in {elapsed_ms} ms")
                    if TTS_COMBINE_CHUNKS and len(chunks) > 1:
                        audio_parts.append(part.inline_data.data)
                    else:
                        audio.play_audio(part.inline_data.data)
                else:
                    break
            else:
                logging.warning(f"TTS empty content. response={response}")
                break
        if TTS_COMBINE_CHUNKS and audio_parts:
            combined_audio = b"".join(audio_parts)
            logging.info(f"Playing combined audio ({len(combined_audio)} bytes)...")
            audio.play_audio(combined_audio)

    threading.Thread(target=process_tts).start()

def main():
    global gemini
    try:
        gemini = GeminiClient()
        logging.info("Gemini Client Initialized.")
    except Exception as e:
        logging.critical(f"Setup failed: {e}")
        return

    if WARMUP_TTS:
        threading.Thread(target=gemini.warmup_tts, daemon=True).start()

    logging.info("Hotkeys registered:")
    logging.info("  - Dictation: Ctrl + Alt + D")
    logging.info("  - Read Aloud: Ctrl + Alt + R")
    
    # Register Hotkeys using pynput
    # pynput hotkeys are cleaner and often more stable on macOS
    
    # Use standard Listener instead of GlobalHotKeys to avoid the "injected" argument bug
    # This gives us raw control over the callback signature
    
    current_keys = set()

    def on_press(key):
        nonlocal current_keys
        try:
            # Add key to set
            if hasattr(key, 'vk'):
                current_keys.add(key.vk)
            else:
                current_keys.add(key)
                
            # Check for Dictation: Ctrl (cmd on mac usually maps differently, but let's stick to standard) + Alt + D
            # On macOS pynput: 
            # Ctrl -> Key.ctrl 
            # Option/Alt -> Key.alt
            # Command -> Key.cmd
            
            # Helper to check combinations
            def check_combo(keys_needed):
                return all(k in current_keys for k in keys_needed)

            # Dictation: Ctrl + Option + D
            # Note: exact key codes can vary, checking simplified logic
            if key == keyboard.KeyCode(char='d') and \
               (keyboard.Key.ctrl in current_keys or keyboard.Key.ctrl_l in current_keys) and \
               (keyboard.Key.alt in current_keys or keyboard.Key.alt_l in current_keys):
                   handle_dictation()
                   
            # Read Aloud: Ctrl + Option + R
            if key == keyboard.KeyCode(char='r') and \
               (keyboard.Key.ctrl in current_keys or keyboard.Key.ctrl_l in current_keys) and \
               (keyboard.Key.alt in current_keys or keyboard.Key.alt_l in current_keys):
                   handle_read_aloud()
                   
        except Exception as e:
            logging.error(f"Key handler error: {e}")

    def on_release(key):
        nonlocal current_keys
        try:
            if hasattr(key, 'vk'):
                current_keys.discard(key.vk)
            else:
                current_keys.discard(key)
        except:
            pass

    # Start the listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    logging.info("Agent is running. Press Esc to exit.")
    
    # Keep main thread alive
    try:
        listener.join()
    except KeyboardInterrupt:
        logging.info("Stopping agent...")
        listener.stop()
    except Exception as e:
        logging.error(f"Global listener error: {e}")
        listener.stop()

if __name__ == "__main__":
    main()
