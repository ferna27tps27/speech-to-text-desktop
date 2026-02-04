import os
import sys
import time
import threading
import queue
import logging
import asyncio
import pyaudio
import pyperclip
import pyautogui
from pynput import keyboard
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Optional: ElevenLabs for faster TTS (free tier: 10,000 chars/month)
try:
    from elevenlabs.client import ElevenLabs as ElevenLabsClient
    from elevenlabs import stream as elevenlabs_stream
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

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

# Available TTS voices (30 options):
# Bright: Zephyr, Autonoe
# Upbeat: Puck, Laomedeia
# Informative: Charon, Rasalgethi
# Firm: Kore, Orus, Alnilam
# Excitable: Fenrir
# Youthful: Leda
# Breezy: Aoede
# Easy-going: Callirrhoe, Umbriel
# Breathy: Enceladus
# Clear: Iapetus, Erinome
# Smooth: Algieba, Despina
# Gravelly: Algenib
# Soft: Achernar
# Even: Schedar
# Mature: Gacrux
# Forward: Pulcherrima
# Friendly: Achird
# Casual: Zubenelgenubi
# Gentle: Vindemiatrix
# Lively: Sadachbia
# Knowledgeable: Sadaltager
# Warm: Sulafat
VOICE_NAME = os.getenv("VOICE_NAME", "Kore")

# TTS style/direction for controllable speech generation
# Examples: "naturally", "cheerfully", "in a calm tone", "with enthusiasm", 
#           "slowly and clearly", "in a professional tone", "warmly"
# Set to empty string to disable style prompting
TTS_STYLE = os.getenv("TTS_STYLE", "naturally and clearly")

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

# Live API model for real-time dictation (much lower latency)
LIVE_STT_MODEL = os.getenv("LIVE_STT_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
USE_LIVE_API = os.getenv("USE_LIVE_API", "true").lower() in ("1", "true", "yes")

# Live API VAD configuration
# silence_duration_ms: how long to wait after speech stops before ending turn (default 1000ms)
# Lower = faster response, but may cut off if you pause mid-sentence
LIVE_SILENCE_DURATION_MS = int(os.getenv("LIVE_SILENCE_DURATION_MS", "800"))

# ElevenLabs TTS (much faster than Gemini TTS - ~240ms vs 20-60s)
# Free tier: 10,000 characters/month
# Set USE_ELEVENLABS_TTS=true and provide ELEVENLABS_API_KEY to use
USE_ELEVENLABS_TTS = os.getenv("USE_ELEVENLABS_TTS", "false").lower() in ("1", "true", "yes")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
# Fastest model with lowest latency
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
# Voice ID - default is "George" (deep, warm). Find more at elevenlabs.io/voices
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

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
        # Configure client with retry options for network resilience
        # TTS can take 30+ seconds for long text, so use 60 second timeout
        self.client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(
                timeout=60000,  # 60 second timeout for TTS
            )
        )
        # Retry config for individual requests (handles transient 500 errors)
        self.retry_config = types.HttpRetryOptions(
            initial_delay=1.0,  # Start with 1 second delay
            max_delay=10.0,     # Max 10 second delay
            attempts=3          # Try up to 3 times
        )

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
                    temperature=0.0,  # Low temperature for factual transcription
                    http_options=types.HttpOptions(
                        retry_options=self.retry_config
                    )
                )
            )
            return response.text
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return None

    def _format_tts_prompt(self, text):
        """Format text with controllable TTS style direction."""
        if TTS_STYLE and TTS_STYLE.strip():
            return f"Read the following {TTS_STYLE}:\n\n{text}"
        return text

    def text_to_speech_stream(self, text):
        logging.info("Requesting TTS Stream from Gemini...")
        try:
            # Use controllable prompt for better TTS quality
            prompt = self._format_tts_prompt(text)
            
            response = self.client.models.generate_content_stream(
                model=TTS_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=VOICE_NAME
                            )
                        )
                    ),
                    http_options=types.HttpOptions(
                        retry_options=self.retry_config
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
            # Use controllable prompt for better TTS quality
            prompt = self._format_tts_prompt(text)
            
            response = self.client.models.generate_content(
                model=TTS_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=VOICE_NAME
                            )
                        )
                    ),
                    http_options=types.HttpOptions(
                        retry_options=self.retry_config
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


class LiveDictationSession:
    """Real-time dictation using Gemini Live API with input_audio_transcription.
    
    This is much faster than batch transcription because:
    1. Audio is streamed in real-time (no waiting to record everything first)
    2. Transcription happens as you speak
    3. VAD automatically detects when you stop speaking
    """
    
    def __init__(self, client):
        self.client = client
        self.transcription_parts = []
        self.is_recording = False
        self.stop_requested = False
        self._loop = None
        self._session_thread = None
        self._audio_queue = None
        self._result_queue = queue.Queue()  # Thread-safe queue for result
        self._pyaudio = None
        
    async def _run_live_session(self, pyaudio_instance):
        """Main async loop that runs the entire Live API session."""
        self._audio_queue = asyncio.Queue()
        self.transcription_parts = []
        
        # Use dict-based config as per Live API documentation
        # Note: Native audio model requires AUDIO response modality, but we only
        # use the input_audio_transcription for dictation (ignoring the audio response)
        config = {
            "response_modalities": ["AUDIO"],  # Required for native audio model
            "input_audio_transcription": {},   # Get transcription of user's speech
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "disabled": False,
                    "silence_duration_ms": LIVE_SILENCE_DURATION_MS,
                }
            }
        }
        
        try:
            async with self.client.aio.live.connect(
                model=LIVE_STT_MODEL,
                config=config
            ) as session:
                logging.info("Live API session connected - streaming audio...")
                
                # Start audio capture task
                audio_task = asyncio.create_task(
                    self._capture_and_stream_audio(session, pyaudio_instance)
                )
                
                # Receive transcriptions until turn complete or stop requested
                # The API sends incremental transcription chunks that should be concatenated directly
                try:
                    async for response in session.receive():
                        if response.server_content:
                            # Collect input transcription (incremental, concatenate directly)
                            if response.server_content.input_transcription:
                                text = response.server_content.input_transcription.text
                                if text:
                                    self.transcription_parts.append(text)
                                    # Show running transcription
                                    current_text = "".join(self.transcription_parts)
                                    logging.info(f"[Live] {current_text}")
                            
                            # Check if turn is complete
                            if response.server_content.turn_complete:
                                logging.info("VAD detected end of speech")
                                break
                        
                        # Also check if user manually stopped
                        if self.stop_requested and not self.is_recording:
                            try:
                                await session.send_realtime_input(audio_stream_end=True)
                            except Exception:
                                pass  # Session may already be closing
                            # Give a moment for final transcriptions
                            await asyncio.sleep(0.5)
                            break
                            
                except Exception as e:
                    logging.error(f"Error receiving: {e}")
                finally:
                    audio_task.cancel()
                    try:
                        await audio_task
                    except asyncio.CancelledError:
                        pass
                        
        except Exception as e:
            logging.error(f"Live API session error: {e}")
        
        # Put result in thread-safe queue
        # Chunks are incremental and already have proper spacing, concatenate directly
        full_text = "".join(self.transcription_parts).strip()
        self._result_queue.put(full_text if full_text else None)
    
    async def _capture_and_stream_audio(self, session, pyaudio_instance):
        """Capture audio from mic and stream to Live API."""
        stream = None
        try:
            stream = pyaudio_instance.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE_IN,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            while self.is_recording:
                # Read audio in executor to not block event loop
                data = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: stream.read(CHUNK, exception_on_overflow=False)
                )
                
                if data and self.is_recording:
                    await session.send_realtime_input(
                        audio={"data": data, "mime_type": f"audio/pcm;rate={RATE_IN}"}
                    )
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error(f"Audio streaming error: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            # Signal end of audio
            try:
                await session.send_realtime_input(audio_stream_end=True)
            except Exception:
                pass
    
    def _session_thread_func(self, pyaudio_instance):
        """Thread function that runs the async event loop."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._run_live_session(pyaudio_instance))
        except Exception as e:
            logging.error(f"Session thread error: {e}")
            self._result_queue.put(None)
        finally:
            if self._loop:
                self._loop.close()
                self._loop = None
    
    def start(self, pyaudio_instance):
        """Start the live dictation session (non-blocking)."""
        self.is_recording = True
        self.stop_requested = False
        self._pyaudio = pyaudio_instance
        self._result_queue = queue.Queue()
        
        # Start the session in a background thread
        self._session_thread = threading.Thread(
            target=self._session_thread_func,
            args=(pyaudio_instance,),
            daemon=True
        )
        self._session_thread.start()
        logging.info("Live dictation session started")
    
    def stop(self):
        """Signal to stop recording."""
        self.is_recording = False
        self.stop_requested = True
        logging.info("Live dictation stop requested")
    
    def get_result(self, timeout=5.0):
        """Wait for and return the transcription result."""
        try:
            result = self._result_queue.get(timeout=timeout)
            return result
        except queue.Empty:
            # If queue is empty but we have accumulated text, return it
            # This handles cases where the session didn't close cleanly
            if self.transcription_parts:
                full_text = "".join(self.transcription_parts).strip()
                if full_text:
                    logging.info("Returning accumulated transcription (session didn't close cleanly)")
                    return full_text
            logging.warning("Timeout waiting for Live API result")
            return None
        finally:
            # Clean up
            if self._session_thread:
                self._session_thread.join(timeout=1.0)
                self._session_thread = None


class ElevenLabsTTS:
    """Fast TTS using ElevenLabs API (~240ms latency vs Gemini's 20-60s)."""
    
    def __init__(self, api_key):
        if not ELEVENLABS_AVAILABLE:
            raise ImportError("elevenlabs package not installed")
        self.client = ElevenLabsClient(api_key=api_key)
        self.p = pyaudio.PyAudio()
    
    def speak(self, text):
        """Convert text to speech and play it with streaming."""
        try:
            request_start = time.monotonic()
            
            # Use streaming for lowest latency
            audio_stream = self.client.text_to_speech.stream(
                text=text,
                voice_id=ELEVENLABS_VOICE_ID,
                model_id=ELEVENLABS_MODEL,
                optimize_streaming_latency=4,  # Max optimization
                output_format="pcm_24000"  # Raw PCM for direct playback
            )
            
            # Open audio stream for playback
            stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True
            )
            
            first_chunk = True
            for chunk in audio_stream:
                if isinstance(chunk, bytes):
                    if first_chunk:
                        latency_ms = int((time.monotonic() - request_start) * 1000)
                        logging.info(f"ElevenLabs first audio in {latency_ms}ms")
                        first_chunk = False
                    stream.write(chunk)
            
            stream.stop_stream()
            stream.close()
            logging.info("ElevenLabs playback finished")
            
        except Exception as e:
            logging.error(f"ElevenLabs TTS failed: {e}")
            raise


# Global ElevenLabs instance
elevenlabs_tts = None


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
live_session = None  # For Live API dictation

def _paste_transcription(text):
    """Copy text to clipboard and paste to frontmost app."""
    text = _dedupe_repeated_text(text)
    logging.info(f"Typing: {text}")
    
    try:
        pyperclip.copy(text + " ")
        time.sleep(0.1)  # Tiny delay to ensure clipboard is ready
        
        front_app = _get_frontmost_app_name()
        if front_app:
            logging.info(f"Frontmost app: {front_app}")
        _paste_to_frontmost_app()
        _paste_to_terminal(text)
        
    except Exception as e:
        logging.error(f"Paste failed: {e}")

def handle_dictation():
    """Toggle recording state. Uses Live API for real-time transcription if enabled."""
    global audio, gemini, live_session
    
    if not gemini:
        logging.error("Gemini client not initialized (check API Key).")
        return

    # Check if we're using Live API mode
    if USE_LIVE_API and live_session:
        _handle_live_dictation()
    else:
        _handle_batch_dictation()

def _handle_live_dictation():
    """Handle dictation using the Live API (real-time, lower latency)."""
    global audio, live_session, gemini
    
    if live_session.is_recording:
        # Stop recording and get transcription
        audio.play_system_sound("stop")
        logging.info("Stopping live dictation...")
        
        def process_live_transcription():
            start_time = time.monotonic()
            live_session.stop()
            text = live_session.get_result(timeout=5.0)
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            
            if text:
                logging.info(f"Live transcription completed in {elapsed_ms}ms")
                _paste_transcription(text)
            else:
                logging.warning("No transcription from Live API")
                audio.play_system_sound("error")
        
        threading.Thread(target=process_live_transcription).start()
        
    else:
        # Start Live API session (audio capture happens inside the session)
        audio.play_system_sound("start")
        live_session.start(audio.p)
        logging.info("Live dictation started - speak now...")

def _handle_batch_dictation():
    """Handle dictation using batch API (original method)."""
    global audio, gemini
    
    if audio.recording:
        # Stop and Transcribe
        raw_audio = audio.stop_recording()
        
        def process_transcription():
            text = gemini.transcribe(raw_audio)
            if text:
                _paste_transcription(text)
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
    old_clipboard = pyperclip.paste()
    pyperclip.copy("") 
    
    # Use pynput to simulate Cmd+C (Hardware level simulation)
    # This bypasses AppleScript/System Events permission issues
    controller = keyboard.Controller()
    with controller.pressed(keyboard.Key.cmd):
        controller.press('c')
        controller.release('c')

    # Give Chrome a moment to process the copy command before polling
    time.sleep(0.15)
    
    # Wait for clipboard to update (poll for up to 2 seconds for slower apps like Chrome)
    text = ""
    start = time.monotonic()
    poll_count = 0
    while time.monotonic() - start < 2.0:
        text = pyperclip.paste()
        poll_count += 1
        if text and text.strip() and text != old_clipboard:
            logging.info(f"Clipboard updated after {poll_count} polls ({int((time.monotonic() - start) * 1000)}ms)")
            break
        time.sleep(0.05)

    if not text or not text.strip():
        logging.warning(f"No text selected or clipboard empty after {poll_count} polls. Last clipboard content: '{text[:100] if text else 'EMPTY'}'")
        return

    text = text.strip()
    if not text:
        logging.warning("Clipboard text was only whitespace.")
        return

    logging.info(f"Reading: {text[:50]}...")
    
    def process_tts():
        global elevenlabs_tts
        
        # Use ElevenLabs if enabled (much faster: ~240ms vs 20-60s)
        if USE_ELEVENLABS_TTS and elevenlabs_tts:
            try:
                elevenlabs_tts.speak(text)
                return
            except Exception as e:
                logging.warning(f"ElevenLabs failed, falling back to Gemini: {e}")
        
        # Fallback to Gemini TTS (slower but no API key needed)
        # Split long text into chunks for faster time-to-first-audio
        chunks = _split_text_for_tts(text, max_chars=500)
        
        if not chunks:
            logging.warning("No text to read.")
            return
        
        logging.info(f"Processing {len(chunks)} Gemini TTS chunk(s)...")
        
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            request_start = time.monotonic()
            response_stream = gemini.text_to_speech_stream(chunk)
            
            if response_stream:
                audio.play_audio_stream(response_stream, request_start if idx == 0 else None)
            else:
                logging.warning(f"TTS stream request failed for chunk {idx+1}")
                audio.play_system_sound("error")
                break

    threading.Thread(target=process_tts).start()

def main():
    global gemini, live_session, elevenlabs_tts
    try:
        gemini = GeminiClient()
        logging.info("Gemini Client Initialized.")
    except Exception as e:
        logging.critical(f"Setup failed: {e}")
        return

    # Initialize Live API session for real-time dictation
    if USE_LIVE_API:
        try:
            live_session = LiveDictationSession(gemini.client)
            logging.info(f"Live API dictation enabled (model: {LIVE_STT_MODEL})")
            logging.info(f"  - Silence detection: {LIVE_SILENCE_DURATION_MS}ms")
        except Exception as e:
            logging.warning(f"Live API init failed, using batch mode: {e}")
            live_session = None
    else:
        logging.info("Live API disabled, using batch transcription mode")

    # Initialize ElevenLabs TTS if enabled (much faster than Gemini)
    if USE_ELEVENLABS_TTS:
        if not ELEVENLABS_AVAILABLE:
            logging.warning("ElevenLabs requested but package not installed. Run: pip install elevenlabs")
        elif not ELEVENLABS_API_KEY:
            logging.warning("ElevenLabs requested but ELEVENLABS_API_KEY not set in .env")
        else:
            try:
                elevenlabs_tts = ElevenLabsTTS(ELEVENLABS_API_KEY)
                logging.info(f"ElevenLabs TTS enabled (model: {ELEVENLABS_MODEL})")
                logging.info(f"  - Expected latency: ~240ms (vs Gemini's 20-60s)")
            except Exception as e:
                logging.warning(f"ElevenLabs init failed: {e}")
    
    if not USE_ELEVENLABS_TTS or not elevenlabs_tts:
        logging.info("Using Gemini TTS (slower, but no additional API key needed)")
        if WARMUP_TTS:
            threading.Thread(target=gemini.warmup_tts, daemon=True).start()

    logging.info("Hotkeys registered:")
    logging.info("  - Dictation: Ctrl + Alt + D")
    logging.info("  - Read Aloud: Ctrl + Alt + R")
    if USE_LIVE_API and live_session:
        logging.info("  (Using Live API for real-time dictation)")
    
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
