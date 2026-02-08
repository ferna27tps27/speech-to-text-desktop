"""Speech-to-text transcription via Gemini.

Provides two implementations behind a common Transcriber protocol:
- BatchTranscriber: Records all audio then transcribes in one shot.
- LiveTranscriber: Streams audio in real-time via the Gemini Live API.
"""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import threading
import wave
from typing import Protocol

import pyaudio
from google.genai import types

from .config import Config

logger = logging.getLogger(__name__)

# PyAudio format constants (must match audio.py)
FORMAT = pyaudio.paInt16
CHANNELS = 1


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Transcriber(Protocol):
    """Common interface for speech-to-text implementations."""

    @property
    def is_active(self) -> bool: ...

    def start(self, pyaudio_instance: pyaudio.PyAudio) -> None: ...

    def stop(self) -> None: ...

    def get_result(self, timeout: float = 5.0) -> str | None: ...


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def add_wav_header(
    pcm_data: bytes, rate: int = 16000, channels: int = 1, sampwidth: int = 2
) -> bytes:
    """Wrap raw PCM bytes in a WAV header."""
    with io.BytesIO() as wav_file:
        with wave.open(wav_file, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(rate)
            wf.writeframes(pcm_data)
        return wav_file.getvalue()


def dedupe_repeated_text(text: str) -> str:
    """Collapse obvious duplicated transcriptions."""
    cleaned = " ".join(text.split())
    if len(cleaned) < 20:
        return text
    half = len(cleaned) // 2
    if len(cleaned) % 2 == 0 and cleaned[:half] == cleaned[half:]:
        return cleaned[:half]
    if (
        cleaned.count(cleaned[:half]) >= 2
        and cleaned.startswith(cleaned[:half] + " " + cleaned[:half])
    ):
        return cleaned[:half]
    return text


# ---------------------------------------------------------------------------
# Batch Transcriber
# ---------------------------------------------------------------------------


class BatchTranscriber:
    """Records all audio first, then transcribes in a single API call.

    Uses Gemini's generate_content endpoint with audio input.
    """

    def __init__(self, client: object, config: Config) -> None:
        self._client = client  # genai.Client
        self._config = config
        self._recording = False
        self._frames: list[bytes] = []
        self._stream: object | None = None
        self._result_queue: queue.Queue[str | None] = queue.Queue()
        self._retry_config = types.HttpRetryOptions(
            initial_delay=1.0,
            max_delay=10.0,
            attempts=3,
        )

    @property
    def is_active(self) -> bool:
        return self._recording

    def start(self, pyaudio_instance: pyaudio.PyAudio) -> None:
        """Start recording audio from the microphone."""
        self._recording = True
        self._frames = []
        self._result_queue = queue.Queue()
        try:
            self._stream = pyaudio_instance.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self._config.sample_rate_in,
                input=True,
                frames_per_buffer=self._config.chunk_size,
            )
            logger.info("Microphone input started (batch mode).")
            threading.Thread(target=self._record_loop, daemon=True).start()
        except Exception as e:
            logger.error(f"Failed to open microphone: {e}")
            self._recording = False

    def _record_loop(self) -> None:
        while self._recording and self._stream:
            try:
                data = self._stream.read(
                    self._config.chunk_size, exception_on_overflow=False
                )
                self._frames.append(data)
            except Exception as e:
                logger.error(f"Error recording stream: {e}")
                break

    def stop(self) -> None:
        """Stop recording and begin transcription in background."""
        self._recording = False
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        logger.info("Microphone input stopped (batch mode).")

        raw_audio = b"".join(self._frames)
        threading.Thread(
            target=self._transcribe_async, args=(raw_audio,), daemon=True
        ).start()

    def _transcribe_async(self, audio_bytes: bytes) -> None:
        """Transcribe audio in a background thread."""
        try:
            result = self._transcribe(audio_bytes)
            self._result_queue.put(result)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self._result_queue.put(None)

    def _transcribe(self, audio_bytes: bytes) -> str | None:
        logger.info("Sending audio to Gemini (batch)...")
        wav_data = add_wav_header(audio_bytes, rate=self._config.sample_rate_in)

        response = self._client.models.generate_content(
            model=self._config.stt_model,
            contents=[
                types.Part.from_bytes(data=wav_data, mime_type="audio/wav"),
                (
                    "Transcribe the audio exactly as spoken, but format "
                    "punctuation commands into their symbols.\n\n"
                    "Rules:\n"
                    '- Replace "comma" with ","\n'
                    '- Replace "dot" or "period" with "."\n'
                    '- Replace "semicolon" with ";"\n'
                    '- Replace "colon" with ":"\n'
                    '- Replace "question mark" with "?"\n'
                    '- Replace "exclamation mark" with "!"\n'
                    '- Replace "dash" or "hyphen" with "-"\n'
                    '- Replace "new line" or "next line" with a line break\n'
                    '- Replace "open parenthesis" with "(" and '
                    '"close parenthesis" with ")"\n'
                    '- Replace "open quote" with \'"\' and '
                    '"close quote" with \'"\'\n\n'
                    "Do not add any other markdown or speaker labels. "
                    "Just the formatted text."
                ),
            ],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT"],
                temperature=0.0,
                http_options=types.HttpOptions(
                    retry_options=self._retry_config,
                ),
            ),
        )
        return response.text

    def get_result(self, timeout: float = 5.0) -> str | None:
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            logger.warning("Timeout waiting for batch transcription result")
            return None


# ---------------------------------------------------------------------------
# Live Transcriber (Gemini Live API)
# ---------------------------------------------------------------------------


class LiveTranscriber:
    """Real-time dictation using Gemini Live API with input_audio_transcription.

    Supports **multi-turn** dictation: when VAD detects a pause, the session
    stays open and continues listening for the next utterance. Text is
    accumulated across all turns until the user manually stops via hotkey.

    This prevents the two problems seen with single-turn mode:
    1. Long dictations getting cut off when the user pauses to think.
    2. Model degradation on very long continuous turns (by giving the model
       natural break points between turns).
    """

    def __init__(self, client: object, config: Config) -> None:
        self._client = client  # genai.Client
        self._config = config
        self._transcription_parts: list[str] = []
        self._recording = False
        self._stop_requested = False
        self._turn_count = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session_thread: threading.Thread | None = None
        self._result_queue: queue.Queue[str | None] = queue.Queue()

    @property
    def is_active(self) -> bool:
        return self._recording

    def start(self, pyaudio_instance: pyaudio.PyAudio) -> None:
        """Start the live dictation session (non-blocking)."""
        self._recording = True
        self._stop_requested = False
        self._transcription_parts = []
        self._turn_count = 0
        self._result_queue = queue.Queue()

        self._session_thread = threading.Thread(
            target=self._session_thread_func,
            args=(pyaudio_instance,),
            daemon=True,
        )
        self._session_thread.start()
        logger.info("Live dictation session started")

    def stop(self) -> None:
        """Signal to stop recording."""
        self._recording = False
        self._stop_requested = True
        logger.info("Live dictation stop requested")

    def get_result(self, timeout: float = 5.0) -> str | None:
        """Wait for and return the transcription result."""
        try:
            result = self._result_queue.get(timeout=timeout)
            return result
        except queue.Empty:
            # Fallback: return accumulated text if session didn't close cleanly
            if self._transcription_parts:
                full_text = "".join(self._transcription_parts).strip()
                if full_text:
                    logger.info(
                        "Returning accumulated transcription "
                        "(session didn't close cleanly)"
                    )
                    return full_text
            logger.warning("Timeout waiting for Live API result")
            return None
        finally:
            if self._session_thread:
                self._session_thread.join(timeout=1.0)
                self._session_thread = None

    # -- internal ------------------------------------------------------------

    def _session_thread_func(self, pyaudio_instance: pyaudio.PyAudio) -> None:
        """Thread function that runs the async event loop."""
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(
                self._run_live_session(pyaudio_instance)
            )
        except Exception as e:
            logger.error(f"Session thread error: {e}")
            self._result_queue.put(None)
        finally:
            if self._loop:
                self._loop.close()
                self._loop = None

    async def _run_live_session(self, pyaudio_instance: pyaudio.PyAudio) -> None:
        """Main async loop for the Live API session.

        Multi-turn: the Gemini Live API's ``session.receive()`` iterator
        terminates after each ``turn_complete`` event. To support multi-turn
        dictation we wrap the receive loop in an outer ``while`` loop that
        re-enters ``session.receive()`` after each turn. The audio capture
        task continues streaming mic data uninterrupted throughout.

        The outer loop only exits when the user presses the stop hotkey.

        The native audio model requires ``response_modalities: ["AUDIO"]``.
        The multi-turn while-loop ensures the model gets fresh context after
        each VAD pause, which mitigates the degradation (gibberish) that
        occurs during very long continuous single-turn speech (~30 s+).
        """
        self._transcription_parts = []
        self._turn_count = 0

        config = {
            "response_modalities": ["AUDIO"],
            "input_audio_transcription": {},
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "disabled": False,
                    "silence_duration_ms": self._config.live_silence_duration_ms,
                }
            },
        }

        try:
            async with self._client.aio.live.connect(
                model=self._config.live_stt_model,
                config=config,
            ) as session:
                logger.info("Live API session connected - streaming audio...")

                audio_task = asyncio.create_task(
                    self._capture_and_stream_audio(session, pyaudio_instance)
                )

                try:
                    # Outer loop: re-enter session.receive() after each
                    # turn_complete so the session stays alive across turns.
                    while not self._stop_requested:
                        async for response in session.receive():
                            # -- user manually stopped via hotkey --
                            if self._stop_requested:
                                try:
                                    await session.send_realtime_input(
                                        audio_stream_end=True
                                    )
                                except Exception:
                                    pass
                                await asyncio.sleep(0.3)
                                break

                            if response.server_content:
                                # Accumulate transcription from user's speech
                                if response.server_content.input_transcription:
                                    text = (
                                        response.server_content.input_transcription.text
                                    )
                                    if text:
                                        self._transcription_parts.append(text)
                                        current_text = "".join(
                                            self._transcription_parts
                                        )
                                        logger.info(f"[Live] {current_text}")

                                # VAD detected silence / turn boundary.
                                # Break out of inner async-for so the outer
                                # while loop calls session.receive() again
                                # for the next turn.
                                if response.server_content.turn_complete:
                                    self._turn_count += 1
                                    self._transcription_parts.append(" ")
                                    logger.info(
                                        f"VAD detected pause (turn {self._turn_count}) "
                                        f"-- re-entering receive loop for next turn..."
                                    )
                                    break  # Break inner async-for; outer while restarts it

                except Exception as e:
                    logger.error(f"Error receiving: {e}")
                finally:
                    audio_task.cancel()
                    try:
                        await audio_task
                    except asyncio.CancelledError:
                        pass

        except Exception as e:
            logger.error(f"Live API session error: {e}")

        full_text = "".join(self._transcription_parts).strip()
        logger.info(
            f"Session ended after {self._turn_count} turn(s), "
            f"{len(full_text)} chars accumulated"
        )
        self._result_queue.put(full_text if full_text else None)

    async def _capture_and_stream_audio(
        self, session: object, pyaudio_instance: pyaudio.PyAudio
    ) -> None:
        """Capture audio from mic and stream to Live API."""
        stream = None
        try:
            stream = pyaudio_instance.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self._config.sample_rate_in,
                input=True,
                frames_per_buffer=self._config.chunk_size,
            )

            while self._recording:
                data = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: stream.read(
                        self._config.chunk_size, exception_on_overflow=False
                    ),
                )

                if data and self._recording:
                    await session.send_realtime_input(
                        audio={
                            "data": data,
                            "mime_type": f"audio/pcm;rate={self._config.sample_rate_in}",
                        }
                    )

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Audio streaming error: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            try:
                await session.send_realtime_input(audio_stream_end=True)
            except Exception:
                pass
