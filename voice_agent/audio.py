"""Audio I/O handler using PyAudio.

Manages microphone input recording and audio output playback with
proper resource management via context manager protocol.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterator

import pyaudio

from .config import Config

logger = logging.getLogger(__name__)

# PyAudio format constants
FORMAT = pyaudio.paInt16
CHANNELS = 1


class AudioHandler:
    """Manages audio recording and playback via PyAudio.

    Use as a context manager for proper resource cleanup::

        with AudioHandler(config) as audio:
            audio.start_recording()
            ...
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._p: pyaudio.PyAudio | None = None
        self._recording = False
        self._frames: list[bytes] = []
        self._stream: Any | None = None
        self._init_pyaudio()

    def _init_pyaudio(self) -> None:
        """Initialize or re-initialize the PyAudio instance."""
        if self._p is None:
            self._p = pyaudio.PyAudio()

    @property
    def pyaudio_instance(self) -> pyaudio.PyAudio:
        """Expose the PyAudio instance for components that need direct access."""
        self._init_pyaudio()
        assert self._p is not None
        return self._p

    @property
    def is_recording(self) -> bool:
        return self._recording

    def __enter__(self) -> AudioHandler:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """Release all PyAudio resources."""
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._p:
            try:
                self._p.terminate()
            except Exception:
                pass
            self._p = None

    def start_recording(self) -> None:
        """Open the microphone and begin recording in a background thread."""
        self._recording = True
        self._frames = []
        try:
            self._init_pyaudio()
            assert self._p is not None

            self._stream = self._p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self._config.sample_rate_in,
                input=True,
                frames_per_buffer=self._config.chunk_size,
            )
            logger.info("Microphone input started.")

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

    def stop_recording(self) -> bytes:
        """Stop recording and return the captured PCM audio bytes."""
        self._recording = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        logger.info("Microphone input stopped.")
        return b"".join(self._frames)

    def play_audio_stream(
        self,
        response_stream: Iterator[Any],
        request_start_time: float | None = None,
    ) -> None:
        """Play audio chunks from a Gemini TTS streaming response as they arrive."""
        if not response_stream:
            return

        try:
            logger.info("Starting stream playback...")
            self._init_pyaudio()
            assert self._p is not None

            stream = self._p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self._config.sample_rate_out,
                output=True,
            )

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
                                latency_ms = int(
                                    (time.monotonic() - request_start_time) * 1000
                                )
                                logger.info(
                                    f"First audio chunk played in {latency_ms} ms"
                                )
                            else:
                                logger.info("First audio chunk played (Latency check)")

            stream.stop_stream()
            stream.close()
            logger.info(f"Stream playback finished. Total chunks: {chunk_count}")
        except Exception as e:
            logger.error(f"Stream playback failed: {e}")

    def play_pcm(self, audio_data: bytes) -> None:
        """Play raw PCM audio data."""
        if not audio_data:
            return
        try:
            logger.info(f"Playing audio ({len(audio_data)} bytes)...")
            self._init_pyaudio()
            assert self._p is not None

            stream = self._p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self._config.sample_rate_out,
                output=True,
            )
            stream.write(audio_data)
            stream.stop_stream()
            stream.close()
            logger.info("Playback finished.")
        except Exception as e:
            logger.error(f"Failed to play audio: {e}")
