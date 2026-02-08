"""Text-to-speech providers.

Provides two implementations behind a common TTSProvider protocol:
- GeminiTTSProvider: Uses Gemini's TTS model (slower, no extra API key).
- ElevenLabsTTSProvider: Uses ElevenLabs API (~240ms latency).
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

import pyaudio
from google import genai
from google.genai import types

from .audio import AudioHandler
from .config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TTSProvider(Protocol):
    """Common interface for text-to-speech implementations."""

    def speak(self, text: str) -> None: ...


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def split_text_for_tts(text: str, max_chars: int = 2000) -> list[str]:
    """Split long text into chunks at sentence/word boundaries for TTS."""
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        cut = text.rfind(".", start, end)
        if cut == -1:
            cut = text.rfind(" ", start, end)
        if cut == -1 or cut <= start:
            cut = end
        parts.append(text[start:cut].strip())
        start = cut
    return [p for p in parts if p]


def format_tts_prompt(text: str, style: str) -> str:
    """Format text with controllable TTS style direction."""
    if style and style.strip():
        return f"Read the following {style}:\n\n{text}"
    return text


# ---------------------------------------------------------------------------
# Gemini TTS Provider
# ---------------------------------------------------------------------------


class GeminiTTSProvider:
    """TTS using Google Gemini's native TTS model.

    Slower (~20-60s time-to-first-audio) but requires no additional API key
    beyond the Gemini key. Supports streaming playback and text chunking.
    """

    def __init__(
        self, client: genai.Client, audio: AudioHandler, config: Config
    ) -> None:
        self._client = client
        self._audio = audio
        self._config = config
        self._retry_config = types.HttpRetryOptions(
            initial_delay=1.0,
            max_delay=10.0,
            attempts=3,
        )

    def speak(self, text: str) -> None:
        """Convert text to speech and play it with streaming."""
        chunks = split_text_for_tts(text, max_chars=500)
        if not chunks:
            logger.warning("No text to read.")
            return

        logger.info(f"Processing {len(chunks)} Gemini TTS chunk(s)...")

        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            request_start = time.monotonic()
            response_stream = self._request_stream(chunk)

            if response_stream:
                self._audio.play_audio_stream(
                    response_stream,
                    request_start if idx == 0 else None,
                )
            else:
                logger.warning(f"TTS stream request failed for chunk {idx + 1}")
                break

    def warmup(self) -> None:
        """Optional warmup call to reduce first-request latency."""
        try:
            response = self._client.models.generate_content_stream(
                model=self._config.tts_model,
                contents=[" "],
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self._config.voice_name,
                            )
                        )
                    ),
                ),
            )
            for _ in response:
                break
            logger.info("TTS warmup completed.")
        except Exception as e:
            logger.warning(f"TTS warmup failed: {e}")

    def text_to_speech(self, text: str) -> object | None:
        """Non-streaming TTS (kept for backward compatibility / testing)."""
        logger.info("Requesting TTS from Gemini...")
        try:
            prompt = format_tts_prompt(text, self._config.tts_style)
            response = self._client.models.generate_content(
                model=self._config.tts_model,
                contents=prompt,
                config=self._speech_config(),
            )
            return response
        except Exception as e:
            logger.error(f"TTS Request failed: {e}")
            return None

    def text_to_speech_stream(self, text: str) -> object | None:
        """Streaming TTS (kept for backward compatibility / testing)."""
        return self._request_stream(text)

    def _request_stream(self, text: str) -> object | None:
        logger.info("Requesting TTS Stream from Gemini...")
        try:
            prompt = format_tts_prompt(text, self._config.tts_style)
            response = self._client.models.generate_content_stream(
                model=self._config.tts_model,
                contents=[prompt],
                config=self._speech_config(),
            )
            return response
        except Exception as e:
            logger.error(f"TTS Stream Request failed: {e}")
            return None

    def _speech_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self._config.voice_name,
                    )
                )
            ),
            http_options=types.HttpOptions(
                retry_options=self._retry_config,
            ),
        )


# ---------------------------------------------------------------------------
# ElevenLabs TTS Provider
# ---------------------------------------------------------------------------


class ElevenLabsTTSProvider:
    """Fast TTS using ElevenLabs API (~240ms latency vs Gemini's 20-60s).

    Free tier: 10,000 characters/month.
    """

    def __init__(self, config: Config) -> None:
        try:
            from elevenlabs.client import ElevenLabs as ElevenLabsClient
        except ImportError as e:
            raise ImportError(
                "elevenlabs package not installed. Run: pip install elevenlabs"
            ) from e

        if not config.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY not set in .env")

        self._config = config
        self._client = ElevenLabsClient(api_key=config.elevenlabs_api_key)
        self._p = pyaudio.PyAudio()

    def speak(self, text: str) -> None:
        """Convert text to speech and play it with streaming."""
        try:
            request_start = time.monotonic()

            audio_stream = self._client.text_to_speech.stream(
                text=text,
                voice_id=self._config.elevenlabs_voice_id,
                model_id=self._config.elevenlabs_model,
                optimize_streaming_latency=4,
                output_format="pcm_24000",
            )

            stream = self._p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True,
            )

            first_chunk = True
            for chunk in audio_stream:
                if isinstance(chunk, bytes):
                    if first_chunk:
                        latency_ms = int(
                            (time.monotonic() - request_start) * 1000
                        )
                        logger.info(f"ElevenLabs first audio in {latency_ms}ms")
                        first_chunk = False
                    stream.write(chunk)

            stream.stop_stream()
            stream.close()
            logger.info("ElevenLabs playback finished")

        except Exception as e:
            logger.error(f"ElevenLabs TTS failed: {e}")
            raise
