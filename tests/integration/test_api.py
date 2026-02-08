"""Integration tests that hit the real Gemini API.

These require a valid GEMINI_API_KEY in .env and are not meant
to run in CI without credentials.

Run with: pytest tests/integration/ -v
"""

from __future__ import annotations

import os
import sys

import pytest
import numpy as np

# Skip all tests in this module if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set; skipping integration tests",
)

from voice_agent.config import Config


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.from_env()


@pytest.fixture(scope="module")
def gemini_client(config: Config):
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=config.gemini_api_key,
        http_options=types.HttpOptions(timeout=60000),
    )


class TestGeminiAPI:
    def test_client_initialization(self, gemini_client) -> None:
        """Verify we can create a Gemini client with the API key."""
        assert gemini_client is not None

    def test_tts_streaming(self, gemini_client, config: Config) -> None:
        """Verify TTS streaming returns audio chunks."""
        from voice_agent.tts import GeminiTTSProvider
        from voice_agent.audio import AudioHandler

        with AudioHandler(config) as audio:
            tts = GeminiTTSProvider(gemini_client, audio, config)
            stream = tts.text_to_speech_stream("Hello, this is a test.")
            assert stream is not None

            chunk_count = 0
            for chunk in stream:
                chunk_count += 1
                if chunk_count >= 3:
                    break
            assert chunk_count > 0

    def test_stt_transcription(self, gemini_client, config: Config) -> None:
        """Verify STT can process a synthetic sine wave without errors."""
        from voice_agent.stt import add_wav_header
        from google.genai import types

        # Generate 1s of 440Hz sine wave
        rate = 16000
        t = np.linspace(0, 1, rate, endpoint=False)
        pcm = (np.sin(440 * 2 * np.pi * t) * 32767).astype(np.int16).tobytes()
        wav_data = add_wav_header(pcm, rate=rate)

        response = gemini_client.models.generate_content(
            model=config.stt_model,
            contents=[
                types.Part.from_bytes(data=wav_data, mime_type="audio/wav"),
                "Transcribe the audio exactly as spoken.",
            ],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT"],
                temperature=0.0,
            ),
        )
        # Just verify it didn't raise; content of transcription of a
        # sine wave is unpredictable
        assert response is not None
