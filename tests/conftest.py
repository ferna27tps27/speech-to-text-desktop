"""Shared test fixtures for the voice_agent test suite."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from voice_agent.config import Config


@pytest.fixture
def mock_config() -> Config:
    """A Config instance with test-safe defaults (no real API keys)."""
    return Config(
        gemini_api_key="test-key-not-real",
        elevenlabs_api_key="",
        tts_model="gemini-2.5-flash-preview-tts",
        stt_model="gemini-2.5-flash",
        live_stt_model="gemini-2.5-flash-native-audio-preview-12-2025",
        use_live_api=False,
        use_elevenlabs_tts=False,
        warmup_tts=False,
        terminal_paste_debug=False,
        sample_rate_in=16000,
        sample_rate_out=24000,
        chunk_size=1024,
        live_silence_duration_ms=800,
        voice_name="Kore",
        tts_style="naturally and clearly",
        elevenlabs_model="eleven_flash_v2_5",
        elevenlabs_voice_id="JBFqnCBsd6RMkjVDRZzb",
        tts_chunk_chars=2000,
    )


@pytest.fixture
def mock_pyaudio():
    """Patch PyAudio so tests don't touch real audio hardware."""
    with patch("pyaudio.PyAudio") as mock_pa:
        instance = MagicMock()
        mock_pa.return_value = instance
        yield instance
