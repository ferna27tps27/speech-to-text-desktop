"""Tests for voice_agent.config."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from voice_agent.config import Config


class TestConfigDefaults:
    def test_default_values(self) -> None:
        cfg = Config()
        assert cfg.gemini_api_key == ""
        assert cfg.tts_model == "gemini-2.5-flash-preview-tts"
        assert cfg.stt_model == "gemini-2.5-flash"
        assert cfg.use_live_api is True
        assert cfg.sample_rate_in == 16000
        assert cfg.sample_rate_out == 24000
        assert cfg.voice_name == "Kore"

    def test_frozen(self) -> None:
        cfg = Config()
        with pytest.raises(AttributeError):
            cfg.voice_name = "Puck"  # type: ignore[misc]


class TestConfigFromEnv:
    def test_loads_api_key(self) -> None:
        env = {"GEMINI_API_KEY": "my-test-key"}
        with patch.dict(os.environ, env, clear=False):
            cfg = Config.from_env()
        assert cfg.gemini_api_key == "my-test-key"

    def test_bool_parsing(self) -> None:
        env = {
            "GEMINI_API_KEY": "k",
            "USE_LIVE_API": "false",
            "WARMUP_TTS": "1",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = Config.from_env()
        assert cfg.use_live_api is False
        assert cfg.warmup_tts is True

    def test_int_parsing_fallback(self) -> None:
        env = {
            "GEMINI_API_KEY": "k",
            "LIVE_SILENCE_DURATION_MS": "not-a-number",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = Config.from_env()
        # Should fall back to default
        assert cfg.live_silence_duration_ms == 800

    def test_empty_model_uses_default(self) -> None:
        env = {"GEMINI_API_KEY": "k", "TTS_MODEL": ""}
        with patch.dict(os.environ, env, clear=False):
            cfg = Config.from_env()
        assert cfg.tts_model == "gemini-2.5-flash-preview-tts"
