"""Tests for voice_agent.audio."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from voice_agent.config import Config


class TestAudioHandler:
    @patch("voice_agent.audio.pyaudio.PyAudio")
    def test_initialization(self, mock_pa_cls: MagicMock, mock_config: Config) -> None:
        from voice_agent.audio import AudioHandler

        handler = AudioHandler(mock_config)
        assert handler.is_recording is False
        mock_pa_cls.assert_called_once()

    @patch("voice_agent.audio.pyaudio.PyAudio")
    def test_context_manager_cleanup(
        self, mock_pa_cls: MagicMock, mock_config: Config
    ) -> None:
        from voice_agent.audio import AudioHandler

        instance = MagicMock()
        mock_pa_cls.return_value = instance

        with AudioHandler(mock_config) as handler:
            assert handler is not None

        instance.terminate.assert_called_once()

    @patch("voice_agent.audio.pyaudio.PyAudio")
    def test_pyaudio_instance_property(
        self, mock_pa_cls: MagicMock, mock_config: Config
    ) -> None:
        from voice_agent.audio import AudioHandler

        instance = MagicMock()
        mock_pa_cls.return_value = instance

        handler = AudioHandler(mock_config)
        assert handler.pyaudio_instance is instance
