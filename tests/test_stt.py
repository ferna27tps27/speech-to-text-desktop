"""Tests for voice_agent.stt utilities."""

from __future__ import annotations

from voice_agent.stt import add_wav_header, dedupe_repeated_text


class TestAddWavHeader:
    def test_produces_wav_bytes(self) -> None:
        pcm = b"\x00" * 3200  # 0.1s of silence at 16kHz
        wav = add_wav_header(pcm, rate=16000)
        # WAV files start with RIFF header
        assert wav[:4] == b"RIFF"
        assert b"WAVE" in wav[:12]

    def test_length_increases(self) -> None:
        pcm = b"\x00" * 100
        wav = add_wav_header(pcm)
        # WAV header adds 44 bytes
        assert len(wav) > len(pcm)


class TestDedupeRepeatedText:
    def test_no_change_for_short_text(self) -> None:
        assert dedupe_repeated_text("hello") == "hello"

    def test_dedupes_exact_repeat(self) -> None:
        text = "this is a test sentence" * 2
        result = dedupe_repeated_text(text)
        assert result == "this is a test sentence"

    def test_no_change_for_unique_text(self) -> None:
        text = "this is a unique sentence that is not repeated at all"
        assert dedupe_repeated_text(text) == text
