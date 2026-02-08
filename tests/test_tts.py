"""Tests for voice_agent.tts utilities."""

from __future__ import annotations

from voice_agent.tts import split_text_for_tts, format_tts_prompt


class TestSplitTextForTts:
    def test_short_text_not_split(self) -> None:
        text = "Hello world"
        assert split_text_for_tts(text, max_chars=100) == [text]

    def test_splits_at_sentence_boundary(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        chunks = split_text_for_tts(text, max_chars=30)
        assert len(chunks) >= 2
        # Each chunk should be non-empty
        for chunk in chunks:
            assert len(chunk) > 0

    def test_splits_at_word_boundary(self) -> None:
        text = "word " * 50  # 250 chars of words
        chunks = split_text_for_tts(text.strip(), max_chars=50)
        assert len(chunks) >= 2

    def test_empty_text(self) -> None:
        assert split_text_for_tts("", max_chars=100) == [""]

    def test_zero_max_chars_returns_whole(self) -> None:
        text = "Hello world"
        assert split_text_for_tts(text, max_chars=0) == [text]


class TestFormatTtsPrompt:
    def test_with_style(self) -> None:
        result = format_tts_prompt("hello", "cheerfully")
        assert result == "Read the following cheerfully:\n\nhello"

    def test_empty_style_returns_plain(self) -> None:
        assert format_tts_prompt("hello", "") == "hello"

    def test_whitespace_style_returns_plain(self) -> None:
        assert format_tts_prompt("hello", "   ") == "hello"
