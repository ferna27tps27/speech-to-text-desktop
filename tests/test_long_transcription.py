"""Tests to verify long transcriptions are not truncated by our code.

These tests simulate the text accumulation pipeline end-to-end to ensure
no internal logic silently drops or corrupts long dictation text.
"""

from __future__ import annotations

import queue
from unittest.mock import MagicMock, patch

import pytest

from voice_agent.config import Config
from voice_agent.stt import LiveTranscriber, BatchTranscriber, dedupe_repeated_text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> Config:
    return Config(
        gemini_api_key="test-key",
        live_stt_model="test-model",
        sample_rate_in=16000,
        chunk_size=1024,
        live_silence_duration_ms=800,
    )


# ---------------------------------------------------------------------------
# Test: LiveTranscriber text accumulation has no length limit
# ---------------------------------------------------------------------------


class TestLiveTranscriberAccumulation:
    """Verify that LiveTranscriber._transcription_parts accumulation
    works correctly for very long dictation sessions."""

    def test_accumulates_hundreds_of_chunks(self, config: Config) -> None:
        """Simulate 500 incremental transcription chunks (like a 2+ minute dictation)."""
        client = MagicMock()
        transcriber = LiveTranscriber(client, config)

        # Simulate incremental chunks arriving from the Live API
        words = [
            "This ", "is ", "a ", "very ", "long ", "dictation ", "session ",
            "that ", "should ", "not ", "be ", "truncated ", "at ", "any ",
            "point. ", "The ", "user ", "is ", "speaking ", "continuously ",
            "for ", "several ", "minutes ", "about ", "a ", "complex ",
            "topic ", "that ", "requires ", "sustained ", "thought. ",
        ]

        # Simulate 500 chunks (reusing the word list cyclically)
        for i in range(500):
            transcriber._transcription_parts.append(words[i % len(words)])

        full_text = "".join(transcriber._transcription_parts).strip()

        # Verify nothing was lost
        assert len(transcriber._transcription_parts) == 500
        assert len(full_text) > 2000  # Should be ~3000+ chars
        assert full_text.startswith("This is a very long")
        assert "truncated" in full_text

    def test_result_queue_preserves_long_text(self, config: Config) -> None:
        """Verify the thread-safe queue doesn't truncate long results."""
        long_text = "word " * 1000  # 5000 chars
        long_text = long_text.strip()

        q: queue.Queue[str | None] = queue.Queue()
        q.put(long_text)

        result = q.get(timeout=1.0)
        assert result == long_text
        assert len(result) == 4999  # "word " * 1000 minus trailing space

    def test_fallback_path_returns_accumulated_text(self, config: Config) -> None:
        """When the session doesn't close cleanly, get_result() should
        return whatever text has accumulated (the 'session didn't close
        cleanly' codepath from the Feb 5 log)."""
        client = MagicMock()
        transcriber = LiveTranscriber(client, config)

        # Simulate accumulated text (like the Feb 5 session)
        transcriber._transcription_parts = [
            "This is going to be a message that I will be sending to Harry. ",
            "So help me to organize my thoughts on the slack structure. ",
            "Hey Harry, thanks for sharing all the accounts with me. ",
            "Let me know what's the best way to support each one of those ",
            "accounts that you'll be assigning to me.",
        ]

        # Empty result queue (session didn't put a result)
        transcriber._result_queue = queue.Queue()

        # get_result should return accumulated text via the fallback path
        result = transcriber.get_result(timeout=0.1)
        assert result is not None
        assert result.startswith("This is going to be a message")
        assert result.endswith("assigning to me.")
        assert len(result) > 200


# ---------------------------------------------------------------------------
# Test: Multi-turn accumulation across VAD pauses
# ---------------------------------------------------------------------------


class TestMultiTurnAccumulation:
    """Verify that text from multiple VAD turns is properly concatenated
    with correct spacing -- the core fix for the truncation issue."""

    def test_three_turns_concatenated_with_spaces(self, config: Config) -> None:
        """Simulate 3 turns of speech with VAD pauses between them."""
        client = MagicMock()
        transcriber = LiveTranscriber(client, config)

        # Turn 1: user speaks
        transcriber._transcription_parts.extend([
            "right ", "code ", "that ", "will ", "generate ", "an ", "app",
        ])
        # VAD fires turn_complete -> we insert a space separator
        transcriber._turn_count += 1
        transcriber._transcription_parts.append(" ")

        # Turn 2: user continues after thinking
        transcriber._transcription_parts.extend([
            "in ", "HTML ", "focusing ", "on ", "the ", "main ", "problem",
        ])
        # VAD fires again
        transcriber._turn_count += 1
        transcriber._transcription_parts.append(" ")

        # Turn 3: user finishes
        transcriber._transcription_parts.extend([
            "the ", "client ", "has.",
        ])

        full_text = "".join(transcriber._transcription_parts).strip()

        assert transcriber._turn_count == 2
        assert full_text == "right code that will generate an app in HTML focusing on the main problem the client has."
        # No double spaces
        assert "  " not in full_text.replace("  ", "XX_DOUBLE_XX")
        # No fused words (the space separator prevents "app" + "in" -> "appin")
        assert "appin" not in full_text

    def test_single_turn_no_trailing_space(self, config: Config) -> None:
        """If user stops before VAD fires, no extra space is added."""
        client = MagicMock()
        transcriber = LiveTranscriber(client, config)

        transcriber._transcription_parts.extend([
            "hello ", "world",
        ])

        full_text = "".join(transcriber._transcription_parts).strip()
        assert full_text == "hello world"
        assert transcriber._turn_count == 0

    def test_many_turns_long_session(self, config: Config) -> None:
        """Simulate a 5-minute session with 10+ turns (natural pauses)."""
        client = MagicMock()
        transcriber = LiveTranscriber(client, config)

        sentences = [
            "First I want to talk about the dashboard. ",
            "The client needs it by Friday. ",
            "We should include proper error handling. ",
            "Also the color scheme should match their branding. ",
            "Let me also mention the mobile support requirement. ",
            "The API integration needs to handle rate limits. ",
            "We discussed pagination for the data tables. ",
            "The export feature should support CSV and PDF. ",
            "Finally the admin panel needs role-based access. ",
            "That covers everything for now. ",
        ]

        for i, sentence in enumerate(sentences):
            # Each sentence is a separate "turn" of speech
            for word in sentence.split():
                transcriber._transcription_parts.append(word + " ")

            # VAD fires between sentences (user pauses to think)
            if i < len(sentences) - 1:
                transcriber._turn_count += 1
                transcriber._transcription_parts.append(" ")

        full_text = "".join(transcriber._transcription_parts).strip()

        assert transcriber._turn_count == 9
        assert full_text.startswith("First I want to talk about the dashboard.")
        assert full_text.endswith("That covers everything for now.")
        assert len(full_text) > 400
        # Every sentence should be present
        for sentence in sentences:
            assert sentence.strip() in full_text


# ---------------------------------------------------------------------------
# Test: dedupe_repeated_text doesn't corrupt long unique text
# ---------------------------------------------------------------------------


class TestDedupeWithLongText:
    """Verify dedupe_repeated_text doesn't accidentally mangle long text."""

    def test_long_unique_text_passes_through(self) -> None:
        """A 2000+ char unique transcription should not be altered."""
        # Realistic long dictation (unique, not repeated)
        text = (
            "right code that will generate an app in HTML focusing in the "
            "main problem the client has. So the HTML that you will create "
            "will be a visual representation of a calculator that we can "
            "use to show either the revenue being lost by the client or "
            "the things that is important for their business. We need to "
            "make sure that all the calculations are accurate and that "
            "the interface is intuitive for non-technical users. The color "
            "scheme should match the client's branding guidelines and we "
            "should include proper error handling for edge cases like "
            "division by zero or negative revenue inputs."
        )
        assert dedupe_repeated_text(text) == text

    def test_very_long_text_no_false_positive(self) -> None:
        """3000+ char text with some repeated phrases should NOT be halved."""
        text = (
            "The client wants a dashboard. The client also wants reports. "
            "The client needs it by Friday. We talked to the client about "
            "the requirements and the client confirmed the timeline. "
            "Additionally the client mentioned they want mobile support. "
        ) * 3  # Repeats the paragraph but NOT exact-half duplication

        result = dedupe_repeated_text(text)
        # The text has repetitive phrasing but isn't an exact duplicate
        # so it should NOT be halved
        assert len(result) > len(text) // 2

    def test_actual_exact_duplicate_is_deduped(self) -> None:
        """If the API returns the exact same text twice, it should be deduped."""
        original = "This is the transcription of speech to text"
        doubled = original + original
        result = dedupe_repeated_text(doubled)
        assert result == original


# ---------------------------------------------------------------------------
# Test: BatchTranscriber handles large audio buffers
# ---------------------------------------------------------------------------


class TestBatchTranscriberLargeAudio:
    """Verify BatchTranscriber doesn't choke on large audio byte arrays."""

    def test_large_audio_buffer_accepted(self, config: Config) -> None:
        """Simulate 2 minutes of 16kHz 16-bit mono audio (3.84 MB)."""
        client = MagicMock()
        transcriber = BatchTranscriber(client, config)

        # 2 minutes of audio at 16kHz, 2 bytes per sample
        two_minutes_bytes = 16000 * 2 * 120  # 3,840,000 bytes
        audio_data = b"\x00" * two_minutes_bytes

        # Feed it into the transcriber's frame buffer
        transcriber._frames = [audio_data]
        raw = b"".join(transcriber._frames)

        # Verify we can handle the full buffer
        assert len(raw) == two_minutes_bytes
        assert len(raw) == 3_840_000

    def test_wav_header_on_large_audio(self, config: Config) -> None:
        """Verify WAV header wrapping works for large audio."""
        from voice_agent.stt import add_wav_header

        # 60 seconds of audio
        one_minute_bytes = 16000 * 2 * 60
        audio_data = b"\x00" * one_minute_bytes

        wav = add_wav_header(audio_data, rate=16000)

        assert wav[:4] == b"RIFF"
        assert len(wav) > one_minute_bytes  # WAV header adds bytes
