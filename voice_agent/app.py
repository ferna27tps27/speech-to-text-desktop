"""Application orchestrator.

Wires all components together and provides the high-level dictation
and read-aloud workflows. No global state -- everything is owned by
the App instance.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from google import genai
from google.genai import types

from .audio import AudioHandler
from .config import Config
from .hotkeys import HotkeyManager, VK_D, VK_R
from .platform_macos import MacOSBridge
from .stt import BatchTranscriber, LiveTranscriber, Transcriber, dedupe_repeated_text
from .tts import ElevenLabsTTSProvider, GeminiTTSProvider, TTSProvider

logger = logging.getLogger(__name__)


class App:
    """Main application that owns all components via dependency injection."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.platform = MacOSBridge(config)
        self.audio = AudioHandler(config)

        # Initialize Gemini client
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is missing")

        self._gemini_client = genai.Client(
            api_key=config.gemini_api_key,
            http_options=types.HttpOptions(timeout=60000),
        )
        logger.info("Gemini Client initialized.")

        # Build STT transcriber
        self.transcriber: Transcriber = self._build_transcriber()

        # Build TTS provider
        self.tts: TTSProvider = self._build_tts_provider()

        # Hotkey manager
        self._hotkeys = HotkeyManager()

    # -- builders -----------------------------------------------------------

    def _build_transcriber(self) -> Transcriber:
        if self.config.use_live_api:
            try:
                transcriber = LiveTranscriber(self._gemini_client, self.config)
                logger.info(
                    f"Live API dictation enabled "
                    f"(model: {self.config.live_stt_model})"
                )
                logger.info(
                    f"  - Silence detection: "
                    f"{self.config.live_silence_duration_ms}ms"
                )
                return transcriber
            except Exception as e:
                logger.warning(f"Live API init failed, using batch mode: {e}")

        logger.info("Using batch transcription mode")
        return BatchTranscriber(self._gemini_client, self.config)

    def _build_tts_provider(self) -> TTSProvider:
        # Try ElevenLabs first (much faster)
        if self.config.use_elevenlabs_tts:
            try:
                provider = ElevenLabsTTSProvider(self.config)
                logger.info(
                    f"ElevenLabs TTS enabled "
                    f"(model: {self.config.elevenlabs_model})"
                )
                logger.info("  - Expected latency: ~240ms (vs Gemini's 20-60s)")
                return provider
            except (ImportError, ValueError) as e:
                logger.warning(f"ElevenLabs TTS unavailable: {e}")

        # Fallback to Gemini TTS
        logger.info("Using Gemini TTS (slower, but no additional API key needed)")
        gemini_tts = GeminiTTSProvider(
            self._gemini_client, self.audio, self.config
        )
        if self.config.warmup_tts:
            threading.Thread(target=gemini_tts.warmup, daemon=True).start()
        return gemini_tts

    # -- workflows -----------------------------------------------------------

    def handle_dictation(self) -> None:
        """Toggle dictation: start recording or stop and paste transcription."""
        if self.transcriber.is_active:
            self.platform.play_feedback("stop")
            logger.info("Stopping dictation...")

            def process() -> None:
                start_time = time.monotonic()
                self.transcriber.stop()
                text = self.transcriber.get_result(timeout=5.0)
                elapsed_ms = int((time.monotonic() - start_time) * 1000)

                if text:
                    logger.info(f"Transcription completed in {elapsed_ms}ms")
                    text = dedupe_repeated_text(text)
                    logger.info(f"Typing: {text}")
                    self.platform.paste_text(text)
                else:
                    logger.warning("No transcription returned")
                    self.platform.play_feedback("error")

            threading.Thread(target=process, daemon=True).start()
        else:
            self.platform.play_feedback("start")
            self.transcriber.start(self.audio.pyaudio_instance)
            logger.info("Dictation started - speak now...")

    def handle_read_aloud(self) -> None:
        """Copy selected text and read it aloud via TTS."""
        logger.info("Simulating copy...")
        text = self.platform.copy_selected_text()
        if not text:
            return

        logger.info(f"Reading: {text[:50]}...")

        def process() -> None:
            try:
                self.tts.speak(text)
            except Exception as e:
                logger.warning(f"Primary TTS failed: {e}")
                # If ElevenLabs failed, try Gemini as fallback
                if not isinstance(self.tts, GeminiTTSProvider):
                    logger.info("Falling back to Gemini TTS...")
                    fallback = GeminiTTSProvider(
                        self._gemini_client, self.audio, self.config
                    )
                    fallback.speak(text)
                else:
                    self.platform.play_feedback("error")

        threading.Thread(target=process, daemon=True).start()

    # -- lifecycle -----------------------------------------------------------

    def run(self) -> None:
        """Register hotkeys and run the main event loop."""
        self._hotkeys.register(
            modifiers={"ctrl", "alt"},
            key_code=VK_D,
            callback=lambda: self._on_hotkey(
                "Ctrl+Option+D (Dictation)", self.handle_dictation
            ),
            char_fallback="d",
        )
        self._hotkeys.register(
            modifiers={"ctrl", "alt"},
            key_code=VK_R,
            callback=lambda: self._on_hotkey(
                "Ctrl+Option+R (Read Aloud)", self.handle_read_aloud
            ),
            char_fallback="r",
        )

        logger.info("Hotkeys registered:")
        logger.info("  - Dictation: Ctrl + Alt + D")
        logger.info("  - Read Aloud: Ctrl + Alt + R")

        self._hotkeys.start()
        logger.info("Agent is running. Press Ctrl+C to exit.")

        try:
            self._hotkeys.join()
        except KeyboardInterrupt:
            logger.info("Stopping agent...")
        finally:
            self._hotkeys.stop()
            self.audio.cleanup()

    @staticmethod
    def _on_hotkey(name: str, handler: Callable[[], None]) -> None:
        logger.info(f"Hotkey detected: {name}")
        handler()
