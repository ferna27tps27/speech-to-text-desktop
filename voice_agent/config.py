"""Configuration management for the voice agent.

Loads settings from environment variables with sensible defaults.
All configuration is immutable after creation via a frozen dataclass.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Available TTS voices (30 options):
# Bright: Zephyr, Autonoe          Upbeat: Puck, Laomedeia
# Informative: Charon, Rasalgethi  Firm: Kore, Orus, Alnilam
# Excitable: Fenrir                Youthful: Leda
# Breezy: Aoede                    Easy-going: Callirrhoe, Umbriel
# Breathy: Enceladus               Clear: Iapetus, Erinome
# Smooth: Algieba, Despina         Gravelly: Algenib
# Soft: Achernar                   Even: Schedar
# Mature: Gacrux                   Forward: Pulcherrima
# Friendly: Achird                 Casual: Zubenelgenubi
# Gentle: Vindemiatrix             Lively: Sadachbia
# Knowledgeable: Sadaltager        Warm: Sulafat


@dataclass(frozen=True)
class Config:
    """Immutable application configuration loaded from environment variables."""

    # API Keys
    gemini_api_key: str = ""
    elevenlabs_api_key: str = ""

    # Models
    tts_model: str = "gemini-2.5-flash-preview-tts"
    stt_model: str = "gemini-2.5-flash"
    live_stt_model: str = "gemini-2.5-flash-native-audio-preview-12-2025"

    # Feature flags
    use_live_api: bool = True
    use_elevenlabs_tts: bool = False
    warmup_tts: bool = False
    terminal_paste_debug: bool = True

    # Audio
    sample_rate_in: int = 16000
    sample_rate_out: int = 24000
    chunk_size: int = 1024

    # VAD
    live_silence_duration_ms: int = 800

    # Voice / Style
    voice_name: str = "Kore"
    tts_style: str = "naturally and clearly"
    elevenlabs_model: str = "eleven_flash_v2_5"
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"

    # TTS chunking
    tts_chunk_chars: int = 2000

    @classmethod
    def from_env(cls, env_path: str | None = None) -> Config:
        """Load configuration from .env file and environment variables.

        Args:
            env_path: Optional path to .env file. If None, searches default locations.

        Returns:
            Validated Config instance.

        Raises:
            ValueError: If required configuration is invalid.
        """
        load_dotenv(env_path)

        def _bool(key: str, default: str = "false") -> bool:
            return os.getenv(key, default).lower() in ("1", "true", "yes")

        def _int(key: str, default: str) -> int:
            raw = os.getenv(key, default)
            try:
                return int(raw)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid integer for {key}={raw!r}, using default {default}"
                )
                return int(default)

        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        if not gemini_api_key:
            logger.warning(
                "GEMINI_API_KEY not found in .env. Please ensure it is set."
            )

        return cls(
            gemini_api_key=gemini_api_key,
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            tts_model=os.getenv("TTS_MODEL", "") or "gemini-2.5-flash-preview-tts",
            stt_model=os.getenv("STT_MODEL", "") or "gemini-2.5-flash",
            live_stt_model=os.getenv(
                "LIVE_STT_MODEL",
                "gemini-2.5-flash-native-audio-preview-12-2025",
            ),
            use_live_api=_bool("USE_LIVE_API", "true"),
            use_elevenlabs_tts=_bool("USE_ELEVENLABS_TTS", "false"),
            warmup_tts=_bool("WARMUP_TTS", "false"),
            terminal_paste_debug=_bool("TERMINAL_PASTE_DEBUG", "true"),
            sample_rate_in=_int("SAMPLE_RATE_IN", "16000"),
            sample_rate_out=_int("SAMPLE_RATE_OUT", "24000"),
            chunk_size=_int("CHUNK_SIZE", "1024"),
            live_silence_duration_ms=_int("LIVE_SILENCE_DURATION_MS", "800"),
            voice_name=os.getenv("VOICE_NAME", "Kore"),
            tts_style=os.getenv("TTS_STYLE", "naturally and clearly"),
            elevenlabs_model=os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
            elevenlabs_voice_id=os.getenv(
                "ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"
            ),
            tts_chunk_chars=_int("TTS_CHUNK_CHARS", "2000"),
        )
