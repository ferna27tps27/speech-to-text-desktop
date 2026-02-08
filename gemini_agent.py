"""Backward-compatible entry point.

This module re-exports key symbols from the refactored voice_agent package
so that existing scripts and tests that ``import gemini_agent`` continue
to work.

For new code, prefer importing from ``voice_agent`` directly::

    from voice_agent.config import Config
    from voice_agent.app import App
"""

from __future__ import annotations

import logging
import sys

from voice_agent.config import Config
from voice_agent.audio import AudioHandler
from voice_agent.app import App
from voice_agent.stt import add_wav_header

# Setup logging (preserves original top-level behavior)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Load config from env (provides backward-compatible globals)
_config = Config.from_env()

# Backward-compatible globals that old tests rely on
API_KEY = _config.gemini_api_key


class GeminiClient:
    """Backward-compatible wrapper around the refactored modules.

    Existing tests import and use this class directly. It delegates
    to the new modular implementations under the hood.
    """

    def __init__(self) -> None:
        from google import genai
        from google.genai import types
        from voice_agent.tts import GeminiTTSProvider

        if not _config.gemini_api_key:
            raise ValueError("API Key is missing")

        self.client = genai.Client(
            api_key=_config.gemini_api_key,
            http_options=types.HttpOptions(timeout=60000),
        )

        self._audio = AudioHandler(_config)
        self._tts = GeminiTTSProvider(self.client, self._audio, _config)
        self._retry_config = types.HttpRetryOptions(
            initial_delay=1.0,
            max_delay=10.0,
            attempts=3,
        )

    def transcribe(self, audio_bytes: bytes) -> str | None:
        """Transcribe raw PCM audio bytes to text."""
        from google.genai import types

        logging.info("Sending audio to Gemini...")
        try:
            wav_data = add_wav_header(audio_bytes, rate=_config.sample_rate_in)

            response = self.client.models.generate_content(
                model=_config.stt_model,
                contents=[
                    types.Part.from_bytes(data=wav_data, mime_type="audio/wav"),
                    (
                        "Transcribe the audio exactly as spoken, but format "
                        "punctuation commands into their symbols.\n\n"
                        "Rules:\n"
                        '- Replace "comma" with ","\n'
                        '- Replace "dot" or "period" with "."\n'
                        '- Replace "semicolon" with ";"\n'
                        '- Replace "colon" with ":"\n'
                        '- Replace "question mark" with "?"\n'
                        '- Replace "exclamation mark" with "!"\n'
                        '- Replace "dash" or "hyphen" with "-"\n'
                        '- Replace "new line" or "next line" with a line break\n'
                        '- Replace "open parenthesis" with "(" and '
                        '"close parenthesis" with ")"\n'
                        '- Replace "open quote" with \'"\' and '
                        '"close quote" with \'"\'\n\n'
                        "Do not add any other markdown or speaker labels. "
                        "Just the formatted text."
                    ),
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT"],
                    temperature=0.0,
                    http_options=types.HttpOptions(
                        retry_options=self._retry_config,
                    ),
                ),
            )
            return response.text
        except Exception as e:
            logging.error(f"Transcription failed: {e}")
            return None

    def text_to_speech(self, text: str) -> object | None:
        """Non-streaming TTS for backward compatibility."""
        return self._tts.text_to_speech(text)

    def text_to_speech_stream(self, text: str) -> object | None:
        """Streaming TTS for backward compatibility."""
        return self._tts.text_to_speech_stream(text)


def main() -> None:
    """Entry point for backward compatibility (``python gemini_agent.py``)."""
    app = App(_config)
    app.run()


if __name__ == "__main__":
    main()
