"""macOS-specific platform integrations.

Handles clipboard operations, system sounds, AppleScript-based paste,
and frontmost app detection. All macOS dependencies are isolated here.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Literal

import pyperclip
from pynput import keyboard

from .config import Config

logger = logging.getLogger(__name__)

# macOS system sound paths
_SYSTEM_SOUNDS = {
    "start": "/System/Library/Sounds/Tink.aiff",
    "stop": "/System/Library/Sounds/Pop.aiff",
    "error": "/System/Library/Sounds/Basso.aiff",
}


class MacOSBridge:
    """macOS platform integration using AppleScript and system utilities."""

    def __init__(self, config: Config) -> None:
        self._config = config

    def play_feedback(
        self, event: Literal["start", "stop", "error"]
    ) -> None:
        """Play a system sound for user feedback."""
        try:
            path = _SYSTEM_SOUNDS.get(event)
            if path and os.path.exists(path):
                subprocess.run(["afplay", path], check=False)
        except Exception:
            pass

    def get_frontmost_app(self) -> str | None:
        """Return the name of the currently focused application."""
        try:
            applescript = (
                'tell application "System Events" to get name of '
                'first application process whose frontmost is true'
            )
            result = subprocess.run(
                ["osascript", "-e", applescript],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                return None
            return result.stdout.strip()
        except Exception:
            return None

    def paste_text(self, text: str) -> None:
        """Copy text to clipboard and paste it into the frontmost application."""
        try:
            pyperclip.copy(text + " ")
            time.sleep(0.1)  # Ensure clipboard is ready

            front_app = self.get_frontmost_app()
            if front_app:
                logger.info(f"Frontmost app: {front_app}")

            self._paste_to_frontmost_app()

            if self._config.terminal_paste_debug:
                self._paste_to_terminal(text)

        except Exception as e:
            logger.error(f"Paste failed: {e}")

    def copy_selected_text(self) -> str:
        """Simulate Cmd+C to copy selected text and return it.

        Polls the clipboard for up to 2 seconds waiting for the copy
        to be processed by the frontmost application.
        """
        old_clipboard = pyperclip.paste()
        pyperclip.copy("")

        # Hardware-level Cmd+C via pynput (bypasses permission issues)
        controller = keyboard.Controller()
        with controller.pressed(keyboard.Key.cmd):
            controller.press("c")
            controller.release("c")

        time.sleep(0.15)  # Give the app time to process

        text = ""
        start = time.monotonic()
        poll_count = 0
        while time.monotonic() - start < 2.0:
            text = pyperclip.paste()
            poll_count += 1
            if text and text.strip() and text != old_clipboard:
                logger.info(
                    f"Clipboard updated after {poll_count} polls "
                    f"({int((time.monotonic() - start) * 1000)}ms)"
                )
                break
            time.sleep(0.05)

        if not text or not text.strip():
            logger.warning(
                f"No text selected or clipboard empty after {poll_count} polls. "
                f"Last clipboard content: '{text[:100] if text else 'EMPTY'}'"
            )
            return ""

        return text.strip()

    # -- internal ------------------------------------------------------------

    @staticmethod
    def _paste_to_frontmost_app() -> None:
        try:
            applescript = """
            tell application "System Events"
                keystroke "v" using command down
            end tell
            """
            subprocess.run(["osascript", "-e", applescript], check=False)
        except Exception:
            pass

    @staticmethod
    def _paste_to_terminal(text: str) -> None:
        try:
            applescript = """
            on run argv
                set theText to item 1 of argv
                tell application "Terminal"
                    do script "printf '%s\\n' " & quoted form of theText in front window
                end tell
            end run
            """
            subprocess.run(
                ["osascript", "-e", applescript, text], check=False
            )
        except Exception:
            pass
