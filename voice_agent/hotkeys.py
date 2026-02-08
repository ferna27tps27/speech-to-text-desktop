"""Global hotkey management using pynput.

Isolates keyboard listener setup and key tracking into a reusable manager.
Uses raw virtual key codes for macOS compatibility with modifier keys.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from pynput import keyboard

logger = logging.getLogger(__name__)

# Minimum interval between repeated firings of the same hotkey (seconds).
# Prevents double-fires from pynput sending both vk and char matches,
# or from OS key-repeat kicking in while modifiers are held.
_DEBOUNCE_SECONDS = 0.5

# macOS virtual key codes (consistent regardless of modifiers)
VK_D = 2
VK_R = 15


class HotkeyManager:
    """Manages global hotkey bindings using pynput's keyboard Listener.

    Uses a standard Listener instead of GlobalHotKeys to avoid the
    'injected' argument bug on macOS. Tracks currently held keys and
    fires callbacks when modifier+key combinations are detected.
    """

    def __init__(self) -> None:
        self._bindings: list[
            tuple[frozenset[str], int | str, Callable[[], None]]
        ] = []
        self._current_keys: set[keyboard.Key | keyboard.KeyCode] = set()
        self._listener: keyboard.Listener | None = None
        # Track last fire time per callback to debounce duplicate events
        self._last_fire: dict[Callable[[], None], float] = {}

    def register(
        self,
        modifiers: set[str],
        key_code: int,
        callback: Callable[[], None],
        char_fallback: str | None = None,
    ) -> None:
        """Register a hotkey binding.

        Args:
            modifiers: Set of modifier names: "ctrl", "alt".
            key_code: macOS virtual key code for the target key.
            callback: Function to call when the hotkey is pressed.
            char_fallback: Optional character to also match (e.g. 'd').
        """
        self._bindings.append(
            (frozenset(modifiers), key_code, callback)
        )
        if char_fallback:
            self._bindings.append(
                (frozenset(modifiers), char_fallback, callback)
            )

    def start(self) -> None:
        """Start listening for hotkeys (non-blocking - listener runs in a thread)."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        logger.info("Hotkey listener started.")

    def join(self) -> None:
        """Block the calling thread until the listener stops."""
        if self._listener:
            self._listener.join()

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if self._listener:
            self._listener.stop()
            self._listener = None

    # -- internal ------------------------------------------------------------

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        try:
            self._current_keys.add(key)

            vk = getattr(key, "vk", None)
            char = None
            if hasattr(key, "char") and key.char:
                char = key.char.lower()

            ctrl_pressed = any(
                k in self._current_keys
                for k in [
                    keyboard.Key.ctrl,
                    keyboard.Key.ctrl_l,
                    keyboard.Key.ctrl_r,
                ]
            )
            alt_pressed = any(
                k in self._current_keys
                for k in [
                    keyboard.Key.alt,
                    keyboard.Key.alt_l,
                    keyboard.Key.alt_r,
                ]
            )

            held_modifiers: set[str] = set()
            if ctrl_pressed:
                held_modifiers.add("ctrl")
            if alt_pressed:
                held_modifiers.add("alt")

            for required_mods, target_key, callback in self._bindings:
                if required_mods != frozenset(held_modifiers):
                    continue

                matched = (
                    (isinstance(target_key, int) and vk == target_key)
                    or (isinstance(target_key, str) and char == target_key)
                )
                if not matched:
                    continue

                # Debounce: skip if this callback fired too recently
                now = time.monotonic()
                last = self._last_fire.get(callback, float("-inf"))
                if now - last < _DEBOUNCE_SECONDS:
                    break
                self._last_fire[callback] = now
                callback()
                break  # Only fire once per keypress event

        except Exception as e:
            logger.error(f"Key handler error: {e}")

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        try:
            self._current_keys.discard(key)
        except Exception:
            pass
