"""Tests for voice_agent.hotkeys."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from voice_agent.hotkeys import HotkeyManager, VK_D, VK_R, _DEBOUNCE_SECONDS


class TestHotkeyManager:
    def test_register_adds_bindings(self) -> None:
        mgr = HotkeyManager()
        callback = MagicMock()
        mgr.register({"ctrl", "alt"}, VK_D, callback)
        assert len(mgr._bindings) == 1

    def test_register_with_char_fallback(self) -> None:
        mgr = HotkeyManager()
        callback = MagicMock()
        mgr.register({"ctrl", "alt"}, VK_D, callback, char_fallback="d")
        # Should have two bindings: one for vk code, one for char
        assert len(mgr._bindings) == 2

    def test_stop_without_start_is_safe(self) -> None:
        mgr = HotkeyManager()
        mgr.stop()  # Should not raise

    def test_vk_constants(self) -> None:
        assert VK_D == 2
        assert VK_R == 15

    def test_debounce_prevents_double_fire(self) -> None:
        """Simulate the exact bug: vk match and char fallback both match
        across two rapid on_press calls. Only one callback should fire."""
        mgr = HotkeyManager()
        callback = MagicMock()
        mgr.register({"ctrl", "alt"}, VK_R, callback, char_fallback="r")

        from pynput import keyboard

        # Simulate holding Ctrl+Alt
        mgr._current_keys = {keyboard.Key.ctrl, keyboard.Key.alt}

        # Create a KeyCode like pynput would for 'r' with vk=15
        fake_key = keyboard.KeyCode(vk=VK_R, char="r")

        # First press -- should fire
        mgr._on_press(fake_key)
        assert callback.call_count == 1

        # Second press immediately after (same keypress, duplicate event) -- debounced
        mgr._on_press(fake_key)
        assert callback.call_count == 1  # Still 1, not 2

    def test_debounce_constant_is_reasonable(self) -> None:
        assert 0.3 <= _DEBOUNCE_SECONDS <= 1.0
