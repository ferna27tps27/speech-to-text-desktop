import unittest
import time
import subprocess
import pyperclip
import sys
import os

# Add parent dir to path
sys.path.append(os.getcwd())

from pynput import keyboard

# ... (existing code)

class TestCopyFunctionality(unittest.TestCase):
    def test_clipboard_copy_simulation(self):
        """
        Simulates the exact copy mechanism used in the agent.
        User should highlight text in an editor before running this test.
        """
        print("\n[TEST] You have 5 seconds to highlight some text in any app...")
        for i in range(5, 0, -1):
            print(f"[TEST] {i}...")
            time.sleep(1)
        
        # Clear clipboard first
        pyperclip.copy("")
        
        # Execute pynput Copy
        print("[TEST] Sending Cmd+C via pynput...")
        controller = keyboard.Controller()
        with controller.pressed(keyboard.Key.cmd):
            controller.press('c')
            controller.release('c')
            
        time.sleep(0.8)
        
        # Check result
        copied_text = pyperclip.paste()
        print(f"[TEST] Clipboard content: '{copied_text}'")
        
        if not copied_text:
            print("[TEST] FAIL: Clipboard is empty.")
            self.fail("Clipboard was empty after copy simulation.")
        else:
            print("[TEST] SUCCESS: Text copied successfully.")

if __name__ == '__main__':
    unittest.main()
