import unittest
import time
import subprocess
import pyperclip
import sys
import os

# Add parent dir to path
sys.path.append(os.getcwd())

class TestPasteFunctionality(unittest.TestCase):
    def test_clipboard_paste_simulation(self):
        """
        Simulates the exact paste mechanism used in the agent.
        User should focus a text field (like Notes) before running this test 
        to visually verify it works.
        """
        test_text = "Hello! This is a test from the automated test suite."
        print(f"\n[TEST] Attempting to paste: '{test_text}'")
        print("[TEST] You have 3 seconds to focus a text field (e.g. Notes app)...")
        time.sleep(3)
        
        # 1. Copy to clipboard
        pyperclip.copy(test_text)
        
        # 2. Paste using AppleScript (Same logic as agent)
        applescript = """
        tell application "System Events"
            keystroke "v" using command down
        end tell
        """
        try:
            subprocess.run(["osascript", "-e", applescript], check=True)
            print("[TEST] Paste command sent via AppleScript.")
        except subprocess.CalledProcessError as e:
            self.fail(f"AppleScript execution failed: {e}")

if __name__ == '__main__':
    unittest.main()
