import unittest
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch
import numpy as np

# Add the current directory to path so we can import the agent
sys.path.append(os.getcwd())

# Import the modules to test
# We need to mock some of the imports in gemini_agent if we don't want to rely on real hardware/libs for unit tests
# However, the user asked for "integration tests" to make sure "everything is running great", 
# implying they want to test the actual API connection and logic.

from gemini_agent import GeminiClient, AudioHandler, API_KEY

class TestGeminiIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n--- Starting Integration Tests ---")
        if not API_KEY or API_KEY == "your_api_key_here":
            print("WARNING: Valid GEMINI_API_KEY not found in .env.")
            print("Tests will likely fail if they hit the real API.")
        else:
            print("API Key found. Proceeding with real API tests.")

    def test_01_api_authentication(self):
        """Test that we can initialize the Gemini client with the key."""
        try:
            client = GeminiClient()
            self.assertIsNotNone(client.client)
            print("✓ Gemini Client initialization successful")
        except Exception as e:
            self.fail(f"Gemini Client initialization failed: {e}")

    def test_02_tts_generation(self):
        """Test sending text to Gemini TTS and receiving audio bytes."""
        client = GeminiClient()
        text = "This is a test of the text to speech system."
        print(f"Testing TTS with text: '{text}'")
        
        audio_data = client.text_to_speech(text)
        
        self.assertIsNotNone(audio_data, "TTS returned None")
        self.assertIsInstance(audio_data, bytes, "TTS should return bytes")
        self.assertGreater(len(audio_data), 100, "TTS audio data seems too small")
        print(f"✓ TTS success. Received {len(audio_data)} bytes of audio.")

    def test_03_stt_transcription(self):
        """Test sending a synthetic audio chunk to Gemini STT."""
        client = GeminiClient()
        
        # Create a dummy silent WAV file in memory to test the connection/pipeline
        # We don't expect accurate transcription of silence, but we expect a valid response (not a 401/500 error)
        # Note: Gemini might return empty string for silence, or hallucinate. 
        # We mainly check that the request completes successfully.
        
        # Generating 1 second of silence at 16kHz
        duration = 1
        rate = 16000
        # Create a simple sine wave (beep) so it's not pure silence (which might be filtered)
        t = np.linspace(0, duration, int(rate * duration), False)
        # 440Hz sine wave
        audio_data = (np.sin(440 * 2 * np.pi * t) * 32767).astype(np.int16).tobytes()
        
        print("Testing STT with synthetic sine wave audio...")
        try:
            text = client.transcribe(audio_data)
            print(f"✓ STT Request completed. Result: '{text}'")
            # We don't assert the content of text strictly because models vary on sine waves,
            # but we assert the function didn't raise an exception.
        except Exception as e:
            self.fail(f"STT Transcription failed: {e}")

    @patch('pyaudio.PyAudio')
    def test_04_audio_handler_initialization(self, mock_pyaudio):
        """Test AudioHandler initialization (Mocked Hardware)."""
        try:
            handler = AudioHandler()
            self.assertIsNotNone(handler.p)
            print("✓ AudioHandler initialized successfully")
        except Exception as e:
            self.fail(f"AudioHandler initialization failed: {e}")

if __name__ == '__main__':
    unittest.main()
