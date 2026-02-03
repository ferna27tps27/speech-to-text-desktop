import unittest
import time
import subprocess
import pyperclip
import sys
import os
import threading
from unittest.mock import MagicMock, patch

# Add parent dir to path
sys.path.append(os.getcwd())

# Import agent modules (mocking hardware dependencies if needed for unit tests, 
# but for integration we want real connections)
from gemini_agent import GeminiClient, AudioHandler, API_KEY

class TestStreamingTTS(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("\n--- Starting Streaming TTS Integration Tests ---")
        if not API_KEY or API_KEY == "your_api_key_here":
            print("WARNING: Valid GEMINI_API_KEY not found in .env.")
            raise ValueError("API Key missing")

    def test_tts_streaming_latency(self):
        """
        Tests that TTS streaming returns the first chunk quickly.
        """
        client = GeminiClient()
        text = "This is a test of the streaming audio latency. It should start playing almost immediately."
        print(f"[TEST] Requesting TTS Stream for: '{text[:20]}...'")
        
        start_time = time.time()
        
        # Call the new streaming method
        response_stream = client.text_to_speech_stream(text)
        self.assertIsNotNone(response_stream, "Stream response was None")
        
        # Iterate to get the first chunk
        first_chunk_time = None
        chunk_count = 0
        
        print("[TEST] Waiting for first audio chunk...")
        for chunk in response_stream:
            chunk_count += 1
            if chunk_count == 1:
                first_chunk_time = time.time()
                latency = first_chunk_time - start_time
                print(f"[TEST] First chunk received in {latency:.4f} seconds")
                self.assertLess(latency, 3.0, "Latency was too high (>3s)")
            
            # We just need to verify we get data, don't need to play it all
            if chunk_count >= 3:
                break
                
        self.assertGreater(chunk_count, 0, "No audio chunks received")

if __name__ == '__main__':
    unittest.main()
