#!/usr/bin/env python3
"""
Test the debug audio files with whisper
"""

import whisper

def test_debug_audio():
    """Test the debug audio files."""
    
    print("Testing debug audio files with Whisper...")
    
    model = whisper.load_model("base")
    
    # Test files
    files = ["debug_audio.wav", "simple_audio.wav", "output.wav"]
    
    for filename in files:
        try:
            print(f"\n--- Testing {filename} ---")
            result = model.transcribe(filename)
            transcription = result["text"].strip()
            print(f"Transcription: '{transcription}'")
            
            if transcription:
                print(f"✓ Got transcription: {len(transcription)} characters")
            else:
                print("❌ Empty transcription")
                
        except Exception as e:
            print(f"✗ Failed to transcribe {filename}: {e}")

if __name__ == "__main__":
    test_debug_audio()