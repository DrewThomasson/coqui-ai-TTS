#!/usr/bin/env python3
"""
Simple StyleTTS2 transcription test.
"""

import os
import sys
import whisper
from pathlib import Path

# Add TTS to path
sys.path.insert(0, str(Path(__file__).parent))

def test_transcription():
    """Test current StyleTTS2 output with transcription."""
    
    print("="*60)
    print("STYLETTS2 TRANSCRIPTION TEST")
    print("="*60)
    
    test_text = "Hello world this is a test file!"
    print(f"Input text: '{test_text}'")
    
    # Generate audio
    print("\n1. Generating audio with StyleTTS2...")
    try:
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/styletts2_ljspeech")
        
        # Generate with reference voice
        output_path = "output.wav"
        if os.path.exists("reference.wav"):
            print("Using reference.wav for voice cloning")
            tts.tts_to_file(test_text, speaker_wav="reference.wav", file_path=output_path)
        else:
            print("Using default voice")
            tts.tts_to_file(test_text, file_path=output_path)
            
        print(f"✓ Audio saved to: {output_path}")
        
    except Exception as e:
        print(f"✗ Audio generation failed: {e}")
        return False
    
    # Test transcription
    print("\n2. Testing transcription...")
    try:
        model = whisper.load_model("base")
        
        # Try multiple approaches
        print("\nTranscribing with different methods...")
        
        # Method 1: Direct transcription
        result = model.transcribe(output_path)
        transcription = result["text"].strip()
        print(f"Direct transcription: '{transcription}'")
        
        # Method 2: With language hint
        result_en = model.transcribe(output_path, language="en")
        transcription_en = result_en["text"].strip()
        print(f"English transcription: '{transcription_en}'")
        
        # Compare results
        if transcription or transcription_en:
            print(f"\n✓ Got transcription results!")
            
            # Check for word matches
            input_words = set(test_text.lower().split())
            trans_words = set((transcription + " " + transcription_en).lower().split())
            
            overlap = input_words.intersection(trans_words)
            overlap_ratio = len(overlap) / len(input_words) if input_words else 0
            
            print(f"Word overlap: {len(overlap)}/{len(input_words)} ({overlap_ratio*100:.1f}%)")
            if overlap:
                print(f"Matching words: {overlap}")
            
            if overlap_ratio > 0.3:
                print("🎉 SUCCESS: StyleTTS2 is producing recognizable speech!")
                return True
            elif transcription.strip() or transcription_en.strip():
                print("⚠️  PARTIAL: StyleTTS2 produces audio but not matching text")
                return "partial"
            else:
                print("❌ FAILED: No recognizable speech")
                return False
        else:
            print("❌ FAILED: Empty transcription")
            return False
            
    except Exception as e:
        print(f"✗ Transcription failed: {e}")
        return False

if __name__ == "__main__":
    result = test_transcription()
    print("\n" + "="*60)
    print("FINAL RESULT")
    print("="*60)
    
    if result is True:
        print("🎉 StyleTTS2 IS WORKING CORRECTLY!")
    elif result == "partial":
        print("⚠️  StyleTTS2 has partial functionality")
    else:
        print("❌ StyleTTS2 NEEDS TO BE FIXED")