#!/usr/bin/env python3
"""
Test script to diagnose current StyleTTS2 implementation issues.
"""
import os
import sys
import torch
import torchaudio
import numpy as np
import logging

# Add TTS to path
sys.path.append('/home/runner/work/coqui-ai-TTS/coqui-ai-TTS')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_reference_audio():
    """Test the reference audio file."""
    try:
        wav, sr = torchaudio.load('/home/runner/work/coqui-ai-TTS/coqui-ai-TTS/reference.wav')
        print(f"Reference audio loaded: shape={wav.shape}, sr={sr}")
        print(f"Duration: {wav.shape[1] / sr:.2f} seconds")
        return True
    except Exception as e:
        print(f"Error loading reference audio: {e}")
        return False

def test_styletts2_api():
    """Test StyleTTS2 through the TTS API."""
    try:
        from TTS.api import TTS
        
        print("Initializing StyleTTS2...")
        tts = TTS("tts_models/multilingual/multi-dataset/styletts2_ljspeech")
        
        print("Running TTS synthesis...")
        output_path = "/home/runner/work/coqui-ai-TTS/coqui-ai-TTS/test_output.wav"
        tts.tts_to_file(
            "Hello world this is a test file!",
            speaker_wav="/home/runner/work/coqui-ai-TTS/coqui-ai-TTS/reference.wav",
            file_path=output_path
        )
        
        # Check if output file was created
        if os.path.exists(output_path):
            # Load the output and analyze it
            wav, sr = torchaudio.load(output_path)
            print(f"Output audio created: shape={wav.shape}, sr={sr}")
            print(f"Duration: {wav.shape[1] / sr:.2f} seconds")
            print(f"Audio range: [{wav.min():.4f}, {wav.max():.4f}]")
            print(f"RMS level: {torch.sqrt(torch.mean(wav**2)):.6f}")
            
            # Analyze audio characteristics
            if torch.all(wav == 0):
                print("⚠️ OUTPUT IS SILENT (all zeros)")
                return False
            elif torch.sqrt(torch.mean(wav**2)) < 1e-6:
                print("⚠️ OUTPUT IS EXTREMELY QUIET (likely silence)")
                return False
            elif torch.max(torch.abs(wav)) > 0.95:
                print("⚠️ OUTPUT MAY BE CLIPPED")
                return False
            else:
                print("✅ Output has reasonable audio levels")
                return True
        else:
            print("❌ No output file created")
            return False
            
    except Exception as e:
        print(f"❌ Error testing StyleTTS2 API: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_with_whisper(audio_path):
    """Test audio output using Whisper speech recognition."""
    try:
        import whisper
        
        print(f"Loading Whisper model...")
        model = whisper.load_model("base")
        
        print(f"Transcribing audio: {audio_path}")
        result = model.transcribe(audio_path)
        
        print(f"Whisper transcription: '{result['text']}'")
        
        # Check if transcription makes sense
        expected_words = ["hello", "world", "test", "file"]
        transcribed_text = result['text'].lower()
        
        matches = sum(1 for word in expected_words if word in transcribed_text)
        print(f"Expected words found: {matches}/{len(expected_words)}")
        
        if matches >= 2:
            print("✅ Transcription indicates real speech")
            return True
        else:
            print("⚠️ Transcription doesn't match expected text (likely noise/distorted)")
            return False
            
    except Exception as e:
        print(f"❌ Error with Whisper transcription: {e}")
        return False

if __name__ == "__main__":
    print("=== StyleTTS2 Implementation Test ===")
    
    # Test 1: Reference audio
    print("\n1. Testing reference audio...")
    ref_ok = test_reference_audio()
    
    if not ref_ok:
        print("❌ Reference audio test failed - stopping")
        sys.exit(1)
    
    # Test 2: StyleTTS2 API
    print("\n2. Testing StyleTTS2 API...")
    api_ok = test_styletts2_api()
    
    # Test 3: Whisper transcription of output
    if api_ok:
        print("\n3. Testing output with Whisper...")
        whisper_ok = test_audio_with_whisper("/home/runner/work/coqui-ai-TTS/coqui-ai-TTS/test_output.wav")
        
        if whisper_ok:
            print("\n✅ OVERALL: StyleTTS2 is working correctly!")
        else:
            print("\n⚠️ OVERALL: StyleTTS2 produces audio but it's not intelligible speech")
    else:
        print("\n❌ OVERALL: StyleTTS2 API failed - cannot test speech quality")
    
    print("\n=== Test Complete ===")