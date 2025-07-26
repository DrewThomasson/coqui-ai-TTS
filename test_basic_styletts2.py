#!/usr/bin/env python3
"""
Quick test to validate the fixed StyleTTS2 implementation
"""
import os
import sys
import torch
import torchaudio
import numpy as np
import logging

# Add TTS to path
sys.path.append('/home/runner/work/coqui-ai-TTS/coqui-ai-TTS')

def test_simple_tts():
    """Test basic TTS without voice cloning first."""
    try:
        from TTS.api import TTS
        
        print("Testing basic StyleTTS2 synthesis (no voice cloning)...")
        tts = TTS("tts_models/multilingual/multi-dataset/styletts2_ljspeech")
        
        # Try basic synthesis without speaker_wav
        output_path = "/home/runner/work/coqui-ai-TTS/coqui-ai-TTS/test_basic.wav"
        tts.tts_to_file(
            "Hello world this is a basic test!",
            file_path=output_path
        )
        
        if os.path.exists(output_path):
            # Load and analyze output
            wav, sr = torchaudio.load(output_path)
            print(f"✅ Basic synthesis successful: shape={wav.shape}, sr={sr}")
            print(f"Audio range: [{wav.min():.4f}, {wav.max():.4f}]")
            print(f"RMS level: {torch.sqrt(torch.mean(wav**2)):.6f}")
            return True
        else:
            print("❌ No output file created")
            return False
            
    except Exception as e:
        print(f"❌ Basic synthesis failed: {e}")
        return False

if __name__ == "__main__":
    print("=== StyleTTS2 Basic Test ===")
    success = test_simple_tts()
    if success:
        print("✅ Basic synthesis works - now testing voice cloning...")
        # If basic works, the issue is only with voice cloning
    else:
        print("❌ Basic synthesis failed - architecture issue")