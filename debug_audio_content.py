#!/usr/bin/env python3
"""
Debug audio content to see what's being generated
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt

# Load the generated audio file
audio_file = "/home/runner/work/coqui-ai-TTS/coqui-ai-TTS/test_output_pretrained.wav"

try:
    audio, sr = librosa.load(audio_file, sr=22050)
    print(f"Audio loaded: {len(audio)} samples at {sr} Hz")
    print(f"Duration: {len(audio) / sr:.2f} seconds")
    print(f"Audio range: [{audio.min():.6f}, {audio.max():.6f}]")
    print(f"RMS level: {np.sqrt(np.mean(audio**2)):.6f}")
    print(f"Max absolute: {np.abs(audio).max():.6f}")
    
    # Check if audio is silent
    if np.abs(audio).max() < 1e-6:
        print("⚠️  Audio appears to be silent (max amplitude < 1e-6)")
    elif np.abs(audio).max() < 1e-3:
        print("⚠️  Audio appears to be very quiet (max amplitude < 1e-3)")
    else:
        print("✅ Audio has reasonable amplitude levels")
    
    # Check for any non-zero content
    non_zero = np.count_nonzero(audio)
    print(f"Non-zero samples: {non_zero}/{len(audio)} ({non_zero/len(audio)*100:.1f}%)")
    
    # Simple statistics
    print(f"Mean: {audio.mean():.6f}")
    print(f"Std: {audio.std():.6f}")
    
except Exception as e:
    print(f"Error loading audio: {e}")