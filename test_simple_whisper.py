#!/usr/bin/env python3
"""
Create a simple speech-like test file and test Whisper
"""

import numpy as np
import soundfile as sf
import whisper
import os

def create_speech_like_audio():
    """Create simple speech-like audio patterns."""
    sr = 22050
    duration = 3.0  # 3 seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create speech-like patterns with clear phoneme boundaries
    audio = np.zeros_like(t)
    
    # Simulate words with distinct patterns
    words = [
        (0.0, 0.8, 200, "hello"),    # Low frequency for "hello" 
        (1.0, 1.8, 300, "world"),   # Mid frequency for "world"
        (2.2, 3.0, 250, "test"),    # Low-mid for "test"
    ]
    
    for start, end, freq, word in words:
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        
        if end_idx > len(audio):
            end_idx = len(audio)
        
        word_t = t[start_idx:end_idx]
        word_audio = np.zeros_like(word_t)
        
        # Add fundamental frequency
        word_audio += 0.5 * np.sin(2 * np.pi * freq * word_t)
        
        # Add harmonics
        word_audio += 0.3 * np.sin(2 * np.pi * freq * 2 * word_t)
        word_audio += 0.2 * np.sin(2 * np.pi * freq * 3 * word_t)
        
        # Add amplitude envelope
        envelope = np.exp(-2 * (word_t - start)**2 / (end - start)**2)
        word_audio *= envelope
        
        audio[start_idx:end_idx] = word_audio
    
    # Normalize
    audio = audio / (np.abs(audio).max() + 1e-6) * 0.8
    
    return audio, sr

def test_simple_audio_with_whisper():
    """Test Whisper with our simple audio."""
    print("🔊 Creating simple speech-like audio...")
    audio, sr = create_speech_like_audio()
    
    # Save to file
    test_file = "/tmp/simple_speech_test.wav"
    sf.write(test_file, audio, sr)
    
    print(f"✅ Saved audio to {test_file}")
    print(f"📊 Audio stats: {len(audio)} samples, {len(audio)/sr:.2f}s duration")
    print(f"   Range: [{audio.min():.3f}, {audio.max():.3f}]")
    print(f"   RMS: {np.sqrt(np.mean(audio**2)):.3f}")
    
    try:
        # Test with Whisper
        print("🎤 Testing with Whisper...")
        model = whisper.load_model("base")
        result = model.transcribe(test_file)
        transcription = result["text"].strip()
        
        print(f"📝 Whisper transcription: '{transcription}'")
        
        if transcription:
            print("✅ Whisper detected some speech-like content!")
            return True
        else:
            print("❌ Whisper detected no speech")
            return False
            
    except Exception as e:
        print(f"❌ Whisper failed: {e}")
        return False
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    test_simple_audio_with_whisper()