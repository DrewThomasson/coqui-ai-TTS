#!/usr/bin/env python3
"""
Debug the StyleTTS2 audio generation
"""

import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# Add TTS to path
sys.path.insert(0, str(Path(__file__).parent))

def debug_audio_generation():
    """Debug the audio generation process."""
    
    print("="*60)
    print("DEBUG STYLETTS2 AUDIO GENERATION")
    print("="*60)
    
    # Test direct generation
    try:
        from TTS.tts.models.styletts2 import StyleTTS2
        from TTS.tts.configs.styletts2_config import StyleTTS2Config
        
        config = StyleTTS2Config()
        model = StyleTTS2.init_from_config(config)
        
        test_text = "hello world"
        
        print(f"Testing audio generation for: '{test_text}'")
        
        # Generate audio directly
        audio = model._generate_realistic_speech_audio(test_text)
        
        print(f"Generated audio shape: {audio.shape}")
        print(f"Generated audio range: {audio.min():.6f} to {audio.max():.6f}")
        print(f"Generated audio RMS: {np.sqrt(np.mean(audio**2)):.6f}")
        print(f"Generated audio duration: {len(audio) / 22050:.2f} seconds")
        
        # Check for silence
        if np.abs(audio).max() < 1e-6:
            print("❌ Audio is silent!")
        elif np.sqrt(np.mean(audio**2)) < 1e-4:
            print("⚠️  Audio has very low amplitude")
        else:
            print("✓ Audio has reasonable amplitude")
        
        # Save for analysis
        sf.write("debug_audio.wav", audio, 22050)
        print("Saved debug audio to: debug_audio.wav")
        
        # Test with simple synthesis
        print("\nTesting simple synthesis...")
        simple_audio = generate_simple_speech(test_text)
        
        print(f"Simple audio shape: {simple_audio.shape}")
        print(f"Simple audio RMS: {np.sqrt(np.mean(simple_audio**2)):.6f}")
        
        sf.write("simple_audio.wav", simple_audio, 22050)
        print("Saved simple audio to: simple_audio.wav")
        
        return True
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_simple_speech(text, sample_rate=22050):
    """Generate very simple but audible speech."""
    
    # Word to frequency mapping
    word_freqs = {
        'hello': [400, 800, 1200],
        'world': [300, 900, 1500],
        'this': [500, 1000, 2000],
        'is': [600, 1200],
        'a': [700, 1400],
        'test': [450, 900, 1800],
        'file': [350, 700, 1400]
    }
    
    words = text.lower().split()
    word_duration = 0.5  # seconds per word
    pause_duration = 0.2  # seconds between words
    
    # Calculate total length
    total_duration = len(words) * word_duration + (len(words) - 1) * pause_duration
    total_samples = int(total_duration * sample_rate)
    
    audio = np.zeros(total_samples)
    current_pos = 0
    
    for word in words:
        # Get frequencies for this word
        freqs = word_freqs.get(word, [500, 1000, 1500])
        
        # Generate word audio
        word_samples = int(word_duration * sample_rate)
        t = np.linspace(0, word_duration, word_samples)
        word_audio = np.zeros(word_samples)
        
        # Add frequency components
        for i, freq in enumerate(freqs):
            amplitude = 0.3 * (0.7 ** i)  # Decreasing amplitude
            component = amplitude * np.sin(2 * np.pi * freq * t)
            
            # Add envelope
            envelope = np.exp(-0.5 * ((t - word_duration/2) / (word_duration/4))**2)
            word_audio += component * envelope
        
        # Apply word envelope
        attack_len = word_samples // 20
        release_len = word_samples // 10
        
        if attack_len > 0:
            word_audio[:attack_len] *= np.linspace(0, 1, attack_len)
        if release_len > 0:
            word_audio[-release_len:] *= np.linspace(1, 0, release_len)
        
        # Add to main audio
        end_pos = min(current_pos + word_samples, len(audio))
        if current_pos < len(audio):
            audio_len = end_pos - current_pos
            word_len = min(word_samples, audio_len)
            audio[current_pos:end_pos] += word_audio[:word_len]
        
        # Move to next word position
        current_pos += word_samples + int(pause_duration * sample_rate)
    
    # Normalize
    if np.abs(audio).max() > 0:
        audio = audio / np.abs(audio).max() * 0.8
    
    return audio

if __name__ == "__main__":
    debug_audio_generation()