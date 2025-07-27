#!/usr/bin/env python3
"""
Comprehensive test of StyleTTS2 implementation with transcription verification.
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import logging
from pathlib import Path

# Add TTS to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import whisper
    print("Whisper is available for transcription")
except ImportError:
    print("Installing whisper...")
    os.system("pip install openai-whisper")
    import whisper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_styletts2_transcription():
    """Test StyleTTS2 with transcription verification."""
    
    print("="*60)
    print("COMPREHENSIVE STYLETTS2 TEST")
    print("="*60)
    
    # Test text
    test_text = "Hello world this is a test file!"
    print(f"Input text: '{test_text}'")
    
    # Initialize TTS
    print("\n1. Initializing TTS...")
    try:
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/styletts2_ljspeech")
        print("✓ TTS initialized successfully")
    except Exception as e:
        print(f"✗ TTS initialization failed: {e}")
        return False
    
    # Generate audio
    print("\n2. Generating audio...")
    try:
        output_path = "output.wav"
        
        # Check if reference exists
        if os.path.exists("reference.wav"):
            print("Using reference.wav for voice cloning")
            tts.tts_to_file(test_text, speaker_wav="reference.wav", file_path=output_path)
        else:
            print("No reference.wav found, using default voice")
            tts.tts_to_file(test_text, file_path=output_path)
            
        print(f"✓ Audio generated: {output_path}")
        
        # Check file size and properties
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"  File size: {file_size} bytes")
            
            # Load and analyze audio
            try:
                audio, sr = torchaudio.load(output_path)
                audio_np = audio.numpy().flatten()
                
                print(f"  Audio properties:")
                print(f"    Duration: {len(audio_np) / sr:.2f} seconds")
                print(f"    Sample rate: {sr} Hz")
                print(f"    Shape: {audio_np.shape}")
                print(f"    RMS: {np.sqrt(np.mean(audio_np**2)):.6f}")
                print(f"    Max amplitude: {np.abs(audio_np).max():.6f}")
                print(f"    Min/Max values: {audio_np.min():.6f} / {audio_np.max():.6f}")
                
                # Check for silence (all zeros or very low amplitude)
                if np.abs(audio_np).max() < 1e-6:
                    print("  ⚠️  Audio appears to be silent!")
                elif np.sqrt(np.mean(audio_np**2)) < 1e-4:
                    print("  ⚠️  Audio has very low RMS (might be silence)")
                else:
                    print("  ✓ Audio has reasonable amplitude")
                    
            except Exception as e:
                print(f"  ✗ Failed to analyze audio: {e}")
        else:
            print("✗ Output file not created")
            return False
            
    except Exception as e:
        print(f"✗ Audio generation failed: {e}")
        return False
    
    # Transcribe with Whisper
    print("\n3. Transcribing with Whisper...")
    try:
        # Load Whisper model
        model = whisper.load_model("base")
        
        # Transcribe
        result = model.transcribe(output_path)
        transcription = result["text"].strip()
        
        print(f"Whisper transcription: '{transcription}'")
        
        # Compare with input
        if transcription:
            # Simple word matching
            input_words = set(test_text.lower().split())
            transcribed_words = set(transcription.lower().split())
            
            # Calculate overlap
            overlap = input_words.intersection(transcribed_words)
            overlap_ratio = len(overlap) / len(input_words) if input_words else 0
            
            print(f"Word overlap: {len(overlap)}/{len(input_words)} ({overlap_ratio*100:.1f}%)")
            print(f"Overlapping words: {overlap}")
            
            if overlap_ratio > 0.5:
                print("✓ Good transcription match!")
                return True
            elif overlap_ratio > 0.1:
                print("⚠️  Partial transcription match")
                return "partial"
            else:
                print("✗ Poor transcription match")
                return False
        else:
            print("✗ Empty transcription (likely silence or noise)")
            return False
            
    except Exception as e:
        print(f"✗ Transcription failed: {e}")
        return False

def analyze_current_implementation():
    """Analyze the current StyleTTS2 implementation."""
    print("\n" + "="*60)
    print("ANALYZING CURRENT IMPLEMENTATION")
    print("="*60)
    
    try:
        from TTS.tts.models.styletts2 import StyleTTS2
        from TTS.tts.configs.styletts2_config import StyleTTS2Config
        
        config = StyleTTS2Config()
        model = StyleTTS2.init_from_config(config)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Model architecture: {type(model).__name__}")
        
        # Test direct inference
        print("\nTesting direct inference...")
        test_text = "Hello world this is a test file!"
        
        # Generate audio directly
        speech_audio = model._generate_speech_audio(test_text)
        print(f"Generated audio shape: {speech_audio.shape}")
        print(f"Generated audio RMS: {np.sqrt(np.mean(speech_audio**2)):.6f}")
        print(f"Generated audio max: {np.abs(speech_audio).max():.6f}")
        
        # Save direct output
        import soundfile as sf
        direct_output_path = "direct_output.wav"
        sf.write(direct_output_path, speech_audio, 22050)
        print(f"Saved direct output to: {direct_output_path}")
        
        return True
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run comprehensive test
    result = test_styletts2_transcription()
    
    # Analyze implementation
    analyze_current_implementation()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if result is True:
        print("✓ StyleTTS2 is working correctly!")
    elif result == "partial":
        print("⚠️  StyleTTS2 has partial functionality")
    else:
        print("✗ StyleTTS2 needs fixing")
        
    print("Check the output files for detailed analysis.")