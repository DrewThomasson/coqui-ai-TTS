#!/usr/bin/env python3
"""Direct test of StyleTTS2 model implementation"""

import sys
import os
import torch
import numpy as np
import soundfile as sf
import whisper

def test_styletts2_model_direct():
    """Test StyleTTS2 model implementation directly without TTS framework imports"""
    
    print("=" * 60)
    print("Direct StyleTTS2 Model Test")
    print("=" * 60)
    
    try:
        # Import StyleTTS2 directly from the model file
        sys.path.insert(0, 'TTS/tts/models')
        sys.path.insert(0, 'TTS/tts/configs')
        
        # Try to import StyleTTS2 config and model directly
        from styletts2_config import StyleTTS2Config
        from styletts2 import StyleTTS2
        
        print("✅ StyleTTS2 direct imports successful")
        
        # Initialize model
        config = StyleTTS2Config()
        model = StyleTTS2(config)
        print("✅ StyleTTS2 model initialized")
        
        # Test text
        test_text = "Hello world this is a test file!"
        print(f"📝 Test text: '{test_text}'")
        
        # Generate some synthetic audio for testing
        print("🎵 Generating test audio...")
        
        # Create a simple synthetic audio signal for testing transcription
        sample_rate = 24000
        duration = 2.0  # 2 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate a simple speech-like signal with multiple frequencies
        # This simulates basic speech patterns for Whisper testing
        freq1 = 200 * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))  # Fundamental freq with vibrato
        freq2 = 400 * (1 + 0.05 * np.sin(2 * np.pi * 5 * t))  # Second harmonic
        freq3 = 800 * (1 + 0.03 * np.sin(2 * np.pi * 7 * t))  # Third harmonic
        
        # Create speech-like formants
        audio = (0.3 * np.sin(2 * np.pi * freq1 * t) + 
                0.2 * np.sin(2 * np.pi * freq2 * t) + 
                0.1 * np.sin(2 * np.pi * freq3 * t))
        
        # Add envelope to make it more speech-like
        envelope = np.exp(-2 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
        audio = audio * envelope
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        print(f"✅ Generated test audio: {len(audio)} samples ({len(audio)/sample_rate:.2f}s)")
        
        # Save audio
        output_file = "test_synthetic_audio.wav"
        sf.write(output_file, audio, sample_rate)
        print(f"💾 Audio saved to: {output_file}")
        
        # Load Whisper and transcribe
        print("🎤 Loading Whisper for transcription...")
        whisper_model = whisper.load_model("base")
        
        print("📝 Transcribing synthetic audio...")
        result = whisper_model.transcribe(output_file)
        transcription = result["text"].strip()
        
        print("\n" + "=" * 40)
        print("TRANSCRIPTION TEST RESULTS")
        print("=" * 40)
        print(f"Input: Synthetic speech-like signal")
        print(f"Whisper output: '{transcription}'")
        
        # Check if Whisper detects any speech content
        if len(transcription) > 0:
            print("✅ Whisper detected some audio content")
            print("⚠️  Note: This is synthetic audio, not real StyleTTS2 output")
            print("🔧 Next step: Need to test actual StyleTTS2 synthesis")
            return "synthetic_detected"
        else:
            print("❌ Whisper detected no speech content")
            print("🔧 This suggests Whisper is working but needs better audio")
            return "no_speech"
            
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("🔧 StyleTTS2 model files may have dependency issues")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        for file in ["test_synthetic_audio.wav"]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    pass

def check_styletts2_files():
    """Check what StyleTTS2 files exist and their basic structure"""
    
    print("\n" + "=" * 60)
    print("StyleTTS2 File Structure Check")
    print("=" * 60)
    
    styletts2_files = [
        "TTS/tts/models/styletts2.py",
        "TTS/tts/configs/styletts2_config.py",
        "TTS/tts/layers/styletts2/",
    ]
    
    for filepath in styletts2_files:
        if os.path.exists(filepath):
            print(f"✅ Found: {filepath}")
            if filepath.endswith('.py'):
                try:
                    with open(filepath, 'r') as f:
                        lines = f.readlines()
                        print(f"   📄 {len(lines)} lines")
                        # Check for key methods
                        content = ''.join(lines)
                        if 'def inference(' in content or 'def synthesize(' in content:
                            print(f"   🎵 Has synthesis methods")
                        if 'class StyleTTS2' in content:
                            print(f"   🏗️  Has StyleTTS2 class")
                except:
                    print(f"   ❌ Could not read file")
        else:
            print(f"❌ Missing: {filepath}")

if __name__ == "__main__":
    check_styletts2_files()
    result = test_styletts2_model_direct()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if result == "synthetic_detected":
        print("🔧 Whisper working, but need to test real StyleTTS2 output")
        print("❌ Cannot verify StyleTTS2 transcription accuracy yet")
    elif result == "no_speech":
        print("🔧 Whisper working but detecting no speech in test audio")
        print("❌ Cannot verify StyleTTS2 transcription accuracy yet")
    else:
        print("❌ StyleTTS2 testing failed due to import/dependency issues")
    
    print("\n📝 ANSWER TO USER'S QUESTION:")
    print("❌ No, the output files did NOT transcribe correctly during testing.")
    print("🔧 The current StyleTTS2 implementation has dependency issues that prevent proper testing.")
    print("⚠️  Previous claims of working transcriptions were not accurately verified.")