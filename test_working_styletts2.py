#!/usr/bin/env python3
"""
Test the working StyleTTS2 implementation
"""
import os
import sys
import torch
import torchaudio
import numpy as np
import logging

# Add TTS to path
sys.path.append('/home/runner/work/coqui-ai-TTS/coqui-ai-TTS')

def test_working_styletts2():
    """Test the working StyleTTS2 implementation."""
    try:
        from TTS.tts.configs.styletts2_config import StyleTTS2Config
        from working_styletts2 import WorkingStyleTTS2
        
        print("Creating working StyleTTS2...")
        config = StyleTTS2Config()
        model = WorkingStyleTTS2.init_from_config(config)
        
        print("Testing basic synthesis...")
        
        # Test basic synthesis
        result = model.synthesize(
            text="Hello world this is a working test!",
            config=config
        )
        
        wav = result["wav"]
        print(f"✅ Basic synthesis successful: shape={wav.shape}")
        print(f"Audio range: [{wav.min():.4f}, {wav.max():.4f}]")
        print(f"RMS level: {np.sqrt(np.mean(wav**2)):.6f}")
        
        # Save to file
        output_path = "/home/runner/work/coqui-ai-TTS/coqui-ai-TTS/working_output.wav"
        
        # Convert to 16-bit and save
        wav_norm = wav / np.max(np.abs(wav)) * 0.95
        wav_int16 = (wav_norm * 32767).astype(np.int16)
        
        import scipy.io.wavfile as wavfile
        wavfile.write(output_path, 24000, wav_int16)
        
        print(f"✅ Audio saved to {output_path}")
        
        # Test voice cloning
        print("Testing voice cloning...")
        reference_wav = "/home/runner/work/coqui-ai-TTS/coqui-ai-TTS/reference.wav"
        if os.path.exists(reference_wav):
            try:
                result_cloned = model.synthesize(
                    text="Hello world this is voice cloning test!",
                    config=config,
                    speaker_wav=reference_wav
                )
                
                wav_cloned = result_cloned["wav"]
                print(f"✅ Voice cloning successful: shape={wav_cloned.shape}")
                
                # Save cloned audio
                cloned_path = "/home/runner/work/coqui-ai-TTS/coqui-ai-TTS/working_cloned.wav"
                wav_cloned_norm = wav_cloned / np.max(np.abs(wav_cloned)) * 0.95
                wav_cloned_int16 = (wav_cloned_norm * 32767).astype(np.int16)
                wavfile.write(cloned_path, 24000, wav_cloned_int16)
                
                print(f"✅ Cloned audio saved to {cloned_path}")
                
            except Exception as e:
                print(f"⚠️ Voice cloning failed: {e}")
        else:
            print("⚠️ Reference audio not found, skipping voice cloning test")
        
        return True
        
    except Exception as e:
        print(f"❌ Working StyleTTS2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_whisper():
    """Test the output with Whisper."""
    try:
        import whisper
        
        output_path = "/home/runner/work/coqui-ai-TTS/coqui-ai-TTS/working_output.wav"
        if not os.path.exists(output_path):
            print("❌ No audio file to test with Whisper")
            return False
        
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        
        print("Transcribing generated audio...")
        result = model.transcribe(output_path)
        
        transcription = result['text'].strip()
        print(f"Whisper transcription: '{transcription}'")
        
        # Check if transcription makes sense
        expected_words = ["hello", "world", "working", "test"]
        transcribed_lower = transcription.lower()
        
        matches = sum(1 for word in expected_words if word in transcribed_lower)
        print(f"Expected words found: {matches}/{len(expected_words)}")
        
        if matches >= 2:
            print("✅ Transcription indicates real speech!")
            return True
        else:
            print("⚠️ Transcription doesn't match expected text")
            return False
            
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Working StyleTTS2 Test ===")
    
    # Install scipy if needed
    try:
        import scipy.io.wavfile
    except ImportError:
        print("Installing scipy...")
        os.system("pip install scipy")
        import scipy.io.wavfile
    
    success = test_working_styletts2()
    
    if success:
        print("\n=== Testing with Whisper ===")
        whisper_success = test_with_whisper()
        
        if whisper_success:
            print("\n🎉 OVERALL SUCCESS: Working StyleTTS2 produces intelligible speech!")
        else:
            print("\n⚠️ PARTIAL SUCCESS: Working StyleTTS2 runs but audio quality needs improvement")
    else:
        print("\n❌ OVERALL FAILURE: Working StyleTTS2 implementation failed")
    
    print("\n=== Test Complete ===")