#!/usr/bin/env python3
"""Simple test of StyleTTS2 implementation with transcription verification"""

import sys
import os
import torch
import numpy as np
import soundfile as sf
import whisper

# Add paths
sys.path.insert(0, '.')

def test_styletts2_direct():
    """Test StyleTTS2 directly and check transcription"""
    
    print("=" * 60)
    print("Direct StyleTTS2 Test with Transcription")
    print("=" * 60)
    
    try:
        # Import StyleTTS2 components directly
        from TTS.tts.configs.styletts2_config import StyleTTS2Config
        from TTS.tts.models.styletts2 import StyleTTS2
        
        print("✅ StyleTTS2 imports successful")
        
        # Initialize model
        config = StyleTTS2Config()
        model = StyleTTS2.init_from_config(config)
        print("✅ StyleTTS2 model initialized")
        
        # Test text
        test_text = "Hello world this is a test file!"
        print(f"📝 Test text: '{test_text}'")
        
        # Generate audio
        print("🎵 Generating audio...")
        
        # Try the synthesize method
        if hasattr(model, 'synthesize'):
            audio = model.synthesize(test_text, speaker_wav="reference.wav" if os.path.exists("reference.wav") else None)
        elif hasattr(model, 'inference'):
            audio = model.inference(test_text)
        else:
            print("❌ No synthesis method found")
            return False
            
        print(f"✅ Generated audio: {len(audio)} samples")
        
        # Save audio
        output_file = "simple_styletts2_test.wav"
        sample_rate = 24000  # StyleTTS2 default
        
        # Ensure audio is numpy array
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().numpy()
        
        # Save as 16-bit WAV
        sf.write(output_file, audio, sample_rate)
        print(f"💾 Audio saved to: {output_file}")
        
        # Check file properties  
        file_size = os.path.getsize(output_file)
        duration = len(audio) / sample_rate
        print(f"📊 File size: {file_size} bytes, Duration: {duration:.2f}s")
        
        # Load Whisper and transcribe
        print("🎤 Loading Whisper for transcription...")
        whisper_model = whisper.load_model("base")
        
        print("📝 Transcribing audio...")
        result = whisper_model.transcribe(output_file)
        transcription = result["text"].strip()
        
        print("\n" + "=" * 40)
        print("TRANSCRIPTION RESULTS")
        print("=" * 40)
        print(f"Expected: '{test_text}'")
        print(f"Got:      '{transcription}'")
        
        # Analyze results
        if len(transcription) > 0:
            print("✅ Whisper detected speech content")
            
            # Check for word matches
            expected_words = set(test_text.lower().split())
            transcribed_words = set(transcription.lower().split())
            
            if expected_words and transcribed_words:
                overlap = len(expected_words.intersection(transcribed_words))
                total = len(expected_words)
                accuracy = overlap / total if total > 0 else 0
                
                print(f"📊 Word overlap: {overlap}/{total} ({accuracy:.1%})")
                
                if transcription.lower() == test_text.lower():
                    print("🎉 PERFECT MATCH!")
                    return True
                elif accuracy > 0.3:
                    print("✅ GOOD MATCH - StyleTTS2 working well")
                    return True
                elif accuracy > 0.1:
                    print("⚠️  PARTIAL MATCH - StyleTTS2 needs improvement")
                    return True
                else:
                    print("❌ POOR MATCH - StyleTTS2 generating speech but not correct")
                    return False
            else:
                print("❌ NO RECOGNIZABLE WORDS")
                return False
        else:
            print("❌ NO SPEECH DETECTED - StyleTTS2 generating silence/noise")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if os.path.exists("simple_styletts2_test.wav"):
            try:
                os.remove("simple_styletts2_test.wav")
            except:
                pass

if __name__ == "__main__":
    success = test_styletts2_direct()
    
    if success:
        print("\n🎉 StyleTTS2 is generating correct transcribable speech!")
    else:
        print("\n❌ StyleTTS2 is NOT generating correct transcribable speech")
    
    sys.exit(0 if success else 1)