#!/usr/bin/env python3
"""
Test StyleTTS2 with existing pre-trained TTS components integration
"""

import os
import sys
import torch
import whisper

# Add the current directory to the path
sys.path.insert(0, '/home/runner/work/coqui-ai-TTS/coqui-ai-TTS')

try:
    from TTS.api import TTS
    print("✅ TTS API imported successfully")
except Exception as e:
    print(f"❌ Failed to import TTS API: {e}")
    sys.exit(1)

def test_styletts2_with_pretrained():
    """Test StyleTTS2 implementation with pre-trained components and validate with Whisper."""
    
    print("\n🔄 Testing StyleTTS2 with pre-trained TTS components...")
    
    # Test input text
    test_text = "Hello world this is a test of StyleTTS2 speech synthesis using existing components."
    print(f"📝 Input text: '{test_text}'")
    
    # Reference audio path
    reference_wav = "/home/runner/work/coqui-ai-TTS/coqui-ai-TTS/reference.wav"
    output_wav = "/home/runner/work/coqui-ai-TTS/coqui-ai-TTS/test_output_pretrained.wav"
    
    # Clean up previous outputs
    if os.path.exists(output_wav):
        os.remove(output_wav)
    
    try:
        # Initialize StyleTTS2
        print("\n🔧 Loading StyleTTS2 model with pre-trained components...")
        tts = TTS("tts_models/multilingual/multi-dataset/styletts2_ljspeech")
        print("✅ StyleTTS2 loaded successfully")
        
        # Generate speech
        print("\n🎵 Generating speech with voice cloning...")
        tts.tts_to_file(
            text=test_text,
            speaker_wav=reference_wav,
            file_path=output_wav
        )
        
        if os.path.exists(output_wav):
            print("✅ Audio file generated successfully")
            file_size = os.path.getsize(output_wav)
            print(f"📊 Output file size: {file_size} bytes")
            
            # Load Whisper for transcription
            print("\n🎤 Loading Whisper for transcription...")
            model = whisper.load_model("base")
            
            # Transcribe the generated audio
            print("🔍 Transcribing generated audio...")
            result = model.transcribe(output_wav)
            transcription = result["text"].strip()
            
            # Compare results
            print(f"\n📝 Input text:       '{test_text}'")
            print(f"🎯 Transcription:    '{transcription}'")
            
            # Check similarity
            input_words = set(test_text.lower().split())
            transcription_words = set(transcription.lower().split())
            
            # Calculate word overlap
            common_words = input_words & transcription_words
            similarity = len(common_words) / max(len(input_words), 1)
            
            print(f"📈 Word similarity:  {similarity:.2%}")
            print(f"🔗 Common words:     {sorted(list(common_words))}")
            
            # Check for key words that should be present
            key_words = {'hello', 'world', 'test', 'styletts2', 'speech', 'synthesis'}
            found_key_words = key_words & transcription_words
            key_word_coverage = len(found_key_words) / len(key_words)
            
            print(f"🔑 Key word coverage: {key_word_coverage:.2%}")
            print(f"✅ Found key words:  {sorted(list(found_key_words))}")
            
            # Success criteria: at least 40% word similarity OR 50% key word coverage
            if similarity >= 0.4 or key_word_coverage >= 0.5:
                print("✅ PASS: StyleTTS2 with pre-trained components generated recognizable speech!")
                return True
            else:
                print("❌ FAIL: Transcription doesn't sufficiently match input text")
                print(f"   Need either 40% word similarity (got {similarity:.1%}) or 50% key word coverage (got {key_word_coverage:.1%})")
                return False
                
        else:
            print("❌ FAIL: No output audio file generated")
            return False
            
    except Exception as e:
        print(f"❌ FAIL: Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 StyleTTS2 Pre-trained Components Integration Test")
    print("=" * 60)
    
    success = test_styletts2_with_pretrained()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 StyleTTS2 pre-trained integration test PASSED!")
    else:
        print("💥 StyleTTS2 pre-trained integration test FAILED!")
    
    sys.exit(0 if success else 1)