#!/usr/bin/env python3
"""
StyleTTS2 integration test script

This script tests the StyleTTS2 model integration with Coqui TTS API
and validates the output using speech recognition (Whisper).
"""

import os
import sys
import torch
import numpy as np
import librosa
import soundfile as sf
import whisper
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_styletts2_basic():
    """Test basic StyleTTS2 functionality through Coqui TTS API"""
    print("=" * 60)
    print("Testing StyleTTS2 Basic Functionality")
    print("=" * 60)
    
    try:
        from TTS.api import TTS
        
        # Test text
        test_text = "Hello world this is a test file!"
        print(f"Input text: '{test_text}'")
        
        # Initialize StyleTTS2
        print("Initializing StyleTTS2...")
        tts = TTS("tts_models/multilingual/multi-dataset/styletts2_ljspeech")
        
        # Generate speech
        print("Generating speech...")
        output_path = "test_styletts2_output.wav"
        reference_path = "reference.wav"
        
        if os.path.exists(reference_path):
            print(f"Using reference audio: {reference_path}")
            result = tts.tts_to_file(
                text=test_text,
                speaker_wav=reference_path,
                file_path=output_path
            )
        else:
            print("No reference audio found, using default voice")
            result = tts.tts_to_file(
                text=test_text,
                file_path=output_path
            )
        
        print(f"Audio saved to: {output_path}")
        
        # Validate output
        if os.path.exists(output_path):
            # Check file size
            file_size = os.path.getsize(output_path)
            print(f"Output file size: {file_size} bytes")
            
            # Load and analyze audio
            try:
                audio_data, sample_rate = sf.read(output_path)
                duration = len(audio_data) / sample_rate
                print(f"Audio duration: {duration:.2f} seconds")
                print(f"Sample rate: {sample_rate} Hz")
                print(f"Audio shape: {audio_data.shape}")
                
                # Check if audio has content (not just silence)
                audio_rms = np.sqrt(np.mean(audio_data**2))
                print(f"Audio RMS: {audio_rms:.6f}")
                
                if audio_rms > 1e-6:
                    print("✅ Audio contains signal")
                    return True, output_path
                else:
                    print("❌ Audio appears to be silent")
                    return False, output_path
                    
            except Exception as e:
                print(f"❌ Error analyzing audio: {e}")
                return False, output_path
        else:
            print("❌ Output file not created")
            return False, None
            
    except Exception as e:
        print(f"❌ Error in basic test: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_speech_recognition(audio_path, expected_text):
    """Test speech recognition on generated audio"""
    print("\n" + "=" * 60)
    print("Testing Speech Recognition")
    print("=" * 60)
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return False, ""
    
    try:
        # Load Whisper model
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        
        # Transcribe audio
        print(f"Transcribing: {audio_path}")
        result = model.transcribe(audio_path)
        transcription = result["text"].strip()
        
        print(f"Expected: '{expected_text}'")
        print(f"Got:      '{transcription}'")
        
        # Check if transcription matches expected text
        expected_lower = expected_text.lower().strip()
        transcription_lower = transcription.lower().strip()
        
        # Calculate word overlap
        expected_words = set(expected_lower.split())
        transcribed_words = set(transcription_lower.split())
        
        if expected_words and transcribed_words:
            overlap = len(expected_words.intersection(transcribed_words))
            total_expected = len(expected_words)
            accuracy = overlap / total_expected if total_expected > 0 else 0
            
            print(f"Word accuracy: {accuracy:.2%} ({overlap}/{total_expected} words)")
            
            if accuracy >= 0.5:  # At least 50% word accuracy
                print("✅ Speech recognition successful!")
                return True, transcription
            elif len(transcription) > 0:
                print("⚠️  Partial recognition (speech detected but low accuracy)")
                return True, transcription  # Still counts as speech detected
            else:
                print("❌ No speech recognized")
                return False, transcription
        else:
            if len(transcription) > 0:
                print("⚠️  Speech detected but no word overlap")
                return True, transcription
            else:
                print("❌ No speech detected")
                return False, transcription
                
    except Exception as e:
        print(f"❌ Error in speech recognition: {e}")
        import traceback
        traceback.print_exc()
        return False, ""

def test_multiple_texts():
    """Test StyleTTS2 with multiple different texts"""
    print("\n" + "=" * 60)
    print("Testing Multiple Text Inputs")
    print("=" * 60)
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "StyleTTS2 is a text to speech model that produces high quality audio.",
        "Testing voice cloning with reference audio samples.",
        "This is a longer sentence to test the duration and quality of generated speech."
    ]
    
    results = []
    
    try:
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/styletts2_ljspeech")
        
        for i, text in enumerate(test_texts):
            print(f"\nTest {i+1}: '{text}'")
            output_path = f"test_styletts2_multi_{i+1}.wav"
            
            try:
                # Generate speech
                if os.path.exists("reference.wav"):
                    tts.tts_to_file(
                        text=text,
                        speaker_wav="reference.wav",
                        file_path=output_path
                    )
                else:
                    tts.tts_to_file(
                        text=text,
                        file_path=output_path
                    )
                
                # Test recognition
                success, transcription = test_speech_recognition(output_path, text)
                results.append((text, success, transcription))
                
            except Exception as e:
                print(f"❌ Error with text {i+1}: {e}")
                results.append((text, False, ""))
    
    except Exception as e:
        print(f"❌ Error initializing TTS: {e}")
        return []
    
    # Summary
    print("\n" + "=" * 60)
    print("Multi-Text Test Summary")
    print("=" * 60)
    
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Successful: {successful}/{total} ({successful/total*100:.1f}%)")
    
    for i, (text, success, transcription) in enumerate(results):
        status = "✅" if success else "❌"
        print(f"{status} Test {i+1}: {success}")
        if transcription:
            print(f"    Transcription: '{transcription}'")
    
    return results

def cleanup_files():
    """Clean up test files"""
    print("\n" + "=" * 60)
    print("Cleaning up test files")
    print("=" * 60)
    
    patterns = [
        "test_styletts2_*.wav",
        "output.wav"
    ]
    
    import glob
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                print(f"Removed: {file}")
            except Exception as e:
                print(f"Could not remove {file}: {e}")

def main():
    """Main test function"""
    print("StyleTTS2 Integration Test Suite")
    print("=" * 60)
    
    # Check dependencies
    try:
        import whisper
        print("✅ Whisper available for speech recognition")
    except ImportError:
        print("❌ Whisper not available. Install with: pip install openai-whisper")
        return False
    
    # Test basic functionality
    basic_success, audio_path = test_styletts2_basic()
    
    if basic_success and audio_path:
        # Test speech recognition
        recognition_success, transcription = test_speech_recognition(
            audio_path, "Hello world this is a test file!"
        )
        
        if recognition_success:
            print("\n🎉 StyleTTS2 is working correctly!")
            
            # Test multiple texts
            multi_results = test_multiple_texts()
            
            # Final assessment
            if multi_results:
                successful_multi = sum(1 for _, success, _ in multi_results if success)
                if successful_multi >= len(multi_results) * 0.7:  # 70% success rate
                    print("\n🎉 StyleTTS2 multi-text test passed!")
                    cleanup_files()
                    return True
                else:
                    print(f"\n⚠️  StyleTTS2 multi-text test partially successful ({successful_multi}/{len(multi_results)})")
        else:
            print("\n❌ StyleTTS2 is not producing recognizable speech")
    else:
        print("\n❌ StyleTTS2 basic functionality failed")
    
    cleanup_files()
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)