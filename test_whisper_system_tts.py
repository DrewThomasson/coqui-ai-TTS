#!/usr/bin/env python3
"""
Test if Whisper can transcribe system TTS output
"""

import os
import tempfile
import subprocess
import whisper

def test_whisper_with_system_tts():
    """Test Whisper with system TTS to verify Whisper is working."""
    text = "Hello world this is a test"
    
    # Generate audio using system TTS (espeak if available)
    temp_wav = "/tmp/system_tts_test.wav"
    
    try:
        # Try espeak first
        result = subprocess.run([
            "espeak", "-w", temp_wav, "-s", "150", text
        ], capture_output=True, check=True)
        print("✅ System TTS (espeak) generated audio")
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Try festival
            result = subprocess.run([
                "echo", text, "|", "festival", "--tts", "--output", temp_wav
            ], shell=True, capture_output=True, check=True)
            print("✅ System TTS (festival) generated audio")
        except:
            print("❌ No system TTS available")
            return False
    
    if os.path.exists(temp_wav):
        try:
            # Test with Whisper
            model = whisper.load_model("base")
            result = model.transcribe(temp_wav)
            transcription = result["text"].strip()
            
            print(f"🎯 Original: '{text}'")
            print(f"📝 Whisper transcription: '{transcription}'")
            
            # Check similarity
            orig_words = set(text.lower().split())
            trans_words = set(transcription.lower().split())
            overlap = len(orig_words & trans_words) / max(len(orig_words), 1)
            
            print(f"📊 Word overlap: {overlap:.2%}")
            
            if overlap > 0.3:
                print("✅ Whisper can transcribe system TTS")
                return True
            else:
                print("❌ Whisper transcription quality poor")
                return False
                
        except Exception as e:
            print(f"❌ Whisper transcription failed: {e}")
            return False
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
    else:
        print("❌ No audio file generated")
        return False

if __name__ == "__main__":
    test_whisper_with_system_tts()