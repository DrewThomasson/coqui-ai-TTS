#!/usr/bin/env python3
"""
Minimal StyleTTS2 compatibility test using original simple architecture
"""
import os
import sys
import torch
import torchaudio
import logging

# Add TTS to path  
sys.path.append('/home/runner/work/coqui-ai-TTS/coqui-ai-TTS')

def create_minimal_styletts2():
    """Create a minimal StyleTTS2 that just works."""
    from TTS.tts.configs.styletts2_config import StyleTTS2Config
    from TTS.tts.models.styletts2 import StyleTTS2
    
    # Use a minimal config
    config = StyleTTS2Config()
    
    # Override problematic settings
    config.audio = {
        "sample_rate": 24000,
        "hop_length": 300,
        "win_length": 1200,
        "fft_size": 2048,      
        "num_mels": 80,        
        "mel_fmin": 0,         
        "mel_fmax": 12000,     
        "output_sample_rate": 24000,
        "do_trim_silence": True,
        "trim_db": 30
    }
    
    # Create model using simplified approach
    model = StyleTTS2(
        config=config,
        ap=None,  # Don't use AudioProcessor for now
        tokenizer=None,  # Don't use tokenizer for now
        speaker_manager=None,
        language_manager=None
    )
    
    return model, config

def test_minimal_synthesis():
    """Test minimal synthesis."""
    try:
        model, config = create_minimal_styletts2()
        print("✅ Minimal StyleTTS2 created successfully")
        
        # Test with dummy input
        batch_size = 1
        seq_len = 10
        
        # Create dummy text input
        text_input = torch.randint(0, config.n_token, (batch_size, seq_len))
        text_lengths = torch.LongTensor([seq_len])
        
        # Test forward pass
        with torch.no_grad():
            outputs = model.forward(text_input, text_lengths)
            print(f"✅ Forward pass successful: {outputs['model_outputs'].shape}")
            return True
            
    except Exception as e:
        print(f"❌ Minimal synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Minimal StyleTTS2 Test ===")
    success = test_minimal_synthesis()
    if success:
        print("✅ Minimal StyleTTS2 architecture works")
    else:
        print("❌ Even minimal architecture fails")