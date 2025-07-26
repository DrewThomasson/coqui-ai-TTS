#!/usr/bin/env python3
"""
StyleTTS2 Voice Cloning Example

This script demonstrates how to use StyleTTS2 for voice cloning in Coqui TTS.
It shows various ways to clone voices from reference audio files.
"""

import os
import torch
from pathlib import Path

# Import StyleTTS2 components
from TTS.tts.configs.styletts2_config import StyleTTS2Config
from TTS.tts.models.styletts2 import Styletts2
from TTS.tts.utils.styletts2_voice_cloning import (
    StyleTTS2VoiceCloningUtils,
    quick_voice_clone,
    create_voice_cloning_utils
)


def basic_voice_cloning_example():
    """Basic voice cloning example using the model directly."""
    print("=" * 60)
    print("Basic Voice Cloning Example")
    print("=" * 60)
    
    # Initialize model
    config = StyleTTS2Config()
    model = Styletts2.init_from_config(config, [])
    
    # Example text to synthesize
    text = "Hello! This is an example of voice cloning with StyleTTS2."
    
    # Example: Clone voice (requires reference audio file)
    reference_audio = "path/to/your/reference_audio.wav"
    
    print(f"Text to synthesize: '{text}'")
    print(f"Reference audio: {reference_audio}")
    
    if os.path.exists(reference_audio):
        try:
            # Clone voice with different settings
            mel_output = model.clone_voice(
                text=text,
                reference_wav=reference_audio,
                alpha=0.3,  # Style interpolation strength
                diffusion_steps=10
            )
            
            print(f"✓ Voice cloning successful!")
            print(f"✓ Output shape: {mel_output.shape}")
            
        except Exception as e:
            print(f"✗ Voice cloning failed: {e}")
    else:
        print(f"ℹ️  Reference audio file not found. This is a demonstration of the interface.")
        print("   To use voice cloning, provide a valid reference audio file.")


def enhanced_inference_example():
    """Example using enhanced inference method with reference audio."""
    print("\n" + "=" * 60)
    print("Enhanced Inference Example")
    print("=" * 60)
    
    config = StyleTTS2Config()
    model = Styletts2.init_from_config(config, [])
    
    text = "StyleTTS2 can clone voices with remarkable quality and naturalness."
    
    # Standard inference (no voice cloning)
    print("1. Standard inference (no reference):")
    try:
        mel_standard = model.inference(text)
        print(f"   ✓ Standard synthesis: {mel_standard.shape}")
    except Exception as e:
        print(f"   ✗ Standard inference failed: {e}")
    
    # Voice cloning inference
    print("\n2. Voice cloning inference (with reference):")
    reference_audio = "path/to/reference.wav"
    
    if os.path.exists(reference_audio):
        try:
            mel_cloned = model.inference(
                text=text,
                reference_wav=reference_audio,
                alpha=0.5,
                diffusion_steps=15
            )
            print(f"   ✓ Voice cloning synthesis: {mel_cloned.shape}")
        except Exception as e:
            print(f"   ✗ Voice cloning inference failed: {e}")
    else:
        print("   ℹ️  Reference audio not found. Using standard inference.")
        print("   To enable voice cloning, provide: reference_wav='path/to/audio.wav'")


def voice_cloning_utilities_example():
    """Example using voice cloning utilities for advanced operations."""
    print("\n" + "=" * 60)
    print("Voice Cloning Utilities Example")
    print("=" * 60)
    
    config = StyleTTS2Config()
    model = Styletts2.init_from_config(config, [])
    
    # Create voice cloning utilities
    utils = StyleTTS2VoiceCloningUtils(model, config)
    
    # Example texts
    texts = [
        "Welcome to StyleTTS2 voice cloning!",
        "This technology can replicate speaking voices with high fidelity.",
        "Each sentence will sound like the reference speaker."
    ]
    
    reference_audio = "path/to/reference.wav"
    
    print("1. Simple voice cloning:")
    if os.path.exists(reference_audio):
        try:
            mel = utils.clone_voice_simple(
                text=texts[0],
                reference_audio=reference_audio,
                alpha=0.4
            )
            print(f"   ✓ Simple cloning: {mel.shape}")
        except Exception as e:
            print(f"   ✗ Simple cloning failed: {e}")
    else:
        print("   ℹ️  Reference audio not found.")
    
    print("\n2. Batch voice cloning:")
    try:
        # This would process multiple texts with the same reference
        print(f"   📝 Processing {len(texts)} texts...")
        print("   (Requires valid reference audio file)")
        
        if os.path.exists(reference_audio):
            results = utils.clone_voice_batch(
                texts=texts,
                reference_audio=reference_audio,
                output_dir="./voice_cloning_outputs"
            )
            print(f"   ✓ Batch processing: {len(results)} outputs generated")
        else:
            print("   ℹ️  Skipping batch processing (no reference audio)")
            
    except Exception as e:
        print(f"   ✗ Batch processing failed: {e}")
    
    print("\n3. Voice comparison:")
    reference_audios = [
        "path/to/speaker1.wav",
        "path/to/speaker2.wav", 
        "path/to/speaker3.wav"
    ]
    
    existing_refs = [ref for ref in reference_audios if os.path.exists(ref)]
    
    if existing_refs:
        try:
            comparisons = utils.compare_voices(
                text="Compare how different speakers sound.",
                reference_audios=existing_refs
            )
            print(f"   ✓ Voice comparison: {len(comparisons)} speakers compared")
        except Exception as e:
            print(f"   ✗ Voice comparison failed: {e}")
    else:
        print("   ℹ️  No reference audio files found for comparison")
        print("   To compare voices, provide multiple reference audio files")
    
    print("\n4. Reference audio validation:")
    for i, ref_path in enumerate(reference_audios[:2], 1):
        result = utils.validate_reference_audio(ref_path)
        status = "✓" if result["valid"] else "✗"
        print(f"   {status} Reference {i}: {result['message']}")


def quick_voice_cloning_example():
    """Example using the quick voice cloning function."""
    print("\n" + "=" * 60)
    print("Quick Voice Cloning Example")
    print("=" * 60)
    
    text = "This is the quickest way to clone a voice with StyleTTS2!"
    reference_audio = "path/to/reference.wav"
    
    print("One-liner voice cloning:")
    print(f"Text: '{text}'")
    print(f"Reference: {reference_audio}")
    
    if os.path.exists(reference_audio):
        try:
            mel_output = quick_voice_clone(
                text=text,
                reference_audio=reference_audio,
                alpha=0.3,
                diffusion_steps=10
            )
            print(f"✓ Quick voice cloning successful: {mel_output.shape}")
        except Exception as e:
            print(f"✗ Quick voice cloning failed: {e}")
    else:
        print("ℹ️  Reference audio not found. This demonstrates the interface.")
        print("   Usage: quick_voice_clone(text, reference_audio, alpha=0.3)")


def configuration_example():
    """Example showing voice cloning configuration options."""
    print("\n" + "=" * 60)
    print("Voice Cloning Configuration")
    print("=" * 60)
    
    config = StyleTTS2Config()
    
    print("Default voice cloning settings:")
    for key, value in config.voice_cloning.items():
        print(f"  • {key}: {value}")
    
    print("\nCustomizing voice cloning settings:")
    print("```python")
    print("config = StyleTTS2Config()")
    print("config.voice_cloning.update({")
    print("    'style_interpolation_alpha': 0.5,    # Stronger style transfer")
    print("    'diffusion_steps': 20,               # Higher quality")
    print("    'reference_audio_max_length': 15.0,  # Longer reference audio")
    print("    'normalize_reference': True,         # Always normalize")
    print("})")
    print("```")


def main():
    """Main function running all examples."""
    print("🎤 StyleTTS2 Voice Cloning Examples")
    print("Demonstrating voice cloning capabilities in Coqui TTS")
    
    # Run all examples
    basic_voice_cloning_example()
    enhanced_inference_example()
    voice_cloning_utilities_example()
    quick_voice_cloning_example()
    configuration_example()
    
    print("\n" + "=" * 60)
    print("🎉 Voice Cloning Examples Complete!")
    print("=" * 60)
    
    print("\n📋 Summary of Voice Cloning Methods:")
    print("  1. model.clone_voice() - Direct voice cloning method")
    print("  2. model.inference(reference_wav=...) - Enhanced inference")
    print("  3. StyleTTS2VoiceCloningUtils - Utility class for advanced operations")
    print("  4. quick_voice_clone() - One-liner function")
    
    print("\n💡 Tips for Best Results:")
    print("  • Use clear, high-quality reference audio (3-10 seconds)")
    print("  • Adjust alpha parameter to control style transfer strength")
    print("  • Higher diffusion steps may improve quality")
    print("  • Ensure reference audio matches target sampling rate")
    
    print("\n🔗 Next Steps:")
    print("  • Prepare reference audio files in supported formats")
    print("  • Experiment with different alpha values (0.1 to 1.0)")
    print("  • Try different diffusion step counts (5 to 50)")
    print("  • Use batch processing for multiple texts")


if __name__ == "__main__":
    main()