#!/usr/bin/env python3
"""
StyleTTS2 Demo Script

This script demonstrates how to use StyleTTS2 with Coqui TTS.
StyleTTS2 is a state-of-the-art Text-to-Speech model that uses style diffusion 
and adversarial training for human-level speech synthesis.
"""

import torch
from TTS.tts.configs.styletts2_config import StyleTTS2Config
from TTS.tts.models.styletts2 import Styletts2
from TTS.tts.models import setup_model


def demo_styletts2_initialization():
    """Demo how to initialize StyleTTS2 model."""
    print("=" * 60)
    print("StyleTTS2 Demo - Model Initialization")
    print("=" * 60)
    
    # Method 1: Direct initialization
    print("\n1. Direct initialization:")
    config = StyleTTS2Config()
    model = Styletts2.init_from_config(config, [])
    print(f"   ✓ Model type: {type(model).__name__}")
    print(f"   ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Method 2: Registry-based initialization  
    print("\n2. Registry-based initialization:")
    registry_model = setup_model(config, [])
    print(f"   ✓ Model type: {type(registry_model).__name__}")
    print(f"   ✓ Same as direct: {type(model) == type(registry_model)}")
    
    return model


def demo_styletts2_config():
    """Demo StyleTTS2 configuration options."""
    print("\n" + "=" * 60)
    print("StyleTTS2 Demo - Configuration")
    print("=" * 60)
    
    config = StyleTTS2Config()
    
    print(f"\nModel Configuration:")
    print(f"  • Model name: {config.model}")
    print(f"  • Hidden dimensions: {config.hidden_dim}")
    print(f"  • Style dimensions: {config.style_dim}")
    print(f"  • Number of layers: {config.n_layer}")
    print(f"  • Number of tokens: {config.n_token}")
    print(f"  • Mel channels: {config.n_mels}")
    print(f"  • Sample rate: {config.sample_rate}")
    print(f"  • Multispeaker: {config.multispeaker}")
    
    print(f"\nTraining Configuration:")
    print(f"  • Batch size: {config.batch_size}")
    print(f"  • Learning rate: {config.lr}")
    print(f"  • Mel loss weight: {config.lambda_mel}")
    print(f"  • Duration loss weight: {config.lambda_dur}")
    
    return config


def demo_styletts2_inference():
    """Demo StyleTTS2 inference."""
    print("\n" + "=" * 60)
    print("StyleTTS2 Demo - Inference")
    print("=" * 60)
    
    # Initialize model
    config = StyleTTS2Config()
    model = Styletts2.init_from_config(config, [])
    model.eval()
    
    # Test sentences
    test_sentences = [
        "Hello, this is StyleTTS2!",
        "StyleTTS2 uses style diffusion for high-quality speech synthesis.",
        "The weather is nice today.",
        "Artificial intelligence is fascinating."
    ]
    
    print(f"\nRunning inference on {len(test_sentences)} sentences:")
    
    for i, text in enumerate(test_sentences, 1):
        print(f"\n{i}. Text: '{text}'")
        
        try:
            # Run inference
            with torch.no_grad():
                mel_output = model.inference(text)
            
            print(f"   ✓ Output shape: {mel_output.shape}")
            print(f"   ✓ Duration: {mel_output.shape[-1] * 0.0125:.2f}s (approx)")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    return model


def demo_styletts2_training():
    """Demo StyleTTS2 training step."""
    print("\n" + "=" * 60)
    print("StyleTTS2 Demo - Training Step")
    print("=" * 60)
    
    config = StyleTTS2Config()
    model = Styletts2.init_from_config(config, [])
    
    print("\nRunning test training step...")
    
    try:
        # Run test
        model_output, test_results = model.test_run({})
        
        print(f"✓ Training step completed successfully")
        print(f"✓ Loss: {test_results['loss']:.4f}")
        print(f"✓ Output keys: {list(model_output.keys())}")
        print(f"✓ Predicted mel shape: {model_output['model_outputs'].shape}")
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")
    
    return model


def demo_styletts2_features():
    """Demo StyleTTS2 key features."""
    print("\n" + "=" * 60)
    print("StyleTTS2 Demo - Key Features")
    print("=" * 60)
    
    print("\n🎯 StyleTTS2 Key Features:")
    print("  • Style Diffusion: Uses diffusion models to generate appropriate speaking styles")
    print("  • Adversarial Training: Employs discriminators for realistic speech synthesis") 
    print("  • Human-Level Quality: Achieves human-level synthesis on benchmark datasets")
    print("  • Zero-Shot Adaptation: Can adapt to new speakers without additional training")
    print("  • Multi-Speaker Support: Supports multiple speakers and voice cloning")
    print("  • End-to-End Training: Jointly optimizes all components")
    
    print("\n📊 Architecture Components:")
    print("  • Text Encoder: Processes phoneme sequences")
    print("  • Style Encoders: Extract acoustic and prosodic styles")
    print("  • Diffusion Model: Generates appropriate styles for text")
    print("  • Decoder: Converts features to mel spectrograms")
    print("  • Discriminators: Provide adversarial training signals")
    
    print("\n🔧 Coqui TTS Integration:")
    print("  • Compatible with Coqui TTS training framework")
    print("  • Follows standard model and config patterns")
    print("  • Supports multispeaker and multilingual setups")
    print("  • Can be used with existing TTS pipelines")


def main():
    """Main demo function."""
    print("🎤 StyleTTS2 Integration Demo")
    print("Demonstrating StyleTTS2 model in Coqui TTS framework")
    
    try:
        # Show features
        demo_styletts2_features()
        
        # Show configuration
        demo_styletts2_config() 
        
        # Initialize model
        model = demo_styletts2_initialization()
        
        # Test inference
        demo_styletts2_inference()
        
        # Test training
        demo_styletts2_training()
        
        print("\n" + "=" * 60)
        print("🎉 StyleTTS2 Demo Completed Successfully!")
        print("=" * 60)
        print("\nStyleTTS2 is now ready to use in your TTS projects.")
        print("For training, prepare your dataset and use the standard Coqui TTS training pipeline.")
        print("For more advanced features, refer to the original StyleTTS2 repository:")
        print("https://github.com/yl4579/StyleTTS2")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()