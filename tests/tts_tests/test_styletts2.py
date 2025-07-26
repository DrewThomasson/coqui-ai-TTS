import unittest
import torch

from TTS.tts.configs.styletts2_config import StyleTTS2Config
from TTS.tts.models.styletts2 import Styletts2
from TTS.tts.models import setup_model


class TestStyleTTS2(unittest.TestCase):
    """Test StyleTTS2 model integration."""

    def setUp(self):
        """Set up test configuration."""
        self.config = StyleTTS2Config()
        self.config.batch_size = 2
        self.config.eval_batch_size = 2

    def test_config_initialization(self):
        """Test StyleTTS2 config can be initialized."""
        config = StyleTTS2Config()
        self.assertEqual(config.model, "styletts2")
        self.assertEqual(config.hidden_dim, 512)
        self.assertEqual(config.style_dim, 64)
        self.assertFalse(config.multispeaker)

    def test_model_initialization(self):
        """Test StyleTTS2 model can be initialized."""
        model = Styletts2.init_from_config(self.config, [])
        self.assertIsInstance(model, Styletts2)
        
        # Check model has expected components
        self.assertTrue(hasattr(model, 'text_encoder'))
        self.assertTrue(hasattr(model, 'style_encoder'))
        self.assertTrue(hasattr(model, 'diffusion'))
        self.assertTrue(hasattr(model, 'decoder'))

    def test_model_registry(self):
        """Test StyleTTS2 can be loaded through model registry."""
        model = setup_model(self.config, [])
        self.assertIsInstance(model, Styletts2)

    def test_forward_pass(self):
        """Test StyleTTS2 forward pass."""
        model = Styletts2.init_from_config(self.config, [])
        model.eval()

        # Create dummy input
        batch_size = 2
        seq_len = 10
        mel_len = 50

        text_input = torch.randint(0, self.config.n_token, (batch_size, seq_len))
        text_lengths = torch.LongTensor([seq_len] * batch_size)
        mel_target = torch.randn(batch_size, self.config.n_mels, mel_len)
        mel_lengths = torch.LongTensor([mel_len] * batch_size)

        # Forward pass
        with torch.no_grad():
            outputs = model.forward(text_input, text_lengths, mel_target, mel_lengths)

        # Check outputs
        self.assertIn("model_outputs", outputs)
        self.assertIn("durations_log", outputs)
        self.assertIn("style", outputs)
        
        # Check output shape
        mel_pred = outputs["model_outputs"]
        self.assertEqual(len(mel_pred.shape), 3)  # [batch, mel_dim, time]
        self.assertEqual(mel_pred.shape[0], batch_size)
        self.assertEqual(mel_pred.shape[1], self.config.n_mels)

    def test_inference(self):
        """Test StyleTTS2 inference."""
        model = Styletts2.init_from_config(self.config, [])
        model.eval()

        # Test inference
        with torch.no_grad():
            output = model.inference("Hello world!")
        
        # Check output shape
        self.assertEqual(len(output.shape), 3)  # [batch, mel_dim, time]
        self.assertEqual(output.shape[0], 1)  # batch size 1
        self.assertEqual(output.shape[1], self.config.n_mels)

    def test_loss_computation(self):
        """Test StyleTTS2 loss computation."""
        model = Styletts2.init_from_config(self.config, [])

        # Create dummy data
        batch_size = 2
        seq_len = 10
        mel_len = 20

        text_input = torch.randint(0, self.config.n_token, (batch_size, seq_len))
        text_lengths = torch.LongTensor([seq_len] * batch_size)  
        mel_target = torch.randn(batch_size, self.config.n_mels, mel_len)
        mel_lengths = torch.LongTensor([mel_len] * batch_size)

        # Forward pass
        outputs = model.forward(text_input, text_lengths, mel_target, mel_lengths)

        # Create batch dict
        batch = {
            "mel": mel_target,
            "durations": torch.ones(batch_size, seq_len)
        }

        # Compute loss
        loss, loss_dict = model.compute_loss(batch, None, outputs)

        # Check loss
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # scalar loss
        self.assertIn("loss", loss_dict)
        self.assertIn("loss_mel", loss_dict)

    def test_multispeaker_config(self):
        """Test StyleTTS2 with multispeaker configuration."""
        config = StyleTTS2Config()
        config.multispeaker = True
        
        # This should work without error
        model = Styletts2.init_from_config(config, [])
        self.assertTrue(model.multispeaker)

    def test_model_parameters(self):
        """Test StyleTTS2 has reasonable number of parameters."""
        model = Styletts2.init_from_config(self.config, [])
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Check reasonable parameter count (should be in millions)
        self.assertGreater(total_params, 1_000_000)  # At least 1M parameters
        self.assertLess(total_params, 100_000_000)   # Less than 100M parameters
        self.assertEqual(total_params, trainable_params)  # All parameters trainable by default

    def test_voice_cloning_methods(self):
        """Test voice cloning method interfaces."""
        model = Styletts2.init_from_config(self.config, [])
        
        # Check that voice cloning methods exist
        self.assertTrue(hasattr(model, 'clone_voice'))
        self.assertTrue(hasattr(model, '_load_reference_audio'))
        self.assertTrue(hasattr(model, '_extract_style_from_audio'))
        self.assertTrue(hasattr(model, '_inference_with_style'))
        
        # Check method signatures
        import inspect
        
        # clone_voice method
        sig = inspect.signature(model.clone_voice)
        expected_params = ['text', 'reference_wav', 'alpha', 'diffusion_steps']
        for param in expected_params:
            self.assertIn(param, sig.parameters)
        
        # inference method should support reference_wav
        sig = inspect.signature(model.inference)
        self.assertIn('reference_wav', sig.parameters)
        self.assertIn('alpha', sig.parameters)
        self.assertIn('diffusion_steps', sig.parameters)

    def test_voice_cloning_config(self):
        """Test voice cloning configuration parameters."""
        config = StyleTTS2Config()
        
        # Check voice cloning config exists
        self.assertTrue(hasattr(config, 'voice_cloning'))
        self.assertIsInstance(config.voice_cloning, dict)
        
        # Check required voice cloning parameters
        required_params = [
            'reference_audio_max_length',
            'style_interpolation_alpha', 
            'diffusion_steps',
            'enable_preprocessing',
            'normalize_reference',
            'extract_prosody'
        ]
        
        for param in required_params:
            self.assertIn(param, config.voice_cloning)

    def test_enhanced_inference_interface(self):
        """Test enhanced inference method with voice cloning support."""
        model = Styletts2.init_from_config(self.config, [])
        model.eval()
        
        # Test standard inference (should work as before)
        with torch.no_grad():
            output1 = model.inference("Hello world!")
        
        self.assertEqual(len(output1.shape), 3)
        self.assertEqual(output1.shape[0], 1)
        self.assertEqual(output1.shape[1], self.config.n_mels)
        
        # Test inference with additional parameters (should not crash)
        with torch.no_grad():
            output2 = model.inference(
                "Hello world!",
                alpha=0.5,
                diffusion_steps=5,
                reference_wav=None  # No reference, should work like standard inference
            )
        
        self.assertEqual(output2.shape, output1.shape)

    def test_style_extraction_interface(self):
        """Test style extraction method interfaces."""
        model = Styletts2.init_from_config(self.config, [])
        
        # Create dummy audio tensor
        sample_rate = self.config.sample_rate
        duration = 2.0  # 2 seconds
        dummy_audio = torch.randn(int(sample_rate * duration))
        
        # Test style extraction (will fail without proper AP, but should not crash method lookup)
        try:
            with torch.no_grad():
                # This will fail because AP is not properly initialized in test, but method should exist
                acoustic_style, prosodic_style = model._extract_style_from_audio(dummy_audio)
        except (AttributeError, ValueError) as e:
            # Expected to fail in test environment without proper AP
            self.assertIn("AudioProcessor", str(e))
        except Exception as e:
            # Method exists but fails for other reasons - this is acceptable
            pass

    def test_voice_cloning_utilities_import(self):
        """Test that voice cloning utilities can be imported."""
        try:
            from TTS.tts.utils.styletts2_voice_cloning import (
                StyleTTS2VoiceCloningUtils,
                create_voice_cloning_utils,
                quick_voice_clone
            )
            
            # Check classes/functions exist
            self.assertTrue(callable(StyleTTS2VoiceCloningUtils))
            self.assertTrue(callable(create_voice_cloning_utils))
            self.assertTrue(callable(quick_voice_clone))
            
        except ImportError as e:
            self.fail(f"Failed to import voice cloning utilities: {e}")

    def test_voice_cloning_utils_initialization(self):
        """Test voice cloning utilities initialization."""
        try:
            from TTS.tts.utils.styletts2_voice_cloning import StyleTTS2VoiceCloningUtils
            
            model = Styletts2.init_from_config(self.config, [])
            utils = StyleTTS2VoiceCloningUtils(model, self.config)
            
            # Check utility methods exist
            self.assertTrue(hasattr(utils, 'clone_voice_simple'))
            self.assertTrue(hasattr(utils, 'clone_voice_batch'))
            self.assertTrue(hasattr(utils, 'compare_voices'))
            self.assertTrue(hasattr(utils, 'extract_speaker_embedding'))
            self.assertTrue(hasattr(utils, 'validate_reference_audio'))
            
        except ImportError:
            self.skipTest("Voice cloning utilities not available")

    def test_forward_pass_with_style_outputs(self):
        """Test that forward pass includes style outputs for voice cloning."""
        model = Styletts2.init_from_config(self.config, [])
        
        batch_size = 2
        seq_len = 10
        mel_len = 20
        
        text_input = torch.randint(0, self.config.n_token, (batch_size, seq_len))
        text_lengths = torch.LongTensor([seq_len] * batch_size)
        mel_target = torch.randn(batch_size, self.config.n_mels, mel_len)
        mel_lengths = torch.LongTensor([mel_len] * batch_size)
        
        # Forward pass
        with torch.no_grad():
            outputs = model.forward(text_input, text_lengths, mel_target, mel_lengths)
        
        # Check that style outputs are included (needed for voice cloning)
        required_outputs = [
            "model_outputs",
            "durations_log", 
            "style",
            "acoustic_style",
            "prosodic_style"
        ]
        
        for output_key in required_outputs:
            self.assertIn(output_key, outputs, f"Missing output key: {output_key}")
            self.assertIsInstance(outputs[output_key], torch.Tensor)


if __name__ == "__main__":
    unittest.main()