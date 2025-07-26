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


if __name__ == "__main__":
    unittest.main()