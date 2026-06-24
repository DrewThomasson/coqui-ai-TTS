"""Tests for StyleTTS2 model integration."""

import pytest
import torch

from TTS.tts.configs.style_tts2_config import StyleTTS2AudioConfig, StyleTTS2Config
from TTS.tts.models import setup_model
from TTS.tts.models.style_tts2 import StyleTTS2, _length_to_mask, _recursive_munch


class TestStyleTTS2Config:
    """Test StyleTTS2Config dataclass."""

    def test_default_config(self):
        config = StyleTTS2Config()
        assert config.model == "style_tts2"
        assert config.sample_rate == 24000
        assert config.multispeaker is False
        assert config.use_phonemes is False
        assert config.diffusion_steps == 5
        assert config.embedding_scale == 1.0
        assert config.audio.sample_rate == 24000

    def test_multispeaker_config(self):
        config = StyleTTS2Config(multispeaker=True, _supports_cloning=True)
        assert config.multispeaker is True
        assert config.supports_cloning is True
        assert config.alpha == 0.3
        assert config.beta == 0.7

    def test_audio_config(self):
        audio = StyleTTS2AudioConfig()
        assert audio.sample_rate == 24000
        assert audio.n_fft == 2048
        assert audio.win_length == 1200
        assert audio.hop_length == 300
        assert audio.n_mels == 80

    def test_model_params_default(self):
        config = StyleTTS2Config()
        assert config.model_params["n_token"] == 178
        assert config.model_params["style_dim"] == 128
        assert config.model_params["hidden_dim"] == 512
        assert config.model_params["decoder"]["type"] == "istftnet"


class TestStyleTTS2ModelInit:
    """Test StyleTTS2 model initialization."""

    def test_init_from_config(self):
        config = StyleTTS2Config()
        model = StyleTTS2.init_from_config(config)
        assert isinstance(model, StyleTTS2)
        assert model.config.model == "style_tts2"

    def test_setup_model(self):
        config = StyleTTS2Config()
        model = setup_model(config)
        assert isinstance(model, StyleTTS2)
        assert model.config.model == "style_tts2"

    def test_model_has_text_cleaner(self):
        config = StyleTTS2Config()
        model = StyleTTS2(config)
        assert model.text_cleaner is not None

    def test_model_forward_stub(self):
        config = StyleTTS2Config()
        model = StyleTTS2(config)
        # forward and inference are stubs
        assert model.forward() is None
        assert model.inference() is None

    def test_train_step_raises(self):
        config = StyleTTS2Config()
        model = StyleTTS2(config)
        with pytest.raises(NotImplementedError):
            model.train_step()


class TestUtilities:
    """Test utility functions."""

    def test_recursive_munch(self):
        d = {"a": 1, "b": {"c": 2, "d": [3, {"e": 4}]}}
        result = _recursive_munch(d)
        assert result.a == 1
        assert result.b.c == 2
        assert result.b.d[0] == 3
        assert result.b.d[1].e == 4

    def test_length_to_mask(self):
        lengths = torch.LongTensor([3, 5, 2])
        mask = _length_to_mask(lengths)
        assert mask.shape == (3, 5)
        # First row: [F, F, F, T, T] (length 3)
        assert mask[0, 0].item() is False
        assert mask[0, 2].item() is False
        assert mask[0, 3].item() is True
        # Third row: [F, F, T, T, T] (length 2)
        assert mask[2, 1].item() is False
        assert mask[2, 2].item() is True


class TestTextCleaner:
    """Test the TextCleaner from StyleTTS2 text utils."""

    def test_text_cleaner(self):
        from TTS.tts.layers.styletts2.text_utils import TextCleaner

        cleaner = TextCleaner()
        # Simple test with known characters
        result = cleaner("hello")
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(isinstance(x, int) for x in result)

    def test_text_cleaner_symbols(self):
        from TTS.tts.layers.styletts2.text_utils import symbols

        # Should have pad + punctuation + letters + IPA letters
        assert len(symbols) > 100
        assert symbols[0] == "$"  # pad character


class TestLayerImports:
    """Test that all StyleTTS2 layer modules can be imported."""

    def test_diffusion_utils(self):
        from TTS.tts.layers.styletts2.diffusion.utils import default, exists

        assert exists(1) is True
        assert exists(None) is False
        assert default(None, 5) == 5
        assert default(3, 5) == 3

    def test_diffusion_sampler(self):
        from TTS.tts.layers.styletts2.diffusion.sampler import (
            ADPM2Sampler,
            DiffusionSampler,
            KarrasSchedule,
            KDiffusion,
            LogNormalDistribution,
        )

    def test_diffusion_modules(self):
        from TTS.tts.layers.styletts2.diffusion.modules import (
            StyleTransformer1d,
            Transformer1d,
        )

    def test_diffusion_diffusion(self):
        from TTS.tts.layers.styletts2.diffusion.diffusion import (
            AudioDiffusionConditional,
        )

    def test_asr_models(self):
        from TTS.tts.layers.styletts2.Utils.ASR.models import ASRCNN

        model = ASRCNN(input_dim=80, hidden_dim=256, n_token=178)
        assert model is not None

    def test_jdc_model(self):
        from TTS.tts.layers.styletts2.Utils.JDC.model import JDCNet

        model = JDCNet(num_class=1, seq_len=192)
        assert model is not None

    def test_plbert_util(self):
        from TTS.tts.layers.styletts2.Utils.PLBERT.util import CustomAlbert

    def test_models(self):
        from TTS.tts.layers.styletts2.models import (
            ProsodyPredictor,
            StyleEncoder,
            TextEncoder,
            build_model,
        )

    def test_decoders(self):
        from TTS.tts.layers.styletts2.hifigan import Decoder as HiFiGANDecoder
        from TTS.tts.layers.styletts2.istftnet import Decoder as ISTFTDecoder
