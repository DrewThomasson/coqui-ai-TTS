"""Tests for the Qwen3-TTS model integration."""

import os
import tempfile

import pytest

from TTS.config import load_config, register_config
from TTS.tts.configs.qwen3_tts_config import Qwen3TTSConfig
from TTS.tts.models import setup_model
from TTS.tts.models.qwen3_tts import (
    _CUSTOM_VOICE_SPEAKERS,
    Qwen3TTS,
    _resolve_language,
)
from TTS.utils.manage import ModelManager


def test_config_defaults():
    """Test that the config has correct default values."""
    config = Qwen3TTSConfig()
    assert config.model == "qwen3_tts"
    assert config.supports_cloning is True
    assert config.audio.sample_rate == 24000
    assert config.audio.output_sample_rate == 24000
    assert len(config.languages) == 10
    assert "English" in config.languages
    assert "Chinese" in config.languages
    assert config.temperature == 0.9
    assert config.top_k == 50
    assert config.top_p == 1.0
    assert config.max_new_tokens == 2048
    assert config.model_variant == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def test_config_custom_variant():
    """Test config with custom variant."""
    config = Qwen3TTSConfig(model_variant="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    assert config.model_variant == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def test_config_serialization():
    """Test config can be saved and loaded."""
    config = Qwen3TTSConfig(model_variant="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        config.save_json(config_path)

        loaded = load_config(config_path)
        assert loaded.model == "qwen3_tts"
        assert loaded.model_variant == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        assert loaded.audio.sample_rate == 24000
        assert loaded.languages == config.languages


def test_register_config():
    """Test that register_config can find the Qwen3-TTS config."""
    config_class = register_config("qwen3_tts")
    assert config_class is Qwen3TTSConfig


def test_model_init_from_config():
    """Test model can be initialized from config."""
    config = Qwen3TTSConfig()
    model = Qwen3TTS.init_from_config(config)
    assert isinstance(model, Qwen3TTS)
    assert model.qwen_model is None
    assert model._is_base is True
    assert model._is_custom_voice is False
    assert model._is_voice_design is False


def test_model_init_custom_voice():
    """Test model initialization for CustomVoice variant."""
    config = Qwen3TTSConfig(model_variant="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    model = Qwen3TTS.init_from_config(config)
    assert model._is_custom_voice is True
    assert model._is_base is False
    assert model._is_voice_design is False


def test_model_init_voice_design():
    """Test model initialization for VoiceDesign variant."""
    config = Qwen3TTSConfig(model_variant="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    model = Qwen3TTS.init_from_config(config)
    assert model._is_voice_design is True
    assert model._is_base is False
    assert model._is_custom_voice is False


def test_setup_model():
    """Test that setup_model() can discover and create the Qwen3TTS model."""
    config = Qwen3TTSConfig()
    model = setup_model(config)
    assert isinstance(model, Qwen3TTS)


def test_resolve_language():
    """Test language code resolution."""
    assert _resolve_language("English") == "English"
    assert _resolve_language("en") == "English"
    assert _resolve_language("zh") == "Chinese"
    assert _resolve_language("zh-cn") == "Chinese"
    assert _resolve_language("ja") == "Japanese"
    assert _resolve_language("ko") == "Korean"
    assert _resolve_language("de") == "German"
    assert _resolve_language("fr") == "French"
    assert _resolve_language("ru") == "Russian"
    assert _resolve_language("pt") == "Portuguese"
    assert _resolve_language("es") == "Spanish"
    assert _resolve_language("it") == "Italian"
    assert _resolve_language(None) == "Auto"
    # Case insensitive
    assert _resolve_language("ENGLISH") == "English"
    assert _resolve_language("chinese") == "Chinese"


def test_custom_voice_speakers():
    """Test that custom voice speakers list is populated."""
    assert len(_CUSTOM_VOICE_SPEAKERS) == 9
    assert "Ryan" in _CUSTOM_VOICE_SPEAKERS
    assert "Vivian" in _CUSTOM_VOICE_SPEAKERS


def test_synthesize_without_model_loaded():
    """Test that synthesize raises error when model not loaded."""
    config = Qwen3TTSConfig()
    model = Qwen3TTS(config)
    with pytest.raises(RuntimeError, match="Model not loaded"):
        model.synthesize("Hello")


def test_voice_clone_requires_speaker_wav():
    """Test that voice cloning raises error without speaker_wav."""
    config = Qwen3TTSConfig(model_variant="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    model = Qwen3TTS(config)
    # Set qwen_model to a non-None value to bypass the "not loaded" check
    model.qwen_model = True  # dummy to pass the None check
    with pytest.raises(ValueError, match="requires a reference audio"):
        model.synthesize("Hello", language="English")


def test_models_json_entries():
    """Test that Qwen3-TTS models are registered in .models.json."""
    manager = ModelManager(progress_bar=False)
    models = manager.list_tts_models()

    qwen_models = [m for m in models if "qwen3" in m]
    assert len(qwen_models) >= 5, f"Expected at least 5 Qwen3-TTS models, found: {qwen_models}"

    expected = [
        "tts_models/multilingual/multi-dataset/qwen3_tts",
        "tts_models/multilingual/multi-dataset/qwen3_tts_custom_voice",
        "tts_models/multilingual/multi-dataset/qwen3_tts_voice_design",
        "tts_models/multilingual/multi-dataset/qwen3_tts_0.6b",
        "tts_models/multilingual/multi-dataset/qwen3_tts_0.6b_custom_voice",
    ]
    for name in expected:
        assert name in models, f"Model {name} not found in model list"
