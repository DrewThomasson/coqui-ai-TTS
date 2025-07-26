import json
import os
import re
from typing import Any, Union

import fsspec
import yaml
from coqpit import Coqpit

from TTS.config.shared_configs import *
from TTS.utils.generic_utils import find_module


def read_json_with_comments(json_path):
    """for backward compat."""
    # fallback to json
    with fsspec.open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
    # handle comments but not urls with //
    input_str = re.sub(
        r"(\"(?:[^\"\\]|\\.)*\")|(/\*(?:.|[\\n\\r])*?\*/)|(//.*)", lambda m: m.group(1) or m.group(2) or "", input_str
    )
    return json.loads(input_str)


def register_config(model_name: str) -> Coqpit:
    """Find the right config for the given model name.

    Args:
        model_name (str): Model name.

    Raises:
        ModuleNotFoundError: No matching config for the model name.

    Returns:
        Coqpit: config class.
    """
    config_class = None
    config_name = model_name + "_config"

    # TODO: fix this
    if model_name == "xtts":
        from TTS.tts.configs.xtts_config import XttsConfig

        config_class = XttsConfig
    paths = ["TTS.tts.configs", "TTS.vocoder.configs", "TTS.encoder.configs", "TTS.vc.configs"]
    for path in paths:
        try:
            config_class = find_module(path, config_name)
        except ModuleNotFoundError:
            pass
    if config_class is None:
        raise ModuleNotFoundError(f" [!] Config for {model_name} cannot be found.")
    return config_class


def _convert_styletts2_config(config_dict: dict) -> dict:
    """Convert original StyleTTS2 config format to Coqui TTS format.
    
    Args:
        config_dict (dict): Original StyleTTS2 config dictionary.
        
    Returns:
        dict: Converted config dictionary compatible with Coqui TTS.
    """
    # Create a new config dict with Coqui TTS structure
    converted_config = {
        "model": "styletts2",
        "run_name": config_dict.get("log_dir", "styletts2_run"),
        "run_description": "StyleTTS2 model",
    }
    
    # Audio settings
    converted_config["audio"] = {
        "sample_rate": config_dict.get("sr", 24000),
        "hop_length": config_dict.get("hop_length", 300),
        "win_length": config_dict.get("win_length", 1200),
        "n_fft": config_dict.get("n_fft", 2048),
        "n_mels": config_dict.get("n_mels", 80),
        "mel_fmin": config_dict.get("mel_fmin", 0),
        "mel_fmax": config_dict.get("mel_fmax", 12000)
    }
    
    # Model arguments
    converted_config["model_args"] = {
        "hidden_dim": config_dict.get("hidden_dim", 512),
        "style_dim": config_dict.get("style_dim", 64),
        "n_layer": config_dict.get("n_layer", 5),
        "n_token": config_dict.get("n_token", 178),
        "num_chars": config_dict.get("n_token", 178),
        "max_dur": config_dict.get("max_dur", 50),
        "dropout": config_dict.get("dropout", 0.2),
        "dim_in": config_dict.get("dim_in", 64),
        "n_mels": config_dict.get("n_mels", 80),
        "multispeaker": config_dict.get("multispeaker", False),
    }
    
    # Training parameters
    converted_config.update({
        "epochs": config_dict.get("epochs", 200),
        "batch_size": config_dict.get("batch_size", 16),
        "eval_batch_size": config_dict.get("batch_size", 16),
        "lr": config_dict.get("lr", 0.0002),
        "num_loader_workers": config_dict.get("num_workers", 4),
        "num_eval_loader_workers": config_dict.get("num_workers", 4),
    })
    
    # Text processing
    converted_config.update({
        "text_cleaner": "phoneme_cleaners",
        "add_blank": True,
        "min_seq_len": 1,
        "max_seq_len": float("inf"),
    })
    
    # Voice cloning and inference settings
    converted_config["voice_cloning"] = {
        "reference_audio_max_length": 10.0,
        "style_interpolation_alpha": 0.3,
        "diffusion_steps": 10,
        "enable_preprocessing": True,
        "normalize_reference": True,
        "extract_prosody": True
    }
    
    converted_config["inference"] = {
        "diffusion_steps": 10,
        "embedding_scale": 1.0,
        "alpha": 0.3
    }
    
    return converted_config


def _is_styletts2_config(config_dict: dict) -> bool:
    """Check if config dictionary is from original StyleTTS2 format.
    
    Args:
        config_dict (dict): Config dictionary to check.
        
    Returns:
        bool: True if this looks like a StyleTTS2 config.
    """
    # StyleTTS2 configs have these characteristic fields
    styletts2_indicators = [
        "log_dir", "save_freq", "device", "epochs", "batch_size", 
        "lr", "n_token", "ASR_config", "ASR_path", "F0_path"
    ]
    
    # Check if at least 3 StyleTTS2-specific fields are present
    indicator_count = sum(1 for indicator in styletts2_indicators if indicator in config_dict)
    return indicator_count >= 3


def _process_model_name(config_dict: dict) -> str:
    """Format the model name as expected. It is a band-aid for the old `vocoder` model names.

    Args:
        config_dict (dict): A dictionary including the config fields.

    Returns:
        str: Formatted modelname.
    """
    # Handle StyleTTS2 configs that don't have explicit model field
    if "model" in config_dict:
        model_name = config_dict["model"]
    elif "generator_model" in config_dict:
        model_name = config_dict["generator_model"]
    else:
        # Check if this looks like a StyleTTS2 config
        if _is_styletts2_config(config_dict):
            model_name = "styletts2"
        else:
            raise KeyError("Config file must contain either 'model' or 'generator_model' field, or be a recognizable StyleTTS2 config")
    
    model_name = model_name.replace("_generator", "").replace("_discriminator", "")
    return model_name


def load_config(config_path: str | os.PathLike[Any]) -> Coqpit:
    """Import `json` or `yaml` files as TTS configs. First, load the input file as a `dict` and check the model name
    to find the corresponding Config class. Then initialize the Config.

    Args:
        config_path (str): path to the config file.

    Raises:
        TypeError: given config file has an unknown type.

    Returns:
        Coqpit: TTS config object.
    """
    config_path = str(config_path)
    config_dict = {}
    ext = os.path.splitext(config_path)[1]
    if ext in (".yml", ".yaml"):
        with fsspec.open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif ext == ".json":
        try:
            with fsspec.open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.decoder.JSONDecodeError:
            # backwards compat.
            data = read_json_with_comments(config_path)
    else:
        raise TypeError(f" [!] Unknown config file type {ext}")
    config_dict.update(data)
    
    # Convert StyleTTS2 configs to Coqui TTS format if needed
    if _is_styletts2_config(config_dict):
        config_dict = _convert_styletts2_config(config_dict)
    
    model_name = _process_model_name(config_dict)
    config_class = register_config(model_name.lower())
    config = config_class()
    config.from_dict(config_dict)
    return config


def check_config_and_model_args(config, arg_name, value):
    """Check the give argument in `config.model_args` if exist or in `config` for
    the given value.

    Return False if the argument does not exist in `config.model_args` or `config`.
    This is to patch up the compatibility between models with and without `model_args`.

    TODO: Remove this in the future with a unified approach.
    """
    if hasattr(config, "model_args"):
        if arg_name in config.model_args:
            return config.model_args[arg_name] == value
    if hasattr(config, arg_name):
        return config[arg_name] == value
    return False


def get_from_config_or_model_args(config, arg_name):
    """Get the given argument from `config.model_args` if exist or in `config`."""
    if hasattr(config, "model_args"):
        if arg_name in config.model_args:
            return config.model_args[arg_name]
    return config[arg_name]


def get_from_config_or_model_args_with_default(config, arg_name, def_val):
    """Get the given argument from `config.model_args` if exist or in `config`."""
    if hasattr(config, "model_args"):
        if arg_name in config.model_args:
            return config.model_args[arg_name]
    if hasattr(config, arg_name):
        return config[arg_name]
    return def_val
