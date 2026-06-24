from dataclasses import dataclass, field

from TTS.config import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class StyleTTS2AudioConfig(BaseAudioConfig):
    sample_rate: int = 24000
    n_fft: int = 2048
    win_length: int = 1200
    hop_length: int = 300
    n_mels: int = 80


@dataclass
class StyleTTS2Config(BaseTTSConfig):
    """Configuration for StyleTTS2 model.

    Supports both single-speaker (LJSpeech) and multi-speaker (LibriTTS) modes.

    Args:
        model (str): Model name, must be ``"style_tts2"``.
        audio (StyleTTS2AudioConfig): Audio processing parameters.
        _supports_cloning (bool): Whether voice cloning is supported (True for multispeaker).
        multispeaker (bool): Whether the model is multi-speaker.
        use_phonemes (bool): Always ``False`` – StyleTTS2 handles phonemization internally.
        sample_rate (int): Audio sample rate (24 kHz).
        diffusion_steps (int): Number of diffusion sampling steps during inference.
        embedding_scale (float): Classifier-free guidance scale for diffusion.
        model_params (dict): StyleTTS2 model architecture parameters.
    """

    model: str = "style_tts2"
    audio: StyleTTS2AudioConfig = field(default_factory=StyleTTS2AudioConfig)

    _supports_cloning: bool = False
    multispeaker: bool = False

    # StyleTTS2 handles phonemization internally
    use_phonemes: bool = False
    phonemizer: str = None
    phoneme_language: str = "en-us"

    sample_rate: int = 24000

    # Inference parameters
    diffusion_steps: int = 5
    embedding_scale: float = 1.0

    # Multi-speaker blending parameters
    alpha: float = 0.3
    beta: float = 0.7

    # Model architecture parameters (loaded from YAML config)
    model_params: dict = field(default_factory=lambda: {
        "multispeaker": False,
        "dim_in": 64,
        "hidden_dim": 512,
        "max_conv_dim": 512,
        "n_layer": 3,
        "n_mels": 80,
        "n_token": 178,
        "max_dur": 50,
        "style_dim": 128,
        "dropout": 0.2,
        "decoder": {
            "type": "istftnet",
            "resblock_kernel_sizes": [3, 7, 11],
            "upsample_rates": [10, 6],
            "upsample_initial_channel": 512,
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_kernel_sizes": [20, 12],
            "gen_istft_n_fft": 20,
            "gen_istft_hop_size": 5,
        },
        "slm": {
            "model": "microsoft/wavlm-base-plus",
            "sr": 16000,
            "hidden": 768,
            "nlayers": 13,
            "initial_channel": 64,
        },
        "diffusion": {
            "embedding_mask_proba": 0.1,
            "transformer": {
                "num_layers": 3,
                "num_heads": 8,
                "head_features": 64,
                "multiplier": 2,
            },
            "dist": {
                "sigma_data": 0.2,
                "estimate_sigma_data": True,
                "mean": -3.0,
                "std": 1.0,
            },
        },
    })

    # ASR model configuration
    asr_config: dict = field(default_factory=lambda: {
        "input_dim": 80,
        "hidden_dim": 256,
        "n_token": 178,
        "token_embedding_dim": 512,
    })

    # PLBERT model configuration
    plbert_config: dict = field(default_factory=lambda: {
        "vocab_size": 178,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 2048,
        "max_position_embeddings": 512,
        "num_hidden_layers": 12,
        "dropout": 0.1,
    })
