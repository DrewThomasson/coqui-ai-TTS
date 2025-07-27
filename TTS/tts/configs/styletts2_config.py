"""StyleTTS2 configuration file"""

from dataclasses import dataclass, field
from typing import Dict, List

from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class StyleTTS2Config(BaseTTSConfig):
    """Defines parameters for StyleTTS2 model.

    Args:
        model (str):
            Model name used for selecting the right model at initialization. Defaults to `styletts2`.

        # Model specific params
        hidden_dim (int):
            Hidden dimension for encoders. Defaults to 512.

        style_dim (int):
            Style embedding dimension. Defaults to 256.

        n_layer (int):
            Number of layers in text encoder. Defaults to 5.

        n_token (int):
            Number of tokens in vocabulary. Defaults to 178.

        max_dur (int):
            Maximum duration for training. Defaults to 50.

        dropout (float):
            Dropout rate. Defaults to 0.2.

        multispeaker (bool):
            Whether to use multi-speaker model. Defaults to False.

        # Audio processing params
        sample_rate (int):
            Audio sample rate. Defaults to 24000.

        n_mels (int):
            Number of mel frequency bins. Defaults to 80.

        n_fft (int):
            FFT size for spectrogram. Defaults to 2048.

        win_length (int):
            Window length for spectrogram. Defaults to 1200.

        hop_length (int):
            Hop length for spectrogram. Defaults to 300.

        # Model paths
        asr_model_path (str):
            Path to ASR model checkpoint. Defaults to empty string.

        asr_config_path (str):
            Path to ASR model config. Defaults to empty string.

        f0_model_path (str):
            Path to F0 model checkpoint. Defaults to empty string.

        bert_model_path (str):
            Path to BERT model. Defaults to empty string.

        checkpoint_path (str):
            Path to StyleTTS2 checkpoint. Defaults to empty string.
    """

    model: str = "styletts2"

    # Model architecture
    hidden_dim: int = 512
    style_dim: int = 256
    n_layer: int = 5
    n_token: int = 178
    max_dur: int = 50
    dropout: float = 0.2
    multispeaker: bool = False

    # Audio params
    sample_rate: int = 24000
    n_mels: int = 80
    n_fft: int = 2048
    win_length: int = 1200
    hop_length: int = 300

    # Model paths
    asr_model_path: str = ""
    asr_config_path: str = ""
    f0_model_path: str = ""
    bert_model_path: str = ""
    checkpoint_path: str = ""

    # Decoder config
    decoder: Dict = field(default_factory=lambda: {
        "type": "hifigan",
        "resblock_kernel_sizes": [3, 7, 11],
        "upsample_rates": [10, 6],
        "upsample_initial_channel": 512,
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "upsample_kernel_sizes": [20, 12]
    })

    # Diffusion config
    diffusion: Dict = field(default_factory=lambda: {
        "transformer": {
            "num_layers": 8,
            "num_heads": 8,
            "head_features": 64,
            "multiplier": 2,
        },
        "embedding_mask_proba": 0.1,
        "dist": {
            "mean": -3.0,
            "std": 1.0,
            "sigma_data": 0.2
        }
    })

    # SLM discriminator config
    slm: Dict = field(default_factory=lambda: {
        "hidden": 768,
        "nlayers": 13,
        "initial_channel": 64
    })

    def __post_init__(self):
        """Set additional parameters after initialization."""
        super().__post_init__()
        
        # Set required attributes for BaseTTS compatibility
        if not hasattr(self, 'num_chars'):
            self.num_chars = self.n_token
            
        if not hasattr(self, 'model_args'):
            self.model_args = type('', (), {})()
            self.model_args.num_chars = self.n_token