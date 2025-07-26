from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class StyleTTS2Config(BaseTTSConfig):
    """StyleTTS2 Model Configuration"""
    
    model: str = "styletts2"
    
    # Model architecture parameters
    hidden_dim: int = 512
    style_dim: int = 64
    n_layer: int = 5
    n_token: int = 178  # Number of phoneme tokens
    num_chars: int = 178  # Required by BaseTTS
    max_dur: int = 50
    dropout: float = 0.2
    dim_in: int = 64
    n_mels: int = 80
    
    # Multispeaker settings
    multispeaker: bool = False
    
    # Decoder settings
    decoder: Dict = field(default_factory=lambda: {
        "type": "hifigan",
        "resblock_kernel_sizes": [3, 7, 11],
        "upsample_rates": [10, 8, 2, 2, 2],
        "upsample_kernel_sizes": [20, 16, 4, 4, 4],
        "upsample_initial_channel": 512,
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    })
    
    # Diffusion model settings
    diffusion: Dict = field(default_factory=lambda: {
        "transformer": {
            "num_layers": 3,
            "num_heads": 8,
            "head_features": 64,
            "multiplier": 2
        },
        "embedding_mask_proba": 0.1,
        "dist": {
            "mean": -3.0,
            "std": 1.0,
            "sigma_data": 0.2
        }
    })
    
    # Speech Language Model discriminator settings
    slm: Dict = field(default_factory=lambda: {
        "model": "microsoft/wavlm-base-plus",
        "sr": 16000,
        "hidden": 768,
        "nlayers": 13,
        "initial_channel": 64
    })
    
    # Training parameters
    epochs: int = 200
    batch_size: int = 16
    eval_batch_size: int = 16
    lr: float = 2e-4
    betas: List[float] = field(default_factory=lambda: [0.8, 0.99])
    eps: float = 1e-9
    scheduler_params: Dict = field(default_factory=lambda: {
        "milestones": [50, 100, 150],
        "gamma": 0.5
    })
    
    # Data loader settings
    num_loader_workers: int = 4
    num_eval_loader_workers: int = 4
    
    # Text processing
    text_cleaner: str = "phoneme_cleaners"
    add_blank: bool = True
    min_seq_len: int = 1
    max_seq_len: float = float("inf")
    precompute_num_workers: int = 0
    
    # Loss weights
    lambda_mel: float = 45.0
    lambda_dur: float = 1.0
    lambda_f0: float = 1.0
    lambda_gen: float = 1.0
    lambda_slm: float = 1.0
    
    # Data parameters
    sample_rate: int = 24000
    hop_length: int = 300
    win_length: int = 1200
    n_fft: int = 2048
    
    # Preprocessing
    preprocess_texts: bool = True
    phoneme_cache_path: str = ""
    
    # Pre-trained model paths
    pretrained_model: Optional[str] = None
    asr_model_path: Optional[str] = None
    asr_model_config: Optional[str] = None
    f0_model_path: Optional[str] = None
    plbert_model_path: Optional[str] = None
    
    # Inference parameters
    diffusion_steps: int = 10
    embedding_scale: float = 1.0
    alpha: float = 0.3
    
    # Voice cloning parameters
    voice_cloning: Dict = field(default_factory=lambda: {
        "reference_audio_max_length": 10.0,  # Max length in seconds
        "style_interpolation_alpha": 0.3,    # Default alpha for style interpolation
        "diffusion_steps": 10,               # Default diffusion steps for cloning
        "enable_preprocessing": True,        # Enable audio preprocessing
        "normalize_reference": True,         # Normalize reference audio
        "extract_prosody": True,             # Extract prosodic features
    })
    
    def __post_init__(self):
        super().__post_init__()