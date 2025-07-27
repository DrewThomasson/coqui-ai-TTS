from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class StyleTTS2Config(BaseTTSConfig):
    """StyleTTS2 Model Configuration using existing Coqui TTS components"""
    
    model: str = "styletts2"
    
    # Model architecture parameters - optimized for existing components
    hidden_dim: int = 192  # Smaller for efficiency 
    style_dim: int = 256   # Style embedding dimension
    num_chars: int = 178   # Required by BaseTTS
    
    # Audio processing configuration compatible with AudioProcessor
    audio: Dict = field(default_factory=lambda: {
        "sample_rate": 22050,
        "hop_length": 256,
        "win_length": 1024,
        "fft_size": 1024,     # Must be >= win_length
        "num_mels": 80,       # AudioProcessor parameter naming
        "mel_fmin": 0,        # AudioProcessor parameter naming
        "mel_fmax": 8000,     # AudioProcessor parameter naming
        "power": 2.0,
        "preemphasis": 0.97,
        "ref_level_db": 20,
        "do_sound_norm": False,
        "trim_db": 60,
        "do_trim_silence": True,
        "griffin_lim_iters": 60,
    })
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 8
    eval_batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-6
    
    # Loss weights
    lambda_mel: float = 1.0
    lambda_dur: float = 1.0
    
    # Data loader settings
    num_loader_workers: int = 2
    num_eval_loader_workers: int = 2
    
    # Text processing
    text_cleaner: str = "english_cleaners"
    add_blank: bool = True
    
    # Inference parameters
    noise_scale: float = 0.667
    length_scale: float = 1.0
    
    # Multispeaker settings
    use_speaker_embedding: bool = False
    use_d_vector_file: bool = False
    
    def __post_init__(self):
        super().__post_init__()