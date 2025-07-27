"""Real StyleTTS2 configuration file"""

from dataclasses import dataclass, field
from typing import Dict, List

from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass 
class RealStyleTTS2Config(BaseTTSConfig):
    """Defines parameters for Real StyleTTS2 model using actual pretrained weights.

    Args:
        model (str):
            Model name used for selecting the right model at initialization. Defaults to `real_styletts2`.

        # Audio specific params
        sample_rate (int):
            Sample rate for audio processing. Defaults to 24000.
        
        hop_length (int):
            Hop length for mel spectrogram. Defaults to 300.
            
        win_length (int):
            Window length for mel spectrogram. Defaults to 1200.
            
        n_fft (int):
            FFT size for mel spectrogram. Defaults to 2048.
            
        n_mels (int):
            Number of mel bins. Defaults to 80.

        # Model specific params
        model_path (str):
            Path to the pretrained StyleTTS2 model directory. If None, will be auto-downloaded.
            
        config_path (str):
            Path to the StyleTTS2 model configuration file.
            
        diffusion_steps (int):
            Number of diffusion sampling steps. Defaults to 5.
            
        embedding_scale (float):
            Scale factor for style embedding. Defaults to 1.0.

        # Text processing params
        characters (str):
            Characters used for text processing. Auto-generated from phoneme set.
            
        phonemes (str):
            Phonemes used for text processing. Based on StyleTTS2's phoneme set.
            
        phoneme_cache_path (str):
            Path to cache phonemized text.
            
        enable_eos_bos_chars (bool):
            Enable end-of-sequence and beginning-of-sequence characters.
            
        test_sentences_file (str):
            Path to test sentences file.
    """

    model: str = "real_styletts2"
    
    # Audio config
    sample_rate: int = 24000
    hop_length: int = 300
    win_length: int = 1200
    n_fft: int = 2048
    n_mels: int = 80
    
    # Model config
    model_path: str = None
    config_path: str = None
    diffusion_steps: int = 5
    embedding_scale: float = 1.0
    
    # Text processing
    characters: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!\"#&'()*+,-./:;?@[] "
    phonemes: str = "ˈˌaɪoɛɔæɑəɪʊɚɝtdeɪnslrzkθfvhjbpɡmwŋʃʒtʃdʒaɪoɛɔæɑəɪʊɚɝ"
    phoneme_cache_path: str = ""
    enable_eos_bos_chars: bool = False
    test_sentences_file: str = ""
    
    # Training (not used for pretrained)
    lr: float = 1e-4
    optimizer: str = "AdamW"
    optimizer_params: Dict = field(default_factory=lambda: {"betas": [0.9, 0.99], "eps": 1e-9, "weight_decay": 0.01})
    
    def __post_init__(self):
        super().__post_init__()
        
        # Set audio processing parameters
        if not hasattr(self, 'audio'):
            self.audio = {}
        
        self.audio.update({
            'sample_rate': self.sample_rate,
            'hop_length': self.hop_length,
            'win_length': self.win_length,
            'n_fft': self.n_fft,
            'n_mels': self.n_mels,
        })