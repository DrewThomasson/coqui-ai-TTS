"""StyleTTS2 configuration for Coqui TTS"""

from TTS.tts.configs.shared_configs import BaseTTSConfig


class CompleteStyleTTS2Config(BaseTTSConfig):
    """Configuration for CompleteStyleTTS2 model"""
    
    model: str = "complete_styletts2"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Model parameters
        self.num_chars = 178
        self.hidden_dim = 512
        self.style_dim = 256
        
        # Audio configuration
        if not hasattr(self, 'audio') or self.audio is None:
            self.audio = type('AudioConfig', (), {
                'sample_rate': 22050,
                'hop_length': 256,
                'win_length': 1024,
                'n_fft': 2048,
                'num_mels': 80,
                'mel_fmin': 0,
                'mel_fmax': 11025,
                'fft_size': 2048,
                'win_size': 1024,
                'hop_size': 256
            })()
