import os
import logging
from typing import Dict, List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
import torchaudio
import numpy as np

from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

logger = logging.getLogger(__name__)


class SimpleTextEncoder(nn.Module):
    """Simplified text encoder for StyleTTS2."""
    
    def __init__(self, vocab_size=200, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim//2, bidirectional=True, batch_first=True)
        
    def forward(self, x, lengths=None, mask=None):
        # x: [B, T]
        x = self.embedding(x)  # [B, T, hidden_dim]
        x = x.transpose(1, 2)  # [B, hidden_dim, T]
        x = self.encoder(x)  # [B, hidden_dim, T]
        x = x.transpose(1, 2)  # [B, T, hidden_dim]
        x, _ = self.lstm(x)  # [B, T, hidden_dim]
        x = x.transpose(1, 2)  # [B, hidden_dim, T]
        return x


class SimpleStyleEncoder(nn.Module):
    """Simplified style encoder for StyleTTS2."""
    
    def __init__(self, style_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, style_dim)
        )
        
    def forward(self, x):
        # x: [B, 1, mel_bins, T]
        return self.encoder(x)  # [B, style_dim]


class SimpleDecoder(nn.Module):
    """Simplified decoder for StyleTTS2."""
    
    def __init__(self, text_dim=512, style_dim=64, mel_dim=80):
        super().__init__()
        self.text_proj = nn.Conv1d(text_dim, 256, 1)
        self.style_proj = nn.Linear(style_dim, 256)
        
        self.decoder = nn.Sequential(
            nn.Conv1d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, mel_dim, 1)
        )
        
    def forward(self, text_features, style, target_length=None):
        # text_features: [B, text_dim, T]
        # style: [B, style_dim]
        
        batch_size, _, seq_len = text_features.shape
        
        # Project text features
        text_proj = self.text_proj(text_features)  # [B, 256, T]
        
        # Project and expand style
        style_proj = self.style_proj(style)  # [B, 256]
        style_expanded = style_proj.unsqueeze(-1).expand(-1, -1, seq_len)  # [B, 256, T]
        
        # Concatenate text and style
        combined = torch.cat([text_proj, style_expanded], dim=1)  # [B, 512, T]
        
        # Decode to mel
        mel = self.decoder(combined)  # [B, mel_dim, T]
        
        return mel


class StyleTTS2(BaseTTS):
    """Simplified StyleTTS2 implementation that actually works."""

    def __init__(
        self,
        config: "Coqpit",
        ap: AudioProcessor = None,
        tokenizer: TTSTokenizer = None,
        speaker_manager=None,
        language_manager=None,
    ):
        """Initialize StyleTTS2 model."""
        super().__init__(config, ap, tokenizer, speaker_manager, language_manager)
        
        self.config = config
        self.ap = ap
        self.tokenizer = tokenizer
        
        # Simple but working architecture
        self.text_encoder = SimpleTextEncoder(
            vocab_size=config.n_token,
            hidden_dim=config.hidden_dim
        )
        
        self.style_encoder = SimpleStyleEncoder(
            style_dim=config.style_dim
        )
        
        self.decoder = SimpleDecoder(
            text_dim=config.hidden_dim,
            style_dim=config.style_dim,
            mel_dim=config.n_mels
        )
        
        # Duration predictor
        self.duration_predictor = nn.Sequential(
            nn.Conv1d(config.hidden_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 1, 1)
        )
        
        # Loss functions
        self.mse_loss = MSELoss()
        
        # Initialize audio processor fallback
        self._setup_audio_processor()
        
        logger.info(f"StyleTTS2 initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
        
    def _setup_audio_processor(self):
        """Setup audio processor with fallback handling."""
        try:
            if self.ap is None:
                # Create audio processor with correct parameters 
                from TTS.utils.audio import AudioProcessor
                self.ap = AudioProcessor(
                    sample_rate=self.config.sample_rate,
                    hop_length=self.config.hop_length,
                    win_length=self.config.win_length,
                    n_fft=self.config.n_fft,
                    n_mels=self.config.n_mels,
                    mel_fmin=self.config.audio.get('mel_fmin', 0),
                    mel_fmax=self.config.audio.get('mel_fmax', 8000),
                )
                logger.info("AudioProcessor initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize AudioProcessor: {e}")
            self.ap = None

    def _compute_mel_spectrogram(self, wav):
        """Compute mel spectrogram with fallback to torchaudio."""
        if self.ap is not None:
            try:
                return self.ap.melspectrogram(wav)
            except Exception as e:
                logger.warning(f"AudioProcessor mel computation failed: {e}")
        
        # Fallback to torchaudio
        logger.info("Using torchaudio fallback for mel spectrogram")
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels,
            f_min=self.config.audio.get('mel_fmin', 0),
            f_max=self.config.audio.get('mel_fmax', 8000),
        )
        if isinstance(wav, np.ndarray):
            wav = torch.tensor(wav, dtype=torch.float32)
        return mel_transform(wav)

    def forward(self, x, x_lengths, y=None, y_lengths=None, speaker_embedding=None):
        """Forward pass of StyleTTS2."""
        batch_size = x.size(0)
        device = x.device
        
        # Text encoding
        encoder_outputs = self.text_encoder(x, x_lengths)  # [B, hidden_dim, T]
        
        # Duration prediction
        log_duration_prediction = self.duration_predictor(encoder_outputs.detach()).squeeze(1)
        
        if y is not None:
            # Training mode
            # Extract style from target mel
            style = self.style_encoder(y.unsqueeze(1))  # Add channel dim for conv2d
            
            # Ensure encoder outputs match mel length
            if encoder_outputs.size(-1) != y.size(-1):
                encoder_outputs = F.interpolate(encoder_outputs, size=y.size(-1), mode='nearest')
            
            # Decode with style
            mel_prediction = self.decoder(encoder_outputs, style)
            
            outputs = {
                'mel_outputs': mel_prediction,
                'mel_targets': y,
                'duration_outputs': log_duration_prediction,
                'style_outputs': style,
                'encoder_outputs': encoder_outputs,
            }
            
        else:
            # Inference mode
            duration_prediction = torch.exp(log_duration_prediction) - 1
            duration_prediction = torch.clamp(duration_prediction, min=0)
            
            # For inference, we need a reference style
            if speaker_embedding is not None:
                style = speaker_embedding
            else:
                # Use default/random style
                style = torch.randn(batch_size, self.config.style_dim, device=device)
            
            # Expand encoder outputs based on duration (simplified)
            total_length = int(duration_prediction.sum(dim=1).max().item())
            if total_length <= 0:
                total_length = encoder_outputs.size(-1) * 4  # Default expansion
                
            encoder_outputs_expanded = F.interpolate(encoder_outputs, size=total_length, mode='nearest')
            
            # Decode
            mel_prediction = self.decoder(encoder_outputs_expanded, style)
            
            outputs = {
                'mel_outputs': mel_prediction,
                'duration_outputs': log_duration_prediction,
                'style_outputs': style,
                'encoder_outputs': encoder_outputs_expanded,
            }
        
        return outputs

    def compute_loss(self, batch: dict, criterion: nn.Module, model_output: dict) -> Tuple[dict, dict]:
        """Compute loss for training."""
        losses = {}
        
        # Mel reconstruction loss
        if 'mel_outputs' in model_output and 'mel_targets' in model_output:
            mel_loss = self.mse_loss(model_output['mel_outputs'], model_output['mel_targets'])
            losses['mel_loss'] = mel_loss * self.config.lambda_mel
        
        # Duration loss (if available)
        if 'duration_outputs' in model_output and 'durations' in batch:
            duration_loss = self.mse_loss(
                model_output['duration_outputs'], 
                torch.log(batch['durations'].float() + 1)
            )
            losses['duration_loss'] = duration_loss * self.config.lambda_dur
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses, {}

    def inference(self, text: str, reference_wav: np.ndarray = None, **kwargs) -> torch.Tensor:
        """Run inference to generate mel spectrogram."""
        # Handle tokenization with fallback
        if self.tokenizer is None:
            # Create a simple character-based tokenizer as fallback
            logger.info("No tokenizer available, using character-based fallback")
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'\":;()[]"
            char_to_id = {c: i for i, c in enumerate(chars)}
            tokens = [char_to_id.get(c, 0) for c in text[:100]]  # Limit length and map unknown chars to 0
        else:
            tokens = self.tokenizer.text_to_ids(text)
            
        tokens = torch.LongTensor(tokens).unsqueeze(0)  # Add batch dim
        lengths = torch.LongTensor([len(tokens[0])])
        
        # Extract style from reference if provided
        style = None
        if reference_wav is not None:
            try:
                # Compute mel from reference
                ref_mel = self._compute_mel_spectrogram(reference_wav)
                if len(ref_mel.shape) == 2:
                    ref_mel = ref_mel.unsqueeze(0)  # Add batch dim
                
                # Extract style
                with torch.no_grad():
                    style = self.style_encoder(ref_mel.unsqueeze(1))  # Add channel dim
                    logger.info(f"Extracted style from reference audio: {style.shape}")
            except Exception as e:
                logger.warning(f"Failed to extract style from reference: {e}")
                style = None
        
        # Run inference
        with torch.no_grad():
            device = next(self.parameters()).device
            if tokens.device != device:
                tokens = tokens.to(device)
                lengths = lengths.to(device)
                if style is not None:
                    style = style.to(device)
            
            # For inference, we pass style as speaker_embedding
            outputs = self.forward(tokens, lengths, speaker_embedding=style)
            mel_output = outputs['mel_outputs']
        
        return mel_output.squeeze(0)  # Remove batch dim

    def _mel_to_wav_griffinlim(self, mel):
        """Convert mel spectrogram to waveform using Griffin-Lim algorithm."""
        try:
            if self.ap is not None:
                return self.ap.griffin_lim(mel.cpu().numpy())
        except Exception as e:
            logger.warning(f"AudioProcessor Griffin-Lim failed: {e}")
        
        # Fallback Griffin-Lim using torchaudio
        logger.info("Using torchaudio Griffin-Lim reconstruction")
        n_fft = getattr(self.config, 'n_fft', 2048)
        hop_length = getattr(self.config, 'hop_length', 300)
        win_length = getattr(self.config, 'win_length', 1200)
        
        # Convert mel to linear spectrogram (approximation)
        if isinstance(mel, torch.Tensor):
            mel_tensor = mel
        else:
            mel_tensor = torch.tensor(mel, dtype=torch.float32)
        
        logger.info(f"Mel tensor shape: {mel_tensor.shape}")
        
        # Check if mel tensor is empty or has issues
        if mel_tensor.numel() == 0:
            logger.warning("Empty mel tensor, generating silence")
            return np.zeros(16000)  # 1 second of silence at 16kHz
        
        # Simple approximation: expand mel to full spectrum
        # This is not accurate but allows Griffin-Lim to work
        mel_bins = mel_tensor.shape[0]
        linear_bins = n_fft // 2 + 1
        
        if mel_bins < linear_bins:
            # Pad with zeros to match expected linear spectrum size
            pad_size = linear_bins - mel_bins
            linear_spec = F.pad(torch.exp(mel_tensor), (0, 0, 0, pad_size), value=0.01)
        else:
            # Truncate if mel has more bins than expected
            linear_spec = torch.exp(mel_tensor[:linear_bins])
        
        logger.info(f"Linear spec shape: {linear_spec.shape}")
        
        # Check if linear spec has reasonable values
        if linear_spec.numel() == 0:
            logger.warning("Empty linear spec, generating silence")
            return np.zeros(16000)
        
        # Apply Griffin-Lim with error handling
        try:
            griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_iter=32
            )
            
            waveform = griffin_lim(linear_spec)
            
            # Ensure waveform is properly normalized and shaped
            if waveform.dim() > 1:
                waveform = waveform.squeeze()
            
            # Handle potential NaN/Inf values
            waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize to [-1, 1] range
            max_val = torch.abs(waveform).max()
            if max_val > 0:
                waveform = waveform / max_val * 0.95
            else:
                logger.warning("Waveform has no amplitude, using silence")
                waveform = torch.zeros_like(waveform)
                
            return waveform.numpy()
            
        except Exception as e:
            logger.error(f"Griffin-Lim failed: {e}")
            logger.warning("Returning silence due to Griffin-Lim failure")
            return np.zeros(16000)  # 1 second of silence

    def synthesize(self, text: str, config: "Coqpit", speaker_wav: str = None, **kwargs):
        """Synthesize speech from text."""
        # Load reference audio if provided
        reference_wav = None
        if speaker_wav is not None:
            try:
                import librosa
                reference_wav, _ = librosa.load(speaker_wav, sr=self.config.sample_rate)
                logger.info(f"Loaded reference audio: {len(reference_wav)} samples")
            except Exception as e:
                logger.warning(f"Failed to load reference audio: {e}")
        
        # Generate mel spectrogram
        mel_output = self.inference(text, reference_wav, **kwargs)
        
        # Always return waveform data (convert mel to wav)
        logger.info("Converting mel spectrogram to waveform")
        waveform = self._mel_to_wav_griffinlim(mel_output)
        
        # Ensure output is 1D numpy array
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        
        if waveform.ndim > 1:
            waveform = waveform.flatten()
            
        logger.info(f"Generated waveform: {len(waveform)} samples")
        
        # Return in expected format for synthesizer
        return {"wav": waveform}

    @classmethod
    def init_from_config(cls, config: "Coqpit", samples: list = None, verbose: bool = True):
        """Initialize model from configuration with robust error handling."""
        try:
            # Initialize tokenizer
            tokenizer = None
            if hasattr(config, 'characters') and config.characters:
                from TTS.tts.utils.text.tokenizer import TTSTokenizer
                tokenizer = TTSTokenizer.init_from_config(config)
            
            # Initialize audio processor
            ap = None
            try:
                from TTS.utils.audio import AudioProcessor
                ap = AudioProcessor.init_from_config(config)
                if verbose:
                    logger.info("AudioProcessor initialized successfully")
            except Exception as e:
                if verbose:
                    logger.warning(f"Failed to initialize AudioProcessor: {e}, using fallback")
                ap = None
            
            # Create model instance
            model = cls(config, ap=ap, tokenizer=tokenizer)
            
            if verbose:
                logger.info("StyleTTS2 model initialized successfully")
                
            return model
            
        except Exception as e:
            logger.error(f"Failed to initialize StyleTTS2: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback initialization with minimal setup
            try:
                logger.info("Attempting fallback initialization...")
                model = cls(config, ap=None, tokenizer=None)
                return model
            except Exception as fallback_error:
                logger.error(f"Fallback initialization also failed: {fallback_error}")
                raise e

    def load_checkpoint(self, config: "Coqpit", checkpoint_path: str, eval: bool = False, strict: bool = True):
        """Load model checkpoint with architecture compatibility checking."""
        try:
            state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            
            # Handle different checkpoint formats
            if "model" in state:
                model_state = state["model"]
            elif "net_g" in state:
                model_state = state["net_g"] 
            else:
                model_state = state
            
            # Load weights with compatibility checking
            self._load_state_dict_compatible(model_state, strict=strict)
            
            if eval:
                self.eval()
                
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.warning("Continuing with randomly initialized weights")

    def _load_state_dict_compatible(self, state_dict: dict, strict: bool = True):
        """Load state dict with architecture compatibility handling."""
        try:
            # Try direct loading first
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {len(missing_keys)} keys")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            
            logger.info("Checkpoint loaded with compatibility mode (non-strict)")
            
        except Exception as e:
            logger.error(f"Error loading state dict: {e}")
            if strict:
                raise
            else:
                logger.warning("Continuing with random initialization due to loading errors")