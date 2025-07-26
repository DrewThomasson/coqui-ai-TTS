import os
import logging
from typing import Dict, List, Tuple, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
import torchaudio
import numpy as np

from TTS.tts.layers.styletts2.models import (
    StyleEncoder, 
    TextEncoder, 
    SimpleDiffusionModel, 
    SimpleHiFiGANDecoder,
    LinearNorm
)
from TTS.tts.layers.styletts2.losses import MultiResolutionSTFTLoss
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

logger = logging.getLogger(__name__)


class Styletts2(BaseTTS):
    """StyleTTS2 Text-to-Speech model implementation for Coqui TTS."""

    def __init__(
        self,
        config: "Coqpit",
        ap: AudioProcessor = None,
        tokenizer: TTSTokenizer = None,
        speaker_manager=None,
        language_manager=None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager, language_manager)
        
        # Store config params
        self.hidden_dim = config.hidden_dim
        self.style_dim = config.style_dim
        self.n_layer = config.n_layer
        self.n_token = config.n_token
        self.max_dur = config.max_dur
        self.dropout = config.dropout
        self.multispeaker = config.multispeaker
        
        # Initialize model components
        self._build_model()
        
        # Initialize losses
        self.mel_loss = nn.L1Loss()  # Simple L1 loss for mel spectrograms
        self.l1_loss = nn.L1Loss()
        self.mse_loss = MSELoss()
        
    def _build_model(self):
        """Build StyleTTS2 model components."""
        
        # Text encoder
        self.text_encoder = TextEncoder(
            channels=self.hidden_dim,
            kernel_size=5,
            depth=self.n_layer,
            n_symbols=self.n_token
        )
        
        # Style encoders
        self.style_encoder = StyleEncoder(
            dim_in=self.config.n_mels,  # Use n_mels instead of dim_in
            style_dim=self.style_dim,
            max_conv_dim=self.hidden_dim
        )
        
        # Predictor encoder (for prosodic style)
        self.predictor_encoder = StyleEncoder(
            dim_in=self.config.n_mels,  # Use n_mels instead of dim_in
            style_dim=self.style_dim,
            max_conv_dim=self.hidden_dim
        )
        
        # Diffusion model for style generation
        self.diffusion = SimpleDiffusionModel(
            style_dim=self.style_dim,
            context_dim=self.hidden_dim
        )
        
        # Decoder
        self.decoder = SimpleHiFiGANDecoder(
            dim_in=self.hidden_dim,
            style_dim=self.style_dim,
            dim_out=self.config.n_mels
        )
        
        # Duration predictor
        self.duration_predictor = nn.Sequential(
            LinearNorm(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, 1)
        )
        
        if self.multispeaker and self.speaker_manager:
            self.speaker_embedding = nn.Embedding(
                self.speaker_manager.num_speakers, self.style_dim
            )

    def _load_reference_audio(self, reference_wav: str) -> torch.Tensor:
        """Load and preprocess reference audio for voice cloning."""
        if not os.path.exists(reference_wav):
            raise FileNotFoundError(f"Reference audio file not found: {reference_wav}")
        
        # Load audio using torchaudio
        wav, sr = torchaudio.load(reference_wav)
        
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            wav = resampler(wav)
        
        # Normalize audio
        wav = wav / wav.abs().max()
        
        return wav.squeeze(0)  # Remove channel dimension

    def _extract_style_from_audio(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract style embeddings from reference audio."""
        # Convert audio to mel spectrogram using AudioProcessor
        if self.ap is None:
            raise ValueError("AudioProcessor not initialized. Cannot extract mel spectrogram.")
        
        # Convert to numpy for AudioProcessor
        wav_np = wav.cpu().numpy()
        
        # Get mel spectrogram
        mel = self.ap.melspectrogram(wav_np)
        mel = torch.FloatTensor(mel).unsqueeze(0)  # Add batch dimension
        
        if torch.cuda.is_available() and next(self.parameters()).is_cuda:
            mel = mel.cuda()
        
        # Extract style embeddings using the style encoders
        with torch.no_grad():
            # Add channel dimension for conv layers
            mel_2d = mel.unsqueeze(1)  # [B, 1, n_mels, T]
            
            # Extract acoustic and prosodic styles
            acoustic_style = self.style_encoder(mel_2d)
            prosodic_style = self.predictor_encoder(mel_2d)
        
        return acoustic_style, prosodic_style

    def clone_voice(
        self,
        text: str,
        reference_wav: str,
        alpha: float = 0.3,
        diffusion_steps: int = 10,
        **kwargs
    ) -> torch.Tensor:
        """
        Clone a voice from reference audio and synthesize the given text.
        
        Args:
            text (str): Text to synthesize
            reference_wav (str): Path to reference audio file
            alpha (float): Style interpolation factor (0=original, 1=reference style)
            diffusion_steps (int): Number of diffusion steps for style generation
            
        Returns:
            torch.Tensor: Generated mel spectrogram
        """
        logger.info(f"Cloning voice from: {reference_wav}")
        
        # Load and process reference audio
        ref_wav = self._load_reference_audio(reference_wav)
        
        # Extract style from reference audio
        ref_acoustic_style, ref_prosodic_style = self._extract_style_from_audio(ref_wav)
        
        # Tokenize text
        token_ids = self.tokenizer.text_to_ids(text)
        token_ids = torch.LongTensor(token_ids).unsqueeze(0)
        text_lengths = torch.LongTensor([len(token_ids[0])])
        
        if torch.cuda.is_available() and next(self.parameters()).is_cuda:
            token_ids = token_ids.cuda()
            text_lengths = text_lengths.cuda()
        
        # Run inference with reference style
        with torch.no_grad():
            outputs = self._inference_with_style(
                token_ids,
                text_lengths,
                ref_acoustic_style=ref_acoustic_style,
                ref_prosodic_style=ref_prosodic_style,
                alpha=alpha,
                diffusion_steps=diffusion_steps
            )
        
        mel_pred = outputs["model_outputs"]
        logger.info(f"Voice cloning completed. Output shape: {mel_pred.shape}")
        
        return mel_pred

    def _inference_with_style(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        ref_acoustic_style: torch.Tensor = None,
        ref_prosodic_style: torch.Tensor = None,
        alpha: float = 0.3,
        diffusion_steps: int = 10,
        speaker_ids: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Internal inference method with style control."""
        
        # Create text mask
        text_mask = self.text_encoder.length_to_mask(x_lengths).to(x.device)
        
        # Text encoding 
        text_encoded = self.text_encoder(x, x_lengths, text_mask)
        
        # Generate or use provided styles
        batch_size = x.size(0)
        
        if ref_acoustic_style is not None and ref_prosodic_style is not None:
            # Use reference styles
            acoustic_style = ref_acoustic_style
            prosodic_style = ref_prosodic_style
            
            # Optionally interpolate with random style for variation
            if alpha < 1.0:
                random_acoustic = torch.randn_like(acoustic_style)
                random_prosodic = torch.randn_like(prosodic_style)
                
                acoustic_style = alpha * acoustic_style + (1 - alpha) * random_acoustic
                prosodic_style = alpha * prosodic_style + (1 - alpha) * random_prosodic
        else:
            # Generate random styles for inference
            acoustic_style = torch.randn(batch_size, self.style_dim).to(x.device)
            prosodic_style = torch.randn(batch_size, self.style_dim).to(x.device)
        
        # Add speaker embedding if multispeaker
        if self.multispeaker and speaker_ids is not None:
            speaker_emb = self.speaker_embedding(speaker_ids)
            acoustic_style = acoustic_style + speaker_emb
            prosodic_style = prosodic_style + speaker_emb
        
        # Duration prediction
        duration_pred = self.duration_predictor(text_encoded.transpose(-1, -2))
        duration_pred = F.softplus(duration_pred).squeeze(-1)
        
        # Simple duration alignment
        aligned_text = text_encoded
        
        # Apply diffusion model with enhanced steps
        combined_style = torch.cat([acoustic_style, prosodic_style], dim=-1)
        
        # Enhanced diffusion process for better voice cloning
        for step in range(diffusion_steps):
            timestep = torch.full((batch_size,), step / diffusion_steps).to(x.device)
            combined_style = self.diffusion(
                combined_style,
                timesteps=timestep,
                context=aligned_text.mean(dim=-1)
            )
        
        # Split back to acoustic style for decoder
        style_for_decoder = combined_style[:, :self.style_dim]
        
        # Decode to mel spectrogram
        mel_pred = self.decoder(aligned_text, style_for_decoder)
        
        outputs = {
            "model_outputs": mel_pred,
            "durations_log": duration_pred,
            "alignments": None,
            "text_hidden": text_encoded,
            "style": style_for_decoder,
            "acoustic_style": acoustic_style,
            "prosodic_style": prosodic_style,
        }
        
        return outputs

    @staticmethod  
    def init_from_config(
        config: "Coqpit", samples: Union[List[List], List[Dict]] = None, verbose: bool = True
    ):
        """Initialize StyleTTS2 from config"""
        from TTS.utils.audio import AudioProcessor
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.tts.utils.speakers import SpeakerManager
        from TTS.tts.utils.languages import LanguageManager

        ap = AudioProcessor.init_from_config(config)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        language_manager = LanguageManager.init_from_config(config)
        
        return Styletts2(new_config, ap, tokenizer, speaker_manager, language_manager)

    def forward(
        self, 
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor = None,
        y_lengths: torch.Tensor = None,
        speaker_ids: torch.Tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of StyleTTS2."""
        
        # Create text mask
        text_mask = self.text_encoder.length_to_mask(x_lengths).to(x.device)
        
        # Text encoding 
        text_encoded = self.text_encoder(x, x_lengths, text_mask)
        
        # Style encoding from mel spectrogram if available (training)
        if y is not None:
            # Convert mel to appropriate format for style encoder
            y_2d = y.unsqueeze(1)  # Add channel dimension
            acoustic_style = self.style_encoder(y_2d)
            prosodic_style = self.predictor_encoder(y_2d)
        else:
            # Generate random style for inference
            batch_size = x.size(0)
            acoustic_style = torch.randn(batch_size, self.style_dim).to(x.device)
            prosodic_style = torch.randn(batch_size, self.style_dim).to(x.device)
        
        # Add speaker embedding if multispeaker
        if self.multispeaker and speaker_ids is not None:
            speaker_emb = self.speaker_embedding(speaker_ids)
            acoustic_style = acoustic_style + speaker_emb
            prosodic_style = prosodic_style + speaker_emb
        
        # Duration prediction
        duration_pred = self.duration_predictor(text_encoded.transpose(-1, -2))
        duration_pred = F.softplus(duration_pred).squeeze(-1)
        
        # Simple duration alignment (in full implementation this would use alignment)
        if y is not None:
            # Use target lengths for training
            aligned_text = text_encoded
        else:
            # Use predicted durations for inference 
            aligned_text = text_encoded
        
        # Apply diffusion model to generate style
        diffused_style = self.diffusion(
            torch.cat([acoustic_style, prosodic_style], dim=-1),
            timesteps=torch.zeros(acoustic_style.size(0)).to(acoustic_style.device),
            context=aligned_text.mean(dim=-1)  # Simple context
        )
        
        # Decode to mel spectrogram
        # Split diffused style back to original style size
        style_for_decoder = diffused_style[:, :self.style_dim]  # Use first half
        mel_pred = self.decoder(aligned_text, style_for_decoder)
        
        outputs = {
            "model_outputs": mel_pred,
            "durations_log": duration_pred,
            "alignments": None,  # Placeholder
            "text_hidden": text_encoded,
            "style": style_for_decoder,
            "acoustic_style": acoustic_style,
            "prosodic_style": prosodic_style,
        }
        
        return outputs
    
    def compute_loss(
        self, 
        batch: Dict, 
        criterion: nn.Module, 
        model_output: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute StyleTTS2 training losses."""
        
        # Get targets
        mel_target = batch["mel"]
        mel_pred = model_output["model_outputs"]
        
        # Handle dimension mismatch by cropping to smaller size
        min_length = min(mel_pred.size(-1), mel_target.size(-1))
        mel_pred = mel_pred[..., :min_length]
        mel_target = mel_target[..., :min_length]
        
        # Mel reconstruction loss
        mel_loss = self.mel_loss(mel_pred, mel_target)
        
        # Duration loss (simplified)
        duration_loss = 0.0
        if "durations_log" in model_output and "durations" in batch:
            duration_pred = model_output["durations_log"]
            duration_target = batch["durations"].float()
            # Handle length mismatch for duration too
            min_dur_length = min(duration_pred.size(-1), duration_target.size(-1))
            duration_pred = duration_pred[..., :min_dur_length]
            duration_target = duration_target[..., :min_dur_length]
            duration_loss = self.mse_loss(duration_pred, duration_target)
        
        # Total loss
        total_loss = (
            self.config.lambda_mel * mel_loss + 
            self.config.lambda_dur * duration_loss
        )
        
        loss_dict = {
            "loss": total_loss,
            "loss_mel": mel_loss,
            "loss_duration": duration_loss,
        }
        
        return total_loss, loss_dict

    def inference(
        self,
        text: str,
        speaker_id: int = None,
        style_wav: str = None,
        reference_wav: str = None,
        alpha: float = 0.3,
        diffusion_steps: int = 10,
        **kwargs
    ) -> torch.Tensor:
        """
        Run StyleTTS2 inference with optional voice cloning.
        
        Args:
            text (str): Text to synthesize
            speaker_id (int, optional): Speaker ID for multi-speaker models
            style_wav (str, optional): Path to reference audio for style (legacy parameter)
            reference_wav (str, optional): Path to reference audio for voice cloning
            alpha (float): Style interpolation factor for voice cloning
            diffusion_steps (int): Number of diffusion steps
            
        Returns:
            torch.Tensor: Generated mel spectrogram
        """
        
        # Support both style_wav and reference_wav for compatibility
        ref_wav_path = reference_wav or style_wav
        
        # If reference audio is provided, use voice cloning
        if ref_wav_path:
            return self.clone_voice(
                text=text,
                reference_wav=ref_wav_path,
                alpha=alpha,
                diffusion_steps=diffusion_steps,
                **kwargs
            )
        
        # Standard inference without voice cloning
        # Tokenize text
        token_ids = self.tokenizer.text_to_ids(text)
        token_ids = torch.LongTensor(token_ids).unsqueeze(0)
        text_lengths = torch.LongTensor([len(token_ids[0])])
        
        if torch.cuda.is_available() and next(self.parameters()).is_cuda:
            token_ids = token_ids.cuda()
            text_lengths = text_lengths.cuda()
        
        speaker_ids = None
        if speaker_id is not None:
            speaker_ids = torch.LongTensor([speaker_id])
            if torch.cuda.is_available() and next(self.parameters()).is_cuda:
                speaker_ids = speaker_ids.cuda()
        
        # Run forward pass
        with torch.no_grad():
            outputs = self._inference_with_style(
                token_ids,
                text_lengths,
                speaker_ids=speaker_ids,
                diffusion_steps=diffusion_steps
            )
        
        mel_pred = outputs["model_outputs"]
        
        return mel_pred
    
    def test_run(self, assets) -> Tuple[Dict, Dict]:
        """Test run for model validation."""
        
        print("Running StyleTTS2 test...")
        
        # Simple test with dummy data
        batch_size = 2
        seq_len = 20
        mel_len = 100
        
        # Create dummy input
        text_input = torch.randint(0, self.n_token, (batch_size, seq_len))
        text_lengths = torch.LongTensor([seq_len] * batch_size)
        mel_target = torch.randn(batch_size, self.config.n_mels, mel_len)
        mel_lengths = torch.LongTensor([mel_len] * batch_size)
        
        if torch.cuda.is_available():
            text_input = text_input.cuda()
            text_lengths = text_lengths.cuda() 
            mel_target = mel_target.cuda()
            mel_lengths = mel_lengths.cuda()
        
        # Forward pass
        model_output = self.forward(
            text_input,
            text_lengths,
            mel_target,
            mel_lengths
        )
        
        # Dummy batch for loss computation
        batch = {
            "mel": mel_target,
            "durations": torch.ones(batch_size, seq_len)
        }
        
        if torch.cuda.is_available():
            batch["durations"] = batch["durations"].cuda()
        
        # Compute loss
        loss, loss_dict = self.compute_loss(batch, None, model_output)
        
        print(f"StyleTTS2 test completed. Loss: {loss.item():.4f}")
        
        return model_output, {"loss": loss.item()}

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        """StyleTTS2 training step."""
        
        # Get batch data
        text_input = batch["token_ids"]
        text_lengths = batch["token_id_lengths"]
        mel_target = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        speaker_ids = batch.get("speaker_ids", None)
        
        # Forward pass
        outputs = self.forward(
            text_input,
            text_lengths, 
            mel_target,
            mel_lengths,
            speaker_ids=speaker_ids
        )
        
        # Compute loss
        loss, loss_dict = self.compute_loss(batch, criterion, outputs)
        
        return outputs, loss_dict

    def eval_step(self, batch: dict, criterion: nn.Module) -> Tuple[Dict, Dict]:
        """StyleTTS2 evaluation step."""
        return self.train_step(batch, criterion, 0)

    def get_data_loader(
        self,
        config: "Coqpit",
        assets: dict,
        is_eval: bool,
        samples: Union[list, dict],
        verbose: bool = False,
        num_gpus: int = 1,
        rank: int = 0,
    ):
        """Get data loader for StyleTTS2."""
        
        # Use the default TTS data loader for now
        # In a full implementation, this would be customized for StyleTTS2's specific needs
        from TTS.tts.datasets.dataset import TTSDataset
        from torch.utils.data import DataLoader
        
        dataset = TTSDataset(
            outputs_per_step=config.r if hasattr(config, 'r') else 1,
            text_cleaner=config.text_cleaner,
            compute_linear_spec=False,
            ap=self.ap,
            samples=samples,
            tokenizer=self.tokenizer,
            add_blank=config.add_blank if hasattr(config, 'add_blank') else False,
            return_wav=False,
            batch_group_size=0,
            min_seq_len=config.min_seq_len if hasattr(config, 'min_seq_len') else 1,
            max_seq_len=config.max_seq_len if hasattr(config, 'max_seq_len') else float("inf"),
            phoneme_cache_path=config.phoneme_cache_path if hasattr(config, 'phoneme_cache_path') else None,
            precompute_num_workers=config.precompute_num_workers if hasattr(config, 'precompute_num_workers') else 0,
            speaker_id_mapping=self.speaker_manager.name_to_id if self.speaker_manager else None,
            d_vector_mapping=self.speaker_manager.embeddings if self.speaker_manager else None,
            language_id_mapping=self.language_manager.name_to_id if self.language_manager else None,
            use_noise_augment=False,
        )
        
        sampler = None
        shuffle = not is_eval
        if num_gpus > 1:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False
        
        loader = DataLoader(
            dataset,
            batch_size=config.eval_batch_size if is_eval else config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
            pin_memory=True,
        )
        
        return loader

    def load_checkpoint(
        self,
        config: "Coqpit",
        checkpoint_path: str,
        eval: bool = False,
        strict: bool = True,
        cache: bool = False,
    ) -> None:
        """Load a model checkpoint file and get ready for training or inference."""
        
        state = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in state:
            model_state = state['model']
        elif 'state_dict' in state:
            model_state = state['state_dict']
        elif 'net' in state:
            # StyleTTS2 format
            model_state = state['net']
        else:
            model_state = state
        
        # Load model weights
        if isinstance(model_state, dict):
            # If model_state contains sub-modules, load them separately
            missing_keys = []
            unexpected_keys = []
            
            for name, module in self.named_children():
                if name in model_state:
                    try:
                        module.load_state_dict(model_state[name], strict=strict)
                        print(f"Loaded {name}")
                    except Exception as e:
                        print(f"Failed to load {name}: {e}")
                        if strict:
                            raise
                else:
                    missing_keys.append(name)
            
            if missing_keys and strict:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
                
        else:
            # Standard pytorch checkpoint
            self.load_state_dict(model_state, strict=strict)
        
        if eval:
            self.eval()
        else:
            self.train()
            
        print(f"Model loaded from {checkpoint_path}")

    def on_init_end(self, trainer):
        """Called at the end of initialization."""
        # Print model info
        if trainer.rank == 0:
            print(f"StyleTTS2 model initialized with {sum(p.numel() for p in self.parameters())} parameters")