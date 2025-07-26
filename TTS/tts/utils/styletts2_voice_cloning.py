"""
StyleTTS2 Voice Cloning Utilities

This module provides easy-to-use utilities for voice cloning with StyleTTS2.
It includes functions for processing reference audio, extracting speaker characteristics,
and performing voice cloning with various options.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Tuple

import torch
import torchaudio
import numpy as np
from pathlib import Path

from TTS.tts.configs.styletts2_config import StyleTTS2Config
from TTS.tts.models.styletts2 import Styletts2
from TTS.utils.audio import AudioProcessor

logger = logging.getLogger(__name__)


class StyleTTS2VoiceCloningUtils:
    """Utility class for StyleTTS2 voice cloning operations."""
    
    def __init__(self, model: Styletts2, config: StyleTTS2Config):
        """
        Initialize voice cloning utilities.
        
        Args:
            model: StyleTTS2 model instance
            config: StyleTTS2 configuration
        """
        self.model = model
        self.config = config
        self.voice_cloning_config = config.voice_cloning
        
    def preprocess_reference_audio(
        self, 
        audio_path: str,
        max_length: Optional[float] = None
    ) -> torch.Tensor:
        """
        Preprocess reference audio for voice cloning.
        
        Args:
            audio_path: Path to reference audio file
            max_length: Maximum length in seconds (uses config default if None)
            
        Returns:
            torch.Tensor: Preprocessed audio
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")
        
        max_length = max_length or self.voice_cloning_config["reference_audio_max_length"]
        
        # Load audio
        wav, sr = torchaudio.load(audio_path)
        
        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
            wav = resampler(wav)
        
        # Trim to max length
        max_samples = int(max_length * self.config.sample_rate)
        if wav.shape[1] > max_samples:
            wav = wav[:, :max_samples]
        
        # Normalize if enabled
        if self.voice_cloning_config["normalize_reference"]:
            wav = wav / wav.abs().max().clamp(min=1e-8)
        
        return wav.squeeze(0)
    
    def extract_speaker_embedding(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """
        Extract speaker embedding from reference audio.
        
        Args:
            audio_path: Path to reference audio file
            
        Returns:
            Dict containing acoustic and prosodic style embeddings
        """
        # Preprocess audio
        wav = self.preprocess_reference_audio(audio_path)
        
        # Extract style embeddings using the model
        acoustic_style, prosodic_style = self.model._extract_style_from_audio(wav)
        
        return {
            "acoustic_style": acoustic_style,
            "prosodic_style": prosodic_style,
            "audio_path": audio_path
        }
    
    def clone_voice_simple(
        self,
        text: str,
        reference_audio: str,
        alpha: float = None,
        output_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Simple voice cloning interface.
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio file
            alpha: Style interpolation factor (uses config default if None)
            output_path: Optional path to save the output audio
            
        Returns:
            torch.Tensor: Generated mel spectrogram
        """
        alpha = alpha or self.voice_cloning_config["style_interpolation_alpha"]
        diffusion_steps = self.voice_cloning_config["diffusion_steps"]
        
        logger.info(f"Cloning voice from: {reference_audio}")
        logger.info(f"Text: '{text}'")
        logger.info(f"Alpha: {alpha}, Diffusion steps: {diffusion_steps}")
        
        # Perform voice cloning
        mel_output = self.model.clone_voice(
            text=text,
            reference_wav=reference_audio,
            alpha=alpha,
            diffusion_steps=diffusion_steps
        )
        
        # Save output if path provided
        if output_path:
            self.save_mel_spectrogram(mel_output, output_path)
            logger.info(f"Output saved to: {output_path}")
        
        return mel_output
    
    def clone_voice_batch(
        self,
        texts: List[str],
        reference_audio: str,
        alpha: float = None,
        output_dir: Optional[str] = None
    ) -> List[torch.Tensor]:
        """
        Batch voice cloning for multiple texts.
        
        Args:
            texts: List of texts to synthesize
            reference_audio: Path to reference audio file
            alpha: Style interpolation factor
            output_dir: Optional directory to save outputs
            
        Returns:
            List of generated mel spectrograms
        """
        results = []
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}: '{text[:50]}...'")
            
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"cloned_voice_{i:03d}.wav")
            
            mel_output = self.clone_voice_simple(
                text=text,
                reference_audio=reference_audio,
                alpha=alpha,
                output_path=output_path
            )
            
            results.append(mel_output)
        
        return results
    
    def compare_voices(
        self,
        text: str,
        reference_audios: List[str],
        alphas: List[float] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compare voice cloning results from multiple reference audios.
        
        Args:
            text: Text to synthesize
            reference_audios: List of reference audio paths
            alphas: List of alpha values (one per reference)
            output_dir: Optional directory to save outputs
            
        Returns:
            Dict mapping reference audio names to mel spectrograms
        """
        if alphas is None:
            alphas = [self.voice_cloning_config["style_interpolation_alpha"]] * len(reference_audios)
        
        if len(alphas) != len(reference_audios):
            raise ValueError("Number of alphas must match number of reference audios")
        
        results = {}
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for i, (ref_audio, alpha) in enumerate(zip(reference_audios, alphas)):
            ref_name = Path(ref_audio).stem
            logger.info(f"Cloning with reference: {ref_name}")
            
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"comparison_{ref_name}.wav")
            
            mel_output = self.clone_voice_simple(
                text=text,
                reference_audio=ref_audio,
                alpha=alpha,
                output_path=output_path
            )
            
            results[ref_name] = mel_output
        
        return results
    
    def save_mel_spectrogram(self, mel: torch.Tensor, output_path: str):
        """
        Save mel spectrogram as audio file.
        
        Args:
            mel: Mel spectrogram tensor
            output_path: Output file path
        """
        # This would require a vocoder to convert mel to audio
        # For now, save as tensor
        torch.save(mel.cpu(), output_path.replace('.wav', '.pt'))
        logger.info(f"Mel spectrogram saved as tensor: {output_path.replace('.wav', '.pt')}")
    
    def get_supported_audio_formats(self) -> List[str]:
        """Get list of supported audio formats."""
        return ['.wav', '.flac', '.mp3', '.m4a', '.ogg']
    
    def validate_reference_audio(self, audio_path: str) -> Dict[str, Union[bool, str]]:
        """
        Validate reference audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with validation results
        """
        result = {
            "valid": False,
            "message": "",
            "duration": 0.0,
            "sample_rate": 0,
            "channels": 0
        }
        
        try:
            if not os.path.exists(audio_path):
                result["message"] = "File does not exist"
                return result
            
            # Check file extension
            ext = Path(audio_path).suffix.lower()
            if ext not in self.get_supported_audio_formats():
                result["message"] = f"Unsupported format: {ext}"
                return result
            
            # Load and analyze audio
            wav, sr = torchaudio.load(audio_path)
            duration = wav.shape[1] / sr
            
            result.update({
                "valid": True,
                "message": "Valid reference audio",
                "duration": duration,
                "sample_rate": sr,
                "channels": wav.shape[0]
            })
            
            # Add warnings for potential issues
            warnings = []
            if duration < 2.0:
                warnings.append("Audio is quite short (< 2s)")
            if duration > 30.0:
                warnings.append("Audio is very long (> 30s)")
            if sr != self.config.sample_rate:
                warnings.append(f"Will be resampled from {sr}Hz to {self.config.sample_rate}Hz")
            
            if warnings:
                result["message"] += f" (Warnings: {', '.join(warnings)})"
            
        except Exception as e:
            result["message"] = f"Error loading audio: {str(e)}"
        
        return result


def create_voice_cloning_utils(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None
) -> StyleTTS2VoiceCloningUtils:
    """
    Convenience function to create voice cloning utilities.
    
    Args:
        model_path: Path to trained StyleTTS2 model (optional)
        config_path: Path to config file (optional)
        
    Returns:
        StyleTTS2VoiceCloningUtils instance
    """
    # Initialize config
    if config_path:
        # Load config from file (implementation would depend on format)
        config = StyleTTS2Config()  # Placeholder
    else:
        config = StyleTTS2Config()
    
    # Initialize model
    model = Styletts2.init_from_config(config, [])
    
    # Load checkpoint if provided
    if model_path:
        model.load_checkpoint(config, model_path, eval=True)
    
    return StyleTTS2VoiceCloningUtils(model, config)


def quick_voice_clone(
    text: str,
    reference_audio: str,
    alpha: float = 0.3,
    diffusion_steps: int = 10
) -> torch.Tensor:
    """
    Quick voice cloning function for simple use cases.
    
    Args:
        text: Text to synthesize
        reference_audio: Path to reference audio
        alpha: Style interpolation factor
        diffusion_steps: Number of diffusion steps
        
    Returns:
        torch.Tensor: Generated mel spectrogram
    """
    utils = create_voice_cloning_utils()
    return utils.clone_voice_simple(text, reference_audio, alpha)