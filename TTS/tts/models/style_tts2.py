"""StyleTTS2 model integration for Coqui TTS.

Supports both single-speaker (LJSpeech) and multi-speaker (LibriTTS) inference.
Based on https://github.com/yl4579/StyleTTS2
"""

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torchaudio
from coqpit import Coqpit
from munch import Munch

from TTS.tts.configs.style_tts2_config import StyleTTS2Config
from TTS.tts.layers.styletts2.diffusion.sampler import (
    ADPM2Sampler,
    DiffusionSampler,
    KarrasSchedule,
)
from TTS.tts.layers.styletts2.models import (
    build_model,
    load_ASR_models,
    load_F0_models,
)
from TTS.tts.layers.styletts2.text_utils import TextCleaner
from TTS.tts.models.base_tts import BaseTTS
from TTS.utils.generic_utils import is_pytorch_at_least_2_4

logger = logging.getLogger(__name__)


def _recursive_munch(d):
    """Convert nested dicts to Munch objects for attribute access."""
    if isinstance(d, dict):
        return Munch((k, _recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [_recursive_munch(v) for v in d]
    return d


def _length_to_mask(lengths):
    """Create a boolean mask from sequence lengths."""
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


class StyleTTS2(BaseTTS):
    """StyleTTS2 model for Coqui TTS.

    This is an inference-only wrapper around the StyleTTS2 architecture.
    It supports:
    - Single-speaker synthesis (LJSpeech model, iSTFTNet decoder)
    - Multi-speaker / zero-shot synthesis (LibriTTS model, HiFi-GAN decoder)

    For multi-speaker mode, provide ``speaker_wav`` to control the speaker/style.
    """

    config: StyleTTS2Config

    def __init__(self, config: Coqpit) -> None:
        super().__init__(config=config, ap=None, tokenizer=None, speaker_manager=None, language_manager=None)
        self.config = config
        self.text_cleaner = TextCleaner()
        self.phonemizer = None
        self.model = None
        self.sampler = None
        self.model_params = None

        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        self.mel_mean = -4
        self.mel_std = 4

    def _init_phonemizer(self):
        """Lazily initialize the espeak phonemizer."""
        if self.phonemizer is None:
            import phonemizer

            lang = getattr(self.config, "phoneme_language", "en-us") or "en-us"
            self.phonemizer = phonemizer.backend.EspeakBackend(
                language=lang, preserve_punctuation=True, with_stress=True
            )

    def _preprocess_audio(self, wave):
        """Convert audio waveform to normalized log-mel spectrogram."""
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mel_mean) / self.mel_std
        return mel_tensor

    def _compute_style(self, path):
        """Compute style embedding from a reference audio file.

        For multi-speaker mode, returns concatenated acoustic + prosodic style.
        For single-speaker mode, returns just the acoustic style.
        """
        wave, sr = librosa.load(path, sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        mel_tensor = self._preprocess_audio(audio).to(self.device)

        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            if self.config.multispeaker:
                ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))
                return torch.cat([ref_s, ref_p], dim=1)
            return ref_s.squeeze(1)

    def _phonemize(self, text):
        """Convert text to phoneme token IDs."""
        from nltk.tokenize import word_tokenize

        self._init_phonemizer()
        text = text.strip().replace('"', '')
        ps = self.phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        tokens = self.text_cleaner(ps)
        tokens.insert(0, 0)
        return tokens

    @torch.inference_mode()
    def _inference_single_speaker(self, text, diffusion_steps=5, embedding_scale=1.0):
        """Run inference for single-speaker (LJSpeech) model."""
        device = self.device
        tokens = self._phonemize(text)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = _length_to_mask(input_lengths).to(device)

        t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        noise = torch.randn(1, 1, 256).to(device)
        s_pred = self.sampler(
            noise,
            embedding=bert_dur[0].unsqueeze(0),
            num_steps=diffusion_steps,
            embedding_scale=embedding_scale,
        ).squeeze(0)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = self.model.predictor.lstm(d)
        duration = self.model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_dur[-1] += 5

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
        F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
        out = self.model.decoder(
            t_en @ pred_aln_trg.unsqueeze(0).to(device),
            F0_pred, N_pred, ref.squeeze().unsqueeze(0),
        )

        return out.squeeze().cpu().numpy()

    @torch.inference_mode()
    def _inference_multi_speaker(self, text, ref_s, diffusion_steps=5, embedding_scale=1.0,
                                  alpha=0.3, beta=0.7):
        """Run inference for multi-speaker (LibriTTS) model with reference audio style."""
        device = self.device
        tokens = self._phonemize(text)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = _length_to_mask(input_lengths).to(device)

        t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

        noise = torch.randn(1, 256).unsqueeze(1).to(device)
        s_pred = self.sampler(
            noise,
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

        x, _ = self.model.predictor.lstm(d)
        duration = self.model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
        is_hifigan = self.model_params.decoder.type == "hifigan"
        if is_hifigan:
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

        asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
        if is_hifigan:
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        # Trim trailing artifact
        return out.squeeze().cpu().numpy()[..., :-50]

    @property
    def device(self):
        """Return the device of the model parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def synthesize(
        self,
        text: str,
        config=None,
        *,
        speaker=None,
        speaker_wav=None,
        voice_dir=None,
        language=None,
        **kwargs,
    ) -> dict[str, Any]:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize.
            config: Unused (kept for API compatibility).
            speaker: Unused.
            speaker_wav: Path to reference audio for multi-speaker mode.
            voice_dir: Unused.
            language: Unused.
            **kwargs: Additional arguments:
                - diffusion_steps (int): Number of diffusion steps.
                - embedding_scale (float): Guidance scale.
                - alpha (float): Reference blending for acoustic style (multi-speaker).
                - beta (float): Reference blending for prosodic style (multi-speaker).

        Returns:
            Dictionary with ``"wav"`` key containing the audio waveform as numpy array.
        """
        diffusion_steps = kwargs.get("diffusion_steps", self.config.diffusion_steps)
        embedding_scale = kwargs.get("embedding_scale", self.config.embedding_scale)

        if self.config.multispeaker and speaker_wav is not None:
            if isinstance(speaker_wav, (list, tuple)):
                speaker_wav = speaker_wav[0]
            ref_s = self._compute_style(str(speaker_wav))
            alpha = kwargs.get("alpha", self.config.alpha)
            beta = kwargs.get("beta", self.config.beta)
            wav = self._inference_multi_speaker(
                text, ref_s, diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale, alpha=alpha, beta=beta,
            )
        else:
            wav = self._inference_single_speaker(
                text, diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
            )

        return {"wav": wav, "text_inputs": text}

    def forward(self):
        ...

    def inference(self):
        ...

    def train_step(self):
        raise NotImplementedError("StyleTTS2 training is not implemented")

    @staticmethod
    def init_from_config(config: StyleTTS2Config, **kwargs):
        return StyleTTS2(config)

    @staticmethod
    def _download_utils(target_dir: Path) -> None:
        """Download StyleTTS2 utility models (ASR, JDC, PLBERT) if not present.

        Downloads from the official StyleTTS2 GitHub repository.
        """
        from huggingface_hub import hf_hub_download

        repo_id = "yl4579/StyleTTS2-LJSpeech"  # Utils are the same for both models

        utils_files = {
            "Utils/ASR/config.yml": target_dir / "Utils" / "ASR" / "config.yml",
            "Utils/ASR/epoch_00080.pth": target_dir / "Utils" / "ASR" / "epoch_00080.pth",
            "Utils/JDC/bst.t7": target_dir / "Utils" / "JDC" / "bst.t7",
            "Utils/PLBERT/config.yml": target_dir / "Utils" / "PLBERT" / "config.yml",
            "Utils/PLBERT/step_1000000.t7": target_dir / "Utils" / "PLBERT" / "step_1000000.t7",
        }

        # Download from a GitHub raw URL since the HF repo doesn't have utils
        base_url = "https://raw.githubusercontent.com/yl4579/StyleTTS2/main/"
        for remote_path, local_path in utils_files.items():
            if local_path.exists():
                continue
            local_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s ...", remote_path)
            try:
                import urllib.request

                urllib.request.urlretrieve(base_url + remote_path, str(local_path))
            except Exception:
                logger.warning("Failed to download %s from GitHub, trying LFS on HF...", remote_path)
                try:
                    downloaded = hf_hub_download(
                        repo_id=repo_id,
                        filename=remote_path,
                    )
                    import shutil

                    shutil.copy2(downloaded, str(local_path))
                except Exception:
                    logger.error("Could not download %s", remote_path)
                    raise

    def load_checkpoint(
        self,
        config,
        checkpoint_dir,
        eval=False,
        strict=True,
        **kwargs,
    ):
        """Load model checkpoints from a directory.

        The checkpoint directory should contain:
        - A StyleTTS2 config file (``config.yml`` or ``config.json``)
        - A ``.pth`` checkpoint file (e.g. ``epoch_2nd_00100.pth``)

        Utility models (ASR, JDC, PLBERT) are automatically downloaded from the
        StyleTTS2 GitHub repository if not present in the checkpoint directory.

        Args:
            config: Model configuration.
            checkpoint_dir: Path to the checkpoint directory.
            eval: Whether to set the model to eval mode.
            strict: Whether to strictly enforce state dict matching.
        """
        import yaml

        from TTS.tts.layers.styletts2.Utils.PLBERT.util import load_plbert

        checkpoint_dir = Path(checkpoint_dir)
        logger.info("Loading StyleTTS2 from %s", checkpoint_dir)

        # Find StyleTTS2 YAML config - may be nested (e.g. Models/LJSpeech/config.yml)
        config_path = None
        for candidate in [
            checkpoint_dir / "config.yml",
            checkpoint_dir / "config.json",
        ]:
            if candidate.exists():
                config_path = candidate
                break
        if config_path is None:
            # Search in subdirectories (HF snapshot layout)
            for yml in checkpoint_dir.rglob("config.yml"):
                config_path = yml
                break
        if config_path is None:
            msg = f"No config.yml found in {checkpoint_dir}"
            raise FileNotFoundError(msg)

        with open(config_path) as f:
            styletts2_config = yaml.safe_load(f)

        model_params = styletts2_config.get("model_params", self.config.model_params)
        self.model_params = _recursive_munch(model_params)
        self.config.multispeaker = model_params.get("multispeaker", False)
        self.config._supports_cloning = self.config.multispeaker

        # Download utility models if not present
        self._download_utils(checkpoint_dir)

        # Load ASR model
        asr_config_path = checkpoint_dir / "Utils" / "ASR" / "config.yml"
        asr_model_path = checkpoint_dir / "Utils" / "ASR" / "epoch_00080.pth"
        text_aligner = load_ASR_models(str(asr_model_path), str(asr_config_path))

        # Load F0 model
        f0_path = checkpoint_dir / "Utils" / "JDC" / "bst.t7"
        pitch_extractor = load_F0_models(str(f0_path))

        # Load PLBERT
        plbert_dir = checkpoint_dir / "Utils" / "PLBERT"
        plbert = load_plbert(str(plbert_dir))

        # Build model
        model = build_model(
            self.model_params, text_aligner, pitch_extractor, plbert
        )

        # Find and load the checkpoint file
        pth_files = sorted(checkpoint_dir.glob("**/*.pth"))
        # Filter to only main model checkpoint files
        pth_files = [
            f for f in pth_files
            if "Utils" not in str(f)
            and "ASR" not in f.name
            and f.name.startswith("epoch")
        ]
        if not pth_files:
            # Fallback: find any .pth that isn't a utility
            pth_files = [
                f for f in sorted(checkpoint_dir.glob("**/*.pth"))
                if "Utils" not in str(f) and "ASR" not in f.name
            ]
        if not pth_files:
            msg = f"No .pth checkpoint file found in {checkpoint_dir}"
            raise FileNotFoundError(msg)

        ckpt_path = pth_files[-1]
        logger.info("Loading checkpoint: %s", ckpt_path)
        weights_only = is_pytorch_at_least_2_4()
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=weights_only)
        params = state["net"]

        for key in model:
            if key in params:
                try:
                    model[key].load_state_dict(params[key])
                except RuntimeError:
                    # Handle DataParallel prefix
                    new_state_dict = OrderedDict()
                    for k, v in params[key].items():
                        name = k[7:] if k.startswith("module.") else k
                        new_state_dict[name] = v
                    model[key].load_state_dict(new_state_dict, strict=False)
                logger.info("Loaded %s", key)

        self.model = model

        # Build diffusion sampler
        self.sampler = DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

        # Register sub-modules as parameters so .to(device) works
        for key in model:
            if isinstance(model[key], torch.nn.Module):
                self.add_module(f"styletts2_{key}", model[key])

        if eval:
            for key in model:
                model[key].eval()
            self.eval()
