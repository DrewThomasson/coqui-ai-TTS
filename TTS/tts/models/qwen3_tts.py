"""Qwen3-TTS model integration for Coqui TTS.

Qwen3-TTS is a multilingual text-to-speech model supporting 10 languages
with voice cloning, voice design, and custom voice generation capabilities.

Requires the ``qwen-tts`` package: ``pip install qwen-tts``

Paper: https://arxiv.org/abs/2601.15621
Repository: https://github.com/QwenLM/Qwen3-TTS
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from coqpit import Coqpit

from TTS.tts.configs.qwen3_tts_config import Qwen3TTSConfig
from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.languages import LanguageManager
from TTS.utils.generic_utils import warn_synthesize_config_deprecated, warn_synthesize_speaker_id_deprecated

logger = logging.getLogger(__name__)


def _import_qwen_tts():
    """Lazy import for the qwen-tts package."""
    try:
        from qwen_tts import Qwen3TTSModel

        return Qwen3TTSModel
    except ImportError:
        msg = (
            "Qwen3-TTS requires the `qwen-tts` package. "
            "Install it with: pip install qwen-tts"
        )
        raise ImportError(msg)


# Map coqui-style language codes to Qwen3-TTS language names
_LANGUAGE_MAP = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
    # Also accept the full names directly
    "chinese": "Chinese",
    "english": "English",
    "japanese": "Japanese",
    "korean": "Korean",
    "german": "German",
    "french": "French",
    "russian": "Russian",
    "portuguese": "Portuguese",
    "spanish": "Spanish",
    "italian": "Italian",
}

# Built-in speakers for the CustomVoice variant
_CUSTOM_VOICE_SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]


def _resolve_language(language: str | None) -> str:
    """Resolve a language code or name to a Qwen3-TTS language name."""
    if language is None:
        return "Auto"
    lang_lower = language.lower().strip()
    resolved = _LANGUAGE_MAP.get(lang_lower, language)
    return resolved


class Qwen3TTS(BaseTTS):
    """Qwen3-TTS model wrapper for Coqui TTS.

    This wraps the ``qwen-tts`` Python package and exposes the Qwen3-TTS model
    family through the standard Coqui TTS API.  It supports:

    * **Custom voice generation** with preset speakers (``CustomVoice`` variants).
    * **Voice cloning** from a reference audio clip (``Base`` variants).
    * **Voice design** from natural language descriptions (``VoiceDesign`` variant).

    Example:
        >>> from TTS.api import TTS
        >>> tts = TTS("tts_models/multilingual/multi-dataset/qwen3_tts").to("cuda")
        >>> tts.tts_to_file("Hello world!", speaker="Ryan", language="English", file_path="out.wav")
    """

    config: Qwen3TTSConfig

    def __init__(self, config: Coqpit) -> None:
        super().__init__(config=config, ap=None, tokenizer=None, speaker_manager=None, language_manager=None)
        self.qwen_model = None
        self._model_variant = config.model_variant
        self._is_custom_voice = "CustomVoice" in self._model_variant
        self._is_voice_design = "VoiceDesign" in self._model_variant
        self._is_base = not self._is_custom_voice and not self._is_voice_design

    @property
    def device(self) -> torch.device:
        """Return the device the model is on."""
        if self.qwen_model is not None and hasattr(self.qwen_model, "device"):
            return self.qwen_model.device
        return torch.device("cpu")

    @staticmethod
    def init_from_config(config: "Qwen3TTSConfig", **kwargs) -> "Qwen3TTS":
        """Create a Qwen3TTS instance from a config.

        Args:
            config: A Qwen3TTSConfig instance.

        Returns:
            A new Qwen3TTS instance (model not yet loaded).
        """
        return Qwen3TTS(config)

    def load_checkpoint(
        self,
        config: "Qwen3TTSConfig",
        checkpoint_dir: str | os.PathLike[Any] | None = None,
        checkpoint_path: str | os.PathLike[Any] | None = None,
        eval: bool = False,
        strict: bool = True,
        **kwargs,
    ) -> None:
        """Load the Qwen3-TTS model.

        For Qwen3-TTS the checkpoint directory is expected to be the HuggingFace
        model directory (or model ID).  When downloaded via :class:`ModelManager`,
        ``checkpoint_dir`` points to the local snapshot of the HuggingFace repo.

        Args:
            config: Model configuration.
            checkpoint_dir: Path to the model directory (HuggingFace repo snapshot).
            checkpoint_path: Alternative path (used as directory).
            eval: Whether to set the model to evaluation mode.
            strict: Not used for Qwen3-TTS.
        """
        Qwen3TTSModel = _import_qwen_tts()

        # Determine the model path – either a local directory or a HF model ID
        model_path = None
        if checkpoint_dir is not None:
            model_path = str(checkpoint_dir)
        elif checkpoint_path is not None:
            p = Path(checkpoint_path)
            model_path = str(p if p.is_dir() else p.parent)
        else:
            model_path = config.model_variant

        # Resolve torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(config.dtype, torch.bfloat16)

        logger.info("Loading Qwen3-TTS model from: %s", model_path)
        self.qwen_model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="auto",
            dtype=dtype,
            attn_implementation=config.attn_implementation,
        )
        self.config = config

        # Initialize language manager so the TTS API can report available languages
        self.language_manager = LanguageManager()
        self.language_manager.name_to_id = {lang: i for i, lang in enumerate(config.languages)}

        if eval:
            self.eval()

    def synthesize(
        self,
        text: str,
        config: BaseTTSConfig | None = None,
        *,
        speaker: str | None = None,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        language: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize.
            config: DEPRECATED. Not used.
            speaker: Speaker name for CustomVoice models, or a speaker ID for
                cached voices in voice cloning.
            speaker_wav: Path(s) to reference audio for voice cloning (Base models).
            voice_dir: Folder for cached voices (not used for Qwen3-TTS).
            language: Language of the text (e.g. ``"English"``, ``"Chinese"``,
                ``"en"``, ``"zh"``).
            **kwargs: Additional generation parameters forwarded to the underlying
                Qwen3-TTS model (``temperature``, ``top_k``, ``top_p``,
                ``max_new_tokens``, ``instruct``, ``ref_text``, etc.).

        Returns:
            Dictionary with ``"wav"`` key containing the output waveform as a
            numpy array.
        """
        if config is not None:
            warn_synthesize_config_deprecated()
        if (speaker_id := kwargs.pop("speaker_id", None)) is not None:
            speaker = speaker_id
            warn_synthesize_speaker_id_deprecated()
        for key in ("use_griffin_lim", "do_trim_silence", "extra_aux_input"):
            kwargs.pop(key, None)

        if self.qwen_model is None:
            msg = "Model not loaded. Call load_checkpoint() first."
            raise RuntimeError(msg)

        lang = _resolve_language(language)

        # Build generation kwargs from config defaults + overrides
        gen_kwargs = {
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "top_k": kwargs.pop("top_k", self.config.top_k),
            "top_p": kwargs.pop("top_p", self.config.top_p),
            "repetition_penalty": kwargs.pop("repetition_penalty", self.config.repetition_penalty),
            "max_new_tokens": kwargs.pop("max_new_tokens", self.config.max_new_tokens),
            "do_sample": kwargs.pop("do_sample", self.config.do_sample),
        }

        if self._is_custom_voice:
            wavs, sr = self._synthesize_custom_voice(text, lang, speaker, gen_kwargs, **kwargs)
        elif self._is_voice_design:
            wavs, sr = self._synthesize_voice_design(text, lang, gen_kwargs, **kwargs)
        else:
            wavs, sr = self._synthesize_voice_clone(text, lang, speaker_wav, gen_kwargs, **kwargs)

        wav = wavs[0] if isinstance(wavs, list) else wavs
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        if isinstance(wav, np.ndarray):
            wav = wav.squeeze()

        return {"wav": wav}

    def _synthesize_custom_voice(
        self,
        text: str,
        language: str,
        speaker: str | None,
        gen_kwargs: dict,
        **kwargs,
    ) -> tuple[list, int]:
        """Synthesize using a preset speaker (CustomVoice variant)."""
        if speaker is None:
            speaker = _CUSTOM_VOICE_SPEAKERS[0]
            logger.warning("No speaker specified, using default: %s", speaker)
        instruct = kwargs.pop("instruct", "")
        wavs, sr = self.qwen_model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct,
            **gen_kwargs,
            **kwargs,
        )
        return wavs, sr

    def _synthesize_voice_design(
        self,
        text: str,
        language: str,
        gen_kwargs: dict,
        **kwargs,
    ) -> tuple[list, int]:
        """Synthesize using a voice description (VoiceDesign variant)."""
        instruct = kwargs.pop("instruct", "")
        if not instruct:
            logger.warning("VoiceDesign model works best with an 'instruct' parameter describing the desired voice.")
        wavs, sr = self.qwen_model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruct,
            **gen_kwargs,
            **kwargs,
        )
        return wavs, sr

    def _synthesize_voice_clone(
        self,
        text: str,
        language: str,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None,
        gen_kwargs: dict,
        **kwargs,
    ) -> tuple[list, int]:
        """Synthesize using voice cloning from reference audio (Base variant)."""
        ref_text = kwargs.pop("ref_text", None)

        if speaker_wav is None:
            msg = (
                "Qwen3-TTS Base model requires a reference audio for voice cloning. "
                "Pass `speaker_wav` with the path to a reference audio file."
            )
            raise ValueError(msg)

        ref_audio = speaker_wav
        if isinstance(ref_audio, list):
            ref_audio = ref_audio[0]
        ref_audio = str(ref_audio)

        clone_kwargs = {
            "text": text,
            "language": language,
            "ref_audio": ref_audio,
        }
        if ref_text is not None:
            clone_kwargs["ref_text"] = ref_text
        clone_kwargs.update(gen_kwargs)
        clone_kwargs.update(kwargs)

        wavs, sr = self.qwen_model.generate_voice_clone(**clone_kwargs)
        return wavs, sr

    def _clone_voice(
        self,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]],
        **generate_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate voice data from reference audio for caching.

        This is called by the :class:`CloningMixin` base class.
        """
        if not self._is_base:
            msg = "Voice cloning is only supported with the Base model variants."
            raise RuntimeError(msg)

        ref_audio = speaker_wav
        if isinstance(ref_audio, list):
            ref_audio = ref_audio[0]
        ref_audio = str(ref_audio)

        voice = {"ref_audio": ref_audio}
        metadata = {"name": self.config.model}
        return voice, metadata

    @torch.inference_mode()
    def inference(
        self,
        text: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Run inference on a single text input.

        This delegates to :meth:`synthesize`.
        """
        return self.synthesize(text, **kwargs)

    def forward(self):
        """Not used for inference-only model."""

    def train_step(self):
        """Not used for inference-only model."""

    def eval(self) -> "Qwen3TTS":
        """Set the model to evaluation mode."""
        # The underlying Qwen model manages its own eval state
        return self

    def cuda(self, device: int | torch.device | None = None) -> "Qwen3TTS":
        """Move the model to CUDA.

        Qwen3-TTS uses HuggingFace ``device_map`` for device management,
        so this method is a no-op if the model is already loaded.
        """
        if self.qwen_model is not None:
            logger.info("Qwen3-TTS model device placement is managed by HuggingFace device_map.")
        return self

    def to(self, *args, **kwargs) -> "Qwen3TTS":
        """Override to prevent moving the HuggingFace-managed model.

        The underlying model uses HuggingFace ``device_map="auto"`` which
        handles device placement internally.
        """
        return self
