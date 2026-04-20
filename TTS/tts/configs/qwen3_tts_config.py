from dataclasses import dataclass, field

from coqpit import Coqpit

from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class Qwen3TTSAudioConfig(Coqpit):
    """Audio configuration for the Qwen3-TTS model.

    Args:
        sample_rate (int): The sample rate of the output audio waveform. Defaults to 24000.
        output_sample_rate (int): The sample rate of the output audio waveform. Defaults to 24000.
    """

    sample_rate: int = 24000
    output_sample_rate: int = 24000


@dataclass
class Qwen3TTSConfig(BaseTTSConfig):
    """Configuration for the Qwen3-TTS model.

    Qwen3-TTS is a multilingual text-to-speech model supporting 10 languages with
    voice cloning, voice design, and custom voice generation capabilities.

    Args:
        model (str):
            Model name used for dynamic module discovery. Do not change.

        model_variant (str):
            The HuggingFace model variant to use. Defaults to ``Qwen/Qwen3-TTS-12Hz-1.7B-Base``.
            Options include:
            - ``Qwen/Qwen3-TTS-12Hz-1.7B-Base`` (voice cloning)
            - ``Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`` (preset speakers with instruction control)
            - ``Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`` (voice design from descriptions)
            - ``Qwen/Qwen3-TTS-12Hz-0.6B-Base`` (smaller voice cloning)
            - ``Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`` (smaller preset speakers)

        tokenizer_model (str):
            The HuggingFace model name for the speech tokenizer. Defaults to
            ``Qwen/Qwen3-TTS-Tokenizer-12Hz``.

        audio (Qwen3TTSAudioConfig):
            Audio processing configuration.

        languages (list[str]):
            List of supported language codes.

        temperature (float):
            Generation temperature. Higher values make output more diverse. Defaults to 0.9.

        top_k (int):
            Top-k sampling parameter. Defaults to 50.

        top_p (float):
            Top-p (nucleus) sampling parameter. Defaults to 1.0.

        repetition_penalty (float):
            Repetition penalty for generation. Defaults to 1.05.

        max_new_tokens (int):
            Maximum number of new tokens to generate. Defaults to 2048.

        dtype (str):
            Data type for model weights. Defaults to ``bfloat16``.

        attn_implementation (str):
            Attention implementation to use. Defaults to ``sdpa``.

    Note:
        Check :class:`TTS.tts.configs.shared_configs.BaseTTSConfig` for the inherited parameters.

    Example:

        >>> from TTS.tts.configs.qwen3_tts_config import Qwen3TTSConfig
        >>> config = Qwen3TTSConfig()
    """

    model: str = "qwen3_tts"
    _supports_cloning: bool = True
    model_variant: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    tokenizer_model: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    audio: Qwen3TTSAudioConfig = field(default_factory=Qwen3TTSAudioConfig)
    languages: list[str] = field(
        default_factory=lambda: [
            "Chinese",
            "English",
            "Japanese",
            "Korean",
            "German",
            "French",
            "Russian",
            "Portuguese",
            "Spanish",
            "Italian",
        ]
    )

    # inference params
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    max_new_tokens: int = 2048
    do_sample: bool = True

    # model loading params
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
