# StyleTTS2 TTS API Integration

StyleTTS2 is now fully integrated with the Coqui TTS Python API, making it easy to use for high-quality text-to-speech synthesis and voice cloning.

## Quick Start

### Basic Usage

```python
from TTS.api import TTS

# Load StyleTTS2
tts = TTS(model_name="tts_models/multilingual/multi-dataset/styletts2")

# Generate speech
wav = tts.tts("StyleTTS2 produces human-level speech synthesis!")

# Save to file
tts.tts_to_file("Hello world!", file_path="output.wav")
```

### Voice Cloning

```python
# Clone voice from reference audio
wav = tts.tts("This will sound like the reference speaker!", 
              speaker_wav="path/to/reference.wav")

# Save cloned voice
tts.tts_to_file("Cloned voice output!", 
                speaker_wav="reference.wav",
                file_path="cloned_voice.wav")
```

## API Compatibility

StyleTTS2 now supports all standard TTS API methods:

- `TTS.list_models()` - Includes StyleTTS2 models
- `tts.tts(text, speaker_wav=None)` - Text-to-speech with optional voice cloning
- `tts.tts_to_file(text, file_path, speaker_wav=None)` - Direct file output
- `tts.speakers` - Speaker management (if multi-speaker model)
- `tts.languages` - Language support (if multi-lingual model)

## Voice Cloning Parameters

- `speaker_wav`: Path to reference audio file (3-10 seconds recommended)
- `alpha`: Style interpolation factor (0.0-1.0, default 0.3)
- `diffusion_steps`: Number of diffusion steps (default 10)

## Advanced Usage

For advanced features, you can use StyleTTS2 directly:

```python
from TTS.tts.configs.styletts2_config import StyleTTS2Config
from TTS.tts.models.styletts2 import Styletts2

config = StyleTTS2Config()
model = Styletts2.init_from_config(config, [])

# Direct inference
mel = model.inference_with_text("Advanced usage text")

# Voice cloning with custom parameters
mel = model.clone_voice(
    text="Custom voice cloning",
    reference_wav="reference.wav",
    alpha=0.5,
    diffusion_steps=20
)
```

## Integration Features

✅ **Full TTS API Compatibility**
- Discoverable through model registry
- Standard interface methods
- Voice cloning support
- File I/O operations

✅ **StyleTTS2 Specific Features**
- Style diffusion for natural synthesis
- Zero-shot voice cloning
- Dual style encoders (acoustic + prosodic)
- Configurable diffusion parameters

✅ **Production Ready**
- Compatible with existing TTS pipelines
- Follows Coqui TTS configuration patterns
- Supports batch processing
- Memory efficient inference

## Example Applications

1. **Content Creation**: Generate natural-sounding narrations
2. **Voice Cloning**: Clone voices from short reference samples
3. **Interactive Applications**: Real-time speech synthesis
4. **Accessibility**: Convert text to speech for visually impaired users
5. **Language Learning**: Generate pronunciation examples

StyleTTS2 brings human-level speech synthesis to the Coqui TTS ecosystem with full API compatibility!