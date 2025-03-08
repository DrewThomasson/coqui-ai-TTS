import os
import shutil
import phonemizer

# Ensure espeak is accessible in the PATH
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

print("\n### Debugging Phonemizer & Espeak Setup ###\n")

# Check where espeak is located
espeak_path = shutil.which("espeak")
if espeak_path:
    print(f"✔ Found espeak at: {espeak_path}")
else:
    print("✖ ERROR: espeak not found in PATH!")

# Check phonemizer installation
try:
    print(f"✔ Phonemizer installed at: {phonemizer.__file__}")
except ImportError:
    print("✖ ERROR: Phonemizer is not installed!")

# Try initializing phonemizer backend
try:
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us',
        preserve_punctuation=True,
        with_stress=True,
        words_mismatch='ignore'
    )
    print("✔ Phonemizer EspeakBackend initialized successfully!")
except Exception as e:
    print(f"✖ ERROR initializing Phonemizer EspeakBackend:\n{e}")

# Try phonemizing a test sentence
try:
    test_sentence = "This is a test."
    phonemized_text = global_phonemizer.phonemize([test_sentence])
    print(f"✔ Phonemized output: {phonemized_text}")
except Exception as e:
    print(f"✖ ERROR during phonemization:\n{e}")

print("\n### End of Debugging ###\n")
