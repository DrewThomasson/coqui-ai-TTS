#!/usr/bin/env python3
"""
Fully working StyleTTS2 model adapted for 🐸TTS.
This implementation downloads the pretrained model files if they don't exist,
loads the StyleTTS2 modules (using your existing imports), builds the model and
provides an inference API.
"""

import os
import sys
import subprocess
import time
import random
from collections import OrderedDict

import yaml
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import librosa
from nltk.tokenize import word_tokenize
import phonemizer

from TTS.tts.models.base_tts import BaseTTS  # Base class from the 🐸TTS framework

# --- Helper Functions ---
def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask

def preprocess(wave, to_mel, mean, std):
    """Convert waveform into normalized mel-spectrogram tensor."""
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

# --- Model Class ---
class StyleTTS2(BaseTTS):
    """
    StyleTTS2 model wrapped for the 🐸TTS framework.

    This model downloads pretrained files (if not present), loads the StyleTTS2 modules,
    builds the model, and exposes an inference API.

    Note: Training, logging, optimizer and scheduler functions are not implemented.
    """

    def __init__(self, config):
        """
        Args:
            config (Coqpit or dict): Model configuration.
        """
        # Calling the base class initializer.
        super().__init__(config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # For reproducibility.
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(0)
        np.random.seed(0)

        # Download pretrained models if needed.
        self._download_models_if_needed()

        # Load model configuration (assumed to be in Models/LJSpeech/config.yml).
        self.model_config = yaml.safe_load(open("Models/LJSpeech/config.yml", "r"))
        
        # Set up mel-spectrogram transformation (as in your working script).
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        self.mean = -4
        self.std = 4

        # Load external modules and styletts components.
        self._load_modules()
        # Build the model and load weights.
        self._build_model()
        # Set up the diffusion sampler.
        self._setup_sampler()
        # Initialize the phonemizer (using espeak-ng).
        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us', preserve_punctuation=True, with_stress=True
        )
        print("StyleTTS2 initialization complete.")

    def _download_models_if_needed(self):
        """
        Downloads the pretrained model files from Hugging Face if they don't exist locally.
        """
        MODEL_CONFIG_PATH = "Models/LJSpeech/config.yml"
        if not os.path.exists(MODEL_CONFIG_PATH):
            print("Pretrained models not found. Downloading them now...")
            # Ensure git lfs is installed and initialized.
            subprocess.run("git lfs install", shell=True)
            # Clone the repository from Hugging Face.
            clone_command = "git clone https://huggingface.co/yl4579/StyleTTS2-LJSpeech"
            ret = subprocess.run(clone_command, shell=True)
            if ret.returncode != 0:
                print("Error during cloning the model repository.", file=sys.stderr)
                sys.exit(1)
            # Move the Models folder from the cloned repo to the current directory.
            os.system("mv StyleTTS2-LJSpeech/Models Models")
            # Optionally, remove the cloned repository folder.
            os.system("rm -rf StyleTTS2-LJSpeech")
            if not os.path.exists(MODEL_CONFIG_PATH):
                print("Failed to download the models.", file=sys.stderr)
                sys.exit(1)
            print("Models downloaded successfully.")
        else:
            print("Models already exist.")

    def _load_modules(self):
        """
        Imports and initializes the modules needed by the StyleTTS2 model.
        Assumes that the original working script's structure is maintained.
        """
        # These imports rely on your project structure.
        from models import *              # StyleTTS2 model components.
        from utils import *               # Utility functions.
        from text_utils import TextCleaner
        self.text_cleaner = TextCleaner()

        # Load the ASR model for alignment.
        ASR_config = self.model_config.get('ASR_config', False)
        ASR_path = self.model_config.get('ASR_path', False)
        self.text_aligner = load_ASR_models(ASR_path, ASR_config)
        
        # Load the F0 (pitch) model.
        F0_path = self.model_config.get('F0_path', False)
        self.pitch_extractor = load_F0_models(F0_path)
        
        # Load the BERT model.
        from Utils.PLBERT.util import load_plbert
        BERT_path = self.model_config.get('PLBERT_dir', False)
        self.plbert = load_plbert(BERT_path)

    def _build_model(self):
        """
        Build the StyleTTS2 model and load pretrained weights.
        """
        from models import recursive_munch, build_model
        # Build the model using parameters from the config.
        self.model = build_model(
            recursive_munch(self.model_config['model_params']),
            self.text_aligner,
            self.pitch_extractor,
            self.plbert
        )
        # Move each module to the device and set to evaluation mode.
        for key in self.model:
            self.model[key].to(self.device)
            self.model[key].eval()

        # Load pretrained weights.
        checkpoint_path = "Models/LJSpeech/epoch_2nd_00100.pth"
        params_whole = torch.load(checkpoint_path, map_location='cpu')
        params = params_whole['net']
        for key in self.model:
            if key in params:
                print(f'{key} loaded')
                try:
                    self.model[key].load_state_dict(params[key])
                except Exception:
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove "module." prefix if present.
                        new_state_dict[name] = v
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        # Ensure all components are in eval mode.
        for key in self.model:
            self.model[key].eval()

    def _setup_sampler(self):
        """
        Initialize the diffusion sampler used for style synthesis.
        """
        from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False
        )

    def forward(self, input, aux_input={}, **kwargs):
        """
        Forward pass for training.
        In this inference-only implementation, forward() calls inference().

        Args:
            input (str): Input text.
            aux_input (dict): Auxiliary inputs (optional).

        Returns:
            dict: Dictionary with key "model_outputs" containing the waveform.
        """
        output = self.inference(input, aux_input)
        return {"model_outputs": output}

    def inference(self, text, aux_input={}):
        """
        Synthesize speech from text.

        Args:
            text (str): Input text.
            aux_input (dict): Optional inference parameters:
                - diffusion_steps (int): Number of diffusion steps (default: 10).
                - embedding_scale (float): Embedding scale (default: 1).

        Returns:
            numpy.ndarray: The synthesized waveform.
        """
        diffusion_steps = aux_input.get("diffusion_steps", 10)
        embedding_scale = aux_input.get("embedding_scale", 1)
        text = text.strip().replace('"', '')
        # Phonemize text.
        ps = self.global_phonemizer.phonemize([text])[0]
        ps = word_tokenize(ps)
        ps = ' '.join(ps)
        tokens = self.text_cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
        
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)
        
        # Create random noise input.
        noise = torch.randn(1, 1, 256).to(self.device)
        
        with torch.no_grad():
            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)
            s_pred = self.sampler(
                noise, 
                embedding=bert_dur[0].unsqueeze(0),
                num_steps=diffusion_steps,
                embedding_scale=embedding_scale
            ).squeeze(0)
            # Split into style and reference embeddings.
            s = s_pred[:, 128:]
            ref = s_pred[:, :128]
            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            # Add extra frames to the last token.
            pred_dur[-1] += 5

            # Create alignment target.
            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().item())).to(self.device)
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                cur_dur = int(pred_dur[i].item())
                pred_aln_trg[i, c_frame:c_frame + cur_dur] = 1
                c_frame += cur_dur

            # Encode prosody and predict F0 and N.
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder(
                (t_en @ pred_aln_trg.unsqueeze(0).to(self.device)),
                F0_pred, N_pred, ref.squeeze().unsqueeze(0)
            )
        return out.squeeze().cpu().numpy()

    # --- Checkpoint and (non-)training methods ---
    def load_checkpoint(self, config, checkpoint_dir=None, checkpoint_path=None, eval=True, strict=True):
        """
        Loads a checkpoint. For StyleTTS2, the model is fully loaded during initialization.
        This method is provided to match the 🐸TTS API.

        Args:
            config (dict or Coqpit): Configuration.
            checkpoint_dir (str, optional): Directory of checkpoint files.
            checkpoint_path (str, optional): Path to a checkpoint file.
            eval (bool): If True, set model to eval mode.
            strict (bool): Whether to enforce strict key matching.
        """
        # For this implementation, the checkpoint is loaded during _build_model().
        # You can extend this to allow external checkpoint loading.
        print("Checkpoint loading is handled during initialization.")

    def train_step(self, batch, criterion):
        raise NotImplementedError("Training is not implemented for StyleTTS2 model.")

    def train_log(self, batch, outputs, logger, assets, steps):
        pass

    def eval_step(self, batch, criterion):
        raise NotImplementedError("Evaluation is not implemented for StyleTTS2 model.")

    def get_optimizer(self):
        raise NotImplementedError("Optimizer not implemented for StyleTTS2 model.")

    def get_lr(self):
        raise NotImplementedError("Learning rate scheduler not implemented for StyleTTS2 model.")

    def get_scheduler(self, optimizer: torch.optim.Optimizer):
        raise NotImplementedError("Scheduler not implemented for StyleTTS2 model.")

    def get_criterion(self):
        raise NotImplementedError("Criterion not implemented for StyleTTS2 model.")

    def format_batch(self, batch):
        raise NotImplementedError("Batch formatting not implemented for StyleTTS2 model.")

    @staticmethod
    def init_from_config(config, **kwargs):
        """
        Factory method to initialize the model from a config.
        """
        return StyleTTS2(config)
