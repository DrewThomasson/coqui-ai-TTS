"""
Simplified working StyleTTS2 implementation for Coqui TTS
This focuses on functionality over perfect architectural compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
import math


class SimpleStyleEncoder(nn.Module):
    """Simplified style encoder that works reliably."""
    
    def __init__(self, input_dim=80, style_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.style_dim = style_dim
        
        # Simple convolutional encoder
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 2, 1),  # downsample
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 2, 1),  # downsample
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.linear = nn.Linear(256, style_dim)
        
    def forward(self, x):
        # x: [B, input_dim, T] or [B, 1, input_dim, T] 
        if x.dim() == 4:
            # Remove channel dimension if present
            x = x.squeeze(1)  # [B, input_dim, T]
        
        x = self.conv_layers(x)  # [B, 256, 1]
        x = x.squeeze(-1)  # [B, 256]
        x = self.linear(x)  # [B, style_dim]
        return x


class SimpleDecoder(nn.Module):
    """Simplified decoder that works reliably."""
    
    def __init__(self, input_dim=512, style_dim=128, output_dim=80):
        super().__init__()
        self.input_dim = input_dim
        self.style_dim = style_dim
        self.output_dim = output_dim
        
        # Style-conditioned layers
        self.style_proj = nn.Linear(style_dim, input_dim)
        
        # Simple upsampling decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(input_dim, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, 1, 1),  
            nn.ReLU(),
            nn.Conv1d(256, output_dim, 3, 1, 1)
        )
        
    def forward(self, x, style):
        # x: [B, input_dim, T]
        # style: [B, style_dim]
        
        # Apply style conditioning
        style_proj = self.style_proj(style).unsqueeze(-1)  # [B, input_dim, 1]
        x = x + style_proj  # Broadcast add
        
        # Decode
        x = self.decoder(x)  # [B, output_dim, T]
        return x


class SimpleDiffusion(nn.Module):
    """Simplified diffusion that works reliably."""
    
    def __init__(self, style_dim=128):
        super().__init__()
        self.style_dim = style_dim
        
        self.net = nn.Sequential(
            nn.Linear(style_dim * 2, style_dim * 2),
            nn.ReLU(),
            nn.Linear(style_dim * 2, style_dim * 2),
            nn.ReLU(),
            nn.Linear(style_dim * 2, style_dim * 2)
        )
        
    def forward(self, x, timesteps=None, context=None):
        # x: [B, style_dim * 2]
        return self.net(x)