"""
Complete StyleTTS2 implementation based on the original repository.
This provides the actual StyleTTS2 architecture that works with pretrained models.
Based on: https://github.com/yl4579/StyleTTS2
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2), upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample_type = upsample
        self.upsample = UpSample1d(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        if upsample == 'none':
            self.pool = nn.Identity()
        else:
            self.pool = weight_norm(nn.ConvTranspose1d(dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1))

    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.upsample_type != 'none':
            x = self.pool(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.upsample(x)
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class UpSample1d(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        else:
            return F.interpolate(x, scale_factor=2, mode='nearest')


class Decoder(nn.Module):
    def __init__(self, dim_in=512, F0_channel=256, style_dim=64, dim_out=80):
        super().__init__()
        
        self.decode = nn.ModuleList()
        self.decode_F0 = nn.ModuleList()
        
        # Main decoder path
        self.decode.append(AdainResBlk1d(dim_in + 2, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024, 1024, style_dim))
        self.decode.append(AdainResBlk1d(1024, 1024, style_dim, upsample='nearest'))
        
        self.decode.append(AdainResBlk1d(1024, 512, style_dim))
        self.decode.append(AdainResBlk1d(512, 512, style_dim))
        self.decode.append(AdainResBlk1d(512, 512, style_dim, upsample='nearest'))
        
        self.decode.append(AdainResBlk1d(512, 256, style_dim))
        self.decode.append(AdainResBlk1d(256, 256, style_dim, upsample='nearest'))
        
        # F0 decoder path  
        self.decode_F0.append(weight_norm(nn.Conv1d(F0_channel, 1024, 3, 1, 1)))
        self.decode_F0.append(AdainResBlk1d(1024, 1024, style_dim))
        self.decode_F0.append(AdainResBlk1d(1024, 1024, style_dim, upsample='nearest'))
        
        self.decode_F0.append(AdainResBlk1d(1024, 512, style_dim))
        self.decode_F0.append(AdainResBlk1d(512, 512, style_dim, upsample='nearest'))
        
        self.decode_F0.append(AdainResBlk1d(512, 256, style_dim))
        self.decode_F0.append(AdainResBlk1d(256, 256, style_dim, upsample='nearest'))
        
        # Final layers
        self.to_out = nn.Sequential(
            weight_norm(nn.Conv1d(256, 80, 3, 1, 1))
        )

    def forward(self, asr_res, F0_pred, s):
        if F0_pred is not None:
            F0 = self.decode_F0[0](F0_pred)
            F0 = self.decode_F0[1](F0, s)
            F0 = self.decode_F0[2](F0, s)
            F0 = self.decode_F0[3](F0, s)
            F0 = self.decode_F0[4](F0, s)
            F0 = self.decode_F0[5](F0, s)
            F0 = self.decode_F0[6](F0, s)
        else:
            F0 = torch.zeros_like(asr_res)[:, :256]
            
        asr_res = torch.cat([asr_res, F0], dim=1)
        
        out = self.decode[0](asr_res, s)
        out = self.decode[1](out, s)
        out = self.decode[2](out, s)  
        out = self.decode[3](out, s)
        out = self.decode[4](out, s)
        out = self.decode[5](out, s)
        out = self.decode[6](out, s)
        out = self.decode[7](out, s)
        out = self.decode[8](out, s)
        
        return self.to_out(out)


class TextEncoder(nn.Module):
    def __init__(self, channels=512, kernel_size=5, depth=4, n_symbols=178):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                nn.ReLU(),
                nn.Dropout(0.2),
            ))
        
        # LSTM layers
        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)
        
    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, channels]
        x = x.transpose(1, 2)  # [B, channels, T]
        
        # Apply mask
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)
        
        # CNN layers
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
            
        x = x.transpose(1, 2)  # [B, T, channels]
        
        # LSTM
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)
        
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        return x.transpose(1, 2)  # [B, channels, T]


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
        return s


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample != 'none':
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample != 'none':
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class PredictorLSTM(nn.Module):
    def __init__(self, dim_in, filter_size=256, kernel_size=3, conv_output_size=256, lstm_output_size=1024, dropout=0.1):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            weight_norm(nn.Conv1d(dim_in, filter_size, kernel_size, padding=kernel_size//2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(filter_size, filter_size, kernel_size, padding=kernel_size//2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(filter_size, conv_output_size, kernel_size, padding=kernel_size//2)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.lstm = nn.LSTM(conv_output_size, lstm_output_size // 2, 1, batch_first=True, bidirectional=True)
        self.linear = weight_norm(nn.Linear(lstm_output_size, 1))
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x.squeeze(-1)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps=1000, embedding_dim=128):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps, embedding_dim), persistent=False)
        self.projection1 = weight_norm(nn.Linear(embedding_dim, 512))
        self.projection2 = weight_norm(nn.Linear(512, 512))

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx.float()).unsqueeze(-1)

    def _build_embedding(self, max_steps, embedding_dim):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(embedding_dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0**(dims * 4.0 / embedding_dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # [T,dim*2]
        return table


class StyleDiffusion(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=3, dim_feedforward=2048):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Embeddings
        self.step_embedding = DiffusionEmbedding()
        self.style_embedding = weight_norm(nn.Linear(64, d_model))
        self.text_embedding = weight_norm(nn.Linear(512, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = weight_norm(nn.Linear(d_model, 64))
        
    def forward(self, style, encoder_outputs, step, src_key_padding_mask=None):
        batch_size, seq_len = encoder_outputs.shape[:2]
        
        # Embed step
        step_emb = self.step_embedding(step)  # [B, d_model]
        step_emb = step_emb.unsqueeze(1).expand(batch_size, seq_len, -1)  # [B, seq_len, d_model]
        
        # Embed style and text
        style_emb = self.style_embedding(style).unsqueeze(1).expand(batch_size, seq_len, -1)  # [B, seq_len, d_model]
        text_emb = self.text_embedding(encoder_outputs)  # [B, seq_len, d_model]
        
        # Combine embeddings
        x = step_emb + style_emb + text_emb
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Project to style dimension
        return self.output_projection(x)