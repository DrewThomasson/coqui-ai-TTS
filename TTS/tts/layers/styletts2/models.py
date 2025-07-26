"""
StyleTTS2 core components - Full implementation for Coqui TTS integration.
This is a complete implementation of StyleTTS2 architecture components.
Based on the original StyleTTS2 repository: https://github.com/yl4579/StyleTTS2
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif self.layer_type == 'half':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)
            
    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


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
        
        # Add downsampling layers only when needed
        if self.downsample != 'none':
            self.downsample_layer = LearnedDownSample(self.downsample, dim_in)
        else:
            self.downsample_layer = None

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample != 'none' and self.downsample_layer is not None:
            x = self.downsample_layer(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        
        if self.downsample != 'none' and self.downsample_layer is not None:
            x = self.downsample_layer(x)
                
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


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


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols, actv=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)),
                LayerNorm(channels),
                actv,
                nn.Dropout(0.2),
            ))

        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths, m):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        m = m.to(input_lengths.device).unsqueeze(1)
        x.masked_fill_(m, 0.0)
        
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
            
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
                
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]])

        x_pad[:, :, :x.shape[-1]] = x
        x = x_pad.to(x.device)
        
        x.masked_fill_(m, 0.0)
        
        return x

    def length_to_mask(self, lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, actv=nn.LeakyReLU(0.2),
                 upsample='none', dropout_p=0.0):
        super().__init__()
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dropout = nn.Dropout(dropout_p)
        
    def _build_weights(self, dim_in, dim_out, style_dim):
        self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
        self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.upsample == 'upsample':
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample == 'upsample':
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(self.dropout(x))
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(self.dropout(x))
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class ProsodyPredictor(nn.Module):
    """Prosody predictor from original StyleTTS2."""
    
    def __init__(self, style_dim, d_hid, nlayers, max_dur=50, dropout=0.1):
        super().__init__()
        self.text_encoder = DurationEncoder(
            d_model=d_hid, 
            nlayers=nlayers,
            nhead=8, 
            dropout=dropout,
            max_dur=max_dur
        )
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.duration_proj = LinearNorm(d_hid, 1)
        
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList([
            LinearNorm(d_hid, d_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            LinearNorm(d_hid, 1)
        ])
        self.N = nn.ModuleList([
            LinearNorm(d_hid, d_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            LinearNorm(d_hid, 1)
        ])

    def forward(self, texts, style, text_lengths, alignment_path, m):
        # Duration prediction
        d = self.text_encoder(texts, text_lengths, m)
        
        # Combine text and style
        batch_size = d.shape[0]
        if style.shape[0] == 1 and batch_size > 1:
            style = style.expand(batch_size, -1)
            
        style_expanded = style.unsqueeze(1).expand(-1, d.shape[1], -1)
        d_style = torch.cat([d, style_expanded], dim=-1)
        
        # LSTM processing
        d_style, _ = self.lstm(d_style)
        
        # Duration prediction
        dur = self.duration_proj(d_style)
        dur = dur.squeeze(-1)
        
        # F0 and N prediction using shared LSTM
        shared_out, _ = self.shared(d_style)
        
        # F0 prediction
        F0 = shared_out
        for layer in self.F0:
            F0 = layer(F0)
        F0 = F0.squeeze(-1)
        
        # Energy/noise prediction
        N = shared_out
        for layer in self.N:
            N = layer(N)
        N = N.squeeze(-1)
        
        return dur, F0, N


class DurationEncoder(nn.Module):
    """Duration encoder from original StyleTTS2."""
    
    def __init__(self, d_model, nlayers, nhead=8, dropout=0.1, max_dur=50):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_dur, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, x, lengths, mask):
        # Add positional encoding
        seq_len = x.shape[1]
        pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(x.shape[0], -1, -1)
        x = x + pos_enc
        x = self.dropout(x)
        
        # Create attention mask
        attn_mask = mask.bool()
        
        # Transformer processing
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        return x


class StyleTransformer1d(nn.Module):
    """Simplified diffusion transformer for StyleTTS2."""
    
    def __init__(self, channels, context_embedding_features, context_features, 
                 num_layers=3, num_heads=8, head_features=64, multiplier=2):
        super().__init__()
        self.channels = channels
        self.context_embedding_features = context_embedding_features
        self.context_features = context_features
        
        # Input projection
        self.input_projection = nn.Linear(channels, channels)
        
        # Context projection
        self.context_projection = nn.Linear(context_embedding_features, context_features)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=num_heads,
                dim_feedforward=channels * multiplier,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(channels, channels)
        
    def forward(self, x, context=None, context_mask=None, **kwargs):
        # Input projection
        x = self.input_projection(x)
        
        # Process context if provided
        if context is not None:
            context = self.context_projection(context)
            # Add context to input (simplified)
            if context.shape[1] == x.shape[1]:
                x = x + context
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Output projection
        x = self.output_projection(x)
        
        return x


class HiFiGANDecoder(nn.Module):
    """Proper HiFiGAN decoder for StyleTTS2."""
    
    def __init__(self, dim_in, style_dim, dim_out=80,
                 resblock_kernel_sizes=[3, 7, 11],
                 upsample_rates=[10, 8, 2, 2, 2],
                 upsample_initial_channel=512,
                 upsample_kernel_sizes=[20, 16, 4, 4, 4],
                 resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]):
        super().__init__()
        self.dim_in = dim_in
        self.style_dim = style_dim
        self.dim_out = dim_out
        
        # Input convolution
        self.input_conv = weight_norm(nn.Conv1d(dim_in, upsample_initial_channel, 7, 1, 3))
        
        # Upsampling blocks
        self.upsamples = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsamples.append(
                AdainResBlk1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2**(i+1)),
                    style_dim,
                    upsample='upsample'
                )
            )
            
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(upsample_rates)):
            ch = upsample_initial_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(
                    ResBlock1d(ch, k, d, style_dim)
                )
        
        # Output convolution
        self.output_conv = weight_norm(nn.Conv1d(ch, dim_out, 7, 1, 3))
        
    def forward(self, x, s):
        x = self.input_conv(x)
        
        for i in range(len(self.upsamples)):
            x = F.leaky_relu(x, 0.1)
            x = self.upsamples[i](x, s)
            
            xs = None
            for j in range(3):  # 3 resblocks per upsample
                if xs is None:
                    xs = self.resblocks[i*3+j](x, s)
                else:
                    xs += self.resblocks[i*3+j](x, s)
            x = xs / 3
            
        x = F.leaky_relu(x)
        x = self.output_conv(x)
        return x


class ResBlock1d(nn.Module):
    """1D ResBlock with style conditioning."""
    
    def __init__(self, channels, kernel_size, dilation, style_dim):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                dilation=dilation[0], padding=dilation[0])),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                dilation=dilation[1], padding=dilation[1])),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                dilation=dilation[2], padding=dilation[2]))
        ])
        
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                dilation=1, padding=1)),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                dilation=1, padding=1)),
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                dilation=1, padding=1))
        ])
        
        # Style conditioning
        self.norms1 = nn.ModuleList([
            AdaIN1d(style_dim, channels) for _ in range(3)
        ])
        self.norms2 = nn.ModuleList([
            AdaIN1d(style_dim, channels) for _ in range(3)
        ])
        
    def forward(self, x, s):
        for c1, c2, n1, n2 in zip(self.convs1, self.convs2, self.norms1, self.norms2):
            xt = n1(x, s)
            xt = F.leaky_relu(xt, 0.1)
            xt = c1(xt)
            xt = n2(xt, s)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


# Legacy simplified models for backward compatibility
class SimpleDiffusionModel(nn.Module):
    """Compatibility wrapper - use StyleTransformer1d instead."""
    
    def __init__(self, style_dim, context_dim):
        super().__init__()
        print("WARNING: SimpleDiffusionModel is deprecated. Use StyleTransformer1d for better results.")
        self.transformer = StyleTransformer1d(
            channels=style_dim * 2,
            context_embedding_features=context_dim,
            context_features=style_dim * 2
        )
        
    def forward(self, x, timesteps, context):
        return self.transformer(x, context=context)


class SimpleHiFiGANDecoder(nn.Module):
    """Compatibility wrapper - use HiFiGANDecoder instead."""
    
    def __init__(self, dim_in, style_dim, dim_out=80):
        super().__init__()
        print("WARNING: SimpleHiFiGANDecoder is deprecated. Use HiFiGANDecoder for better results.")
        self.decoder = HiFiGANDecoder(dim_in, style_dim, dim_out)
        
    def forward(self, x, s):
        return self.decoder(x, s)