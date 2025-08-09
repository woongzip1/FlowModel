import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
from typing import Optional

class GRN1D(nn.Module):
    """ GRN1D (Global Response Normalization) layer
    """
    def __init__(self, dim, groups):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        self.groups = groups

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        if self.groups == 1:
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        else:
            Gx = Gx.view(*Gx.shape[:2], self.groups, -1)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
            Nx = Nx.view(*Nx.shape[:2], -1)
        return self.gamma * (x * Nx) + self.beta + x

class GroupLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 groups: int = 1, device=None, dtype=None) -> None:
        assert in_features % groups == 0 and out_features % groups == 0
        self.groups = groups
        super().__init__(in_features // groups, out_features, bias, device, dtype)

    def forward(self, input):
        if self.groups == 1:
            return super().forward(input)
        else:
            sh = input.shape[:-1]
            input = input.view(*sh, self.groups, -1)
            weight = self.weight.view(self.groups, -1, self.weight.shape[-1])
            output = torch.einsum('...gi,...goi->...go', input, weight)
            output = output.reshape(*sh, -1) + self.bias
            return output
        

class ConvNeXtV2Block(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        adanorm_num_embeddings: Optional[int] = None,
        groups: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None and adanorm_num_embeddings > 1
        if self.adanorm:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        elif groups == 1:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        else:
            self.norm = GroupLayerNorm(groups, dim, eps=1e-6)
        self.pwconv1 = GroupLinear(dim, intermediate_dim, groups=groups)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN1D(intermediate_dim, groups=groups)
        self.pwconv2 = GroupLinear(intermediate_dim, dim, groups=groups)

    def forward(self, x: torch.Tensor, cond_embedding_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization module with learnable embeddings per `num_embeddings` classes

    Args:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimension of the embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.shift = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        x = x * scale.unsqueeze(1) + shift.unsqueeze(1)
        return x


class GroupLayerNorm(nn.Module):
    def __init__(self, groups: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.groups = groups
        self.scale = nn.Parameter(torch.ones([groups, embedding_dim // groups]))
        self.shift = nn.Parameter(torch.zeros([groups, embedding_dim // groups]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sh = x.shape[:-1]
        x = x.reshape(*sh, self.groups, -1)
        x = nn.functional.layer_norm(x, (self.dim // self.groups,), eps=self.eps)
        x = x * self.scale + self.shift
        return x.reshape(*sh, -1)
# ------------------------ 1D --------------------


class ConditionalVectorFieldModel(nn.Module, ABC):
    """
    Base class for DNN-based VF model
    MLP-parameterization of the learned vector field u_t^theta(x)
    """

    @abstractmethod
    def forward(self, x:torch.Tensor, t:torch.Tensor, y:torch.Tensor):
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        pass
    
class SinusoidalTimeEmbedding(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    & DiffWave / WaveFM
    """
    def __init__(self, dim: int=128, mode: str='learnable', time_scale=1):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be an even number"
        assert mode in ['fixed', 'learnable'], "Mode must be 'fixed' or 'learnable'"
        
        self.dim = dim              # D
        self.half_dim = dim // 2
        self.mode = mode
        self.time_scale = time_scale  # 1(diffusion) or 100(flow)
        
        if self.mode == 'learnable':
            self.weights = nn.Parameter(torch.randn(1, self.half_dim)) # [1,D/2]
            
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: Time tensor. Shape can be [B] or [B, 1].
        Returns:
        - embeddings: Time embeddings of shape [B, D]
        """
        # Ensure t has shape [B, 1] for broadcasting
        t = t.view(-1, 1)
        device = t.device

        if self.mode == 'fixed':
            # Create a sequence from 0 to D/2 - 1
            pos = torch.arange(self.half_dim, device=device).unsqueeze(0) # [1,D/2]
            freqs = self.time_scale * t * 10.0 ** (pos * 4.0 / (self.half_dim - 1)) # 100 is a magnitude hyperparameter
            
            sin_embed = torch.sin(freqs)
            cos_embed = torch.cos(freqs)
            
            return torch.cat([sin_embed, cos_embed], dim=-1)

        elif self.mode == 'learnable':
            freqs = t * self.weights * 2 * math.pi
            
            sin_embed = torch.sin(freqs)
            cos_embed = torch.cos(freqs)
            
            return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)
        
# ----- Adopted from Facebook ConvNeXt ----- #
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    
class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x        
        
class Block(nn.Module):
    """ ConvNeXt V2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, padding_mode="reflect") 
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.grn = GRN(4 * dim) # GRN for V2
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # This Block preserves the input shape (C, H, W) -> (C, H, W)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # [N,C,H,W] -> [N,H,W,C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # [N,H,W,C] -> [N,C,H,W]

        x = input + self.drop_path(x) # Residual connection
        return x

class BlockWithEmbedding(nn.Module):
    """ ConvNeXt block with time embedding injection
    """
    def __init__(self, dim, drop_path=0., time_embed_dim=128):
        super().__init__()
        self.block = Block(dim, drop_path)
        self.time_adapter = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, dim),
        )
    def forward(self, x, t_embed):
        t_embed = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1) # [B,C,1,1]
        x = x + t_embed
        x = self.block(x)
        
        return x

class EncoderBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_blocks, drop_path, time_embed_dim):
        super().__init__()
        self.blocks= nn.ModuleList(
            [BlockWithEmbedding(dim_in, drop_path, time_embed_dim) 
             for _ in range(num_blocks)]
        )
        self.downsampler = nn.Sequential(
            LayerNorm(dim_in, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2),
        )
    
    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)
        x = self.downsampler(x)
        return x
        
class Midcoder(nn.Module):
    def __init__(self, dim, num_blocks, drop_path, time_embed_dim):
        super().__init__()
        self.blocks = nn.ModuleList(
            [BlockWithEmbedding(dim, drop_path, time_embed_dim)
             for _ in range(num_blocks)]    
        )
        
    def forward(self, x, t_emb):
        for block in self.blocks:
            x = block(x, t_emb)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_blocks, drop_path, time_embed_dim):
        super().__init__()
        self.upsampler = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.blocks = nn.ModuleList(
            [BlockWithEmbedding(dim_out, drop_path, time_embed_dim)
            for _ in range(num_blocks)]
        )
    def forward(self, x, t_emb):
        x = self.upsampler(x)
        for block in self.blocks:
            x = block(x, t_emb)
        return x

class ConditioningEncoder2D(nn.Module):
    def __init__(self, in_freq_bins, cond_dim, num_blocks=3):
        super().__init__()       
        self.cond_dim = cond_dim        
        self.max_f1 = in_freq_bins
        self.film_generator = nn.Linear(cond_dim, 4)
        self.head = nn.Conv2d(2, cond_dim, kernel_size=1)
        """
        Args:
            cond_dim (int): The main conditioning dimension (D).
            num_blocks (int): The number of shared 2D ConvNeXt blocks.
        """
        # Backbone Blocks
        self.blocks = nn.Sequential(*[
            Block(dim=cond_dim) for _ in range(num_blocks)
        ])
        self.freq_pool = nn.AdaptiveAvgPool2d((1,None))
        
    def forward(self, y_lr, f_emb_lr, sr_values, sr_emb):
        """
        Args:
            y_lr (Tensor): LR Spec [B, 2, F1, T]
            f_emb : f_emb [F,D1]
            sr_values : list of sr values [B]
        Returns:
            z (Tensor): Conditioning Emb [B, D1, T]
        """
        film_params = self.film_generator(f_emb_lr)
        gamma, beta = torch.chunk(film_params, chunks=2, dim=-1) # [F1,2]
        gamma = rearrange(gamma, 'f c -> 1 c f 1')  # [1,2,F1,1]
        beta = rearrange(beta, 'f c -> 1 c f 1')    # [1,2,F1,1]
        z = y_lr * gamma + beta # [B, 2, F1, T]
        
        z = self.head(z)
        z = self.blocks(z)      # [B,D1,F1,T]
        z = self.freq_pool(z).squeeze(2) # [B,D1,T]
        return z
    
class ConditioningEncoder(nn.Module):
    def __init__(self, in_freq_bins, cond_dim, num_blocks=3):
        """
        Args:
            in_freq_bins (int): LR spectrum F bins (F1) // variable
            cond_dim (int): Out dimension (D1)
        """
        super().__init__()
        self.max_f1 = in_freq_bins
        self.film_generator = nn.Linear(cond_dim, 4)
        # 1. Projection (MOE)
        self.projection = nn.Conv1d(in_channels=2*in_freq_bins, 
                                    out_channels=cond_dim, 
                                    kernel_size=1)
        
        # Global Spectral Encoder (Conv1D)
        # 2*F1 -> D0
        ## Must have long receptive field for F-dim 
        self.blocks = nn.Sequential(*[
            ConvNeXtV2Block(dim=cond_dim, intermediate_dim=cond_dim*3)
            for _ in range(num_blocks)
        ])
        
    def forward(self, y_lr, f_emb_lr, sr_values, sr_emb):
        """
        Args:
            y (Tensor): Low-resolution conditioning spectrogram [B, 2, F1, T].
            f_emb_lr (Tensor): [F1,D]
        Returns:
            z (Tensor): Conditioning Emb [B, D1, T]
        """
        film_params = self.film_generator(f_emb_lr)
        gamma, beta = torch.chunk(film_params, chunks=2, dim=-1) # [F1,2]
        gamma = rearrange(gamma, 'f c -> 1 c f 1')  # [1,2,F1,1]
        beta = rearrange(beta, 'f c -> 1 c f 1')    # [1,2,F1,1]
        y_lr = y_lr * gamma + beta # [B, 2, F1, T]
    
        # Reshape for Conv1D
        z = rearrange(y_lr, "b c f t -> b (c f) t") # [B,2xF,T]
        z = self.projection(z)  # [B,D1,T]
        z = self.blocks(z)      # [B,D1,T]        
        return z

class FrequencyPositionalEmbedding(nn.Module):
    def __init__(self, num_bins: int, emb_dim: int):
        super().__init__()
        # (F, D)
        pe = torch.zeros(num_bins, emb_dim)
        position = torch.arange(num_bins, dtype=torch.float32).unsqueeze(1)  # (F,1)
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2, dtype=torch.float32) *
            -(math.log(10000.0) / emb_dim)
        )  # (D/2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self):
        # returns (F, D)
        return self.pe        
        
class ConvNeXtUNetCond(ConditionalVectorFieldModel):
    def __init__(self, in_channels=2, out_channels=2,
                 dims=[64,128,256,512], depths=[2,2,2,4],
                 drop_path=0., time_dim=128,
                 cond_dim=256, # D1
                 lr_freq_bins=80, # F1
                 hr_freq_bins=432,
                 feature_enc_layers=10,
                 cond_dropout_prob=0.1,
                 sampling_rates={8: 85, 12: 128, 16: 170, 24: 256},
                 ):
        super().__init__()
        self.strides = 2**len(dims)
        self.time_embedder = SinusoidalTimeEmbedding(dim=time_dim)
        self.sr_values_list = sorted(list(sampling_rates.keys()))       # (8,12,16,24) kHz
        self.sr_to_idx = {sr: i for i, sr in enumerate(self.sr_values_list)}
        self.sr_embedder = nn.Embedding(len(self.sr_values_list), cond_dim)   # [4,D]
        self.lr_freq_bins = lr_freq_bins
        self.cond_dropout_prob = cond_dropout_prob
        self.cond_dim = cond_dim
        self.uncond_emb = nn.Parameter(torch.randn(cond_dim))

        # PE
        self.freq_pos_enc = FrequencyPositionalEmbedding(num_bins=lr_freq_bins+hr_freq_bins, emb_dim=cond_dim)
        self.film_generator = nn.Linear(cond_dim, cond_dim * 2)
        
        ## ---
        self.conditioning_encoder = ConditioningEncoder(
            in_freq_bins=lr_freq_bins,
            cond_dim=cond_dim,
            num_blocks=feature_enc_layers,
        )

        ## ---
        self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels+cond_dim, dims[0], kernel_size=1),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Encoder              
        for i in range(len(depths)):
            dim_in = dims[i]
            dim_out = dims[i+1] if i+1 < len(dims) else dims[i]
            self.encoders.append(EncoderBlock(dim_in, dim_out, depths[i], drop_path, time_dim))
        
        # Midcoder
        self.midcoder = Midcoder(dims[-1], depths[-1], drop_path, time_dim)
        
        # Decoder
        for i in reversed(range(len(depths))):
            dim_in = dims[i+1] if i+1 < len(dims) else dims[i]
            dim_out = dims[i]
            self.decoders.append(DecoderBlock(dim_in, dim_out, depths[i], drop_path, time_dim))
        
        self.final_conv = nn.Conv2d(dims[0], out_channels, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        
    def _pad_frames(self, x):
        num_frames = x.shape[-1]
        pad_len = (self.strides - num_frames % self.strides) % self.strides
        if pad_len:
            x = torch.nn.functional.pad(x, [0,pad_len,0,0], mode='reflect')
        assert x.shape[-1] % self.strides == 0, \
            f"After padding, time dim:{x.shape(-1)} must be multiples of {self.strides}"        
        return x, pad_len
        
    def forward(self, x, t, y, sr_values):
        """
        x : x_t noisy spec [B,2,F,T]
        t : time embedding [B,1] or [B] 
        y : condition lr spectrum [B,2,F,T]
        """
        # Pad logic
        x, pad_len = self._pad_frames(x)
        if pad_len > 0 and y is not None:
            y = torch.nn.functional.pad(y, [0, pad_len, 0, 0], mode='reflect')
        B, _, F, T = x.shape
        
        pe_full = self.freq_pos_enc()               # [F,D]
        pe_low = pe_full[:self.lr_freq_bins,:]      # [F1,D]
        pe_high = pe_full[self.lr_freq_bins:, :]    # [F2,D]

        t_embed = self.time_embedder(t)
                
        if y is not None:    # (Training) 
            # sr embedding
            sr_indices = torch.tensor([self.sr_to_idx[s.item()] for s in sr_values], device=x.device)
            sr_emb = self.sr_embedder(sr_indices) # [B, D]
            y_cond_real = self.conditioning_encoder(y, pe_low, sr_values, sr_emb)   # [B,D,T]    
            
            # Uncond token masking
            if self.training and self.cond_dropout_prob > 0:
                # random mask for uncond
                mask = (torch.rand(B, device=x.device) < self.cond_dropout_prob) # [B]
                uncond = self.uncond_emb.reshape(1,self.cond_dim,1).expand(B,self.cond_dim,T) # [B,D,T]
                y_cond = torch.where(mask.reshape(B,1,1), uncond, y_cond_real)
            else:
                y_cond = y_cond_real
        else: # Unconditional (inference)
            y_cond = self.uncond_emb.reshape(1,self.cond_dim,1).expand(B,self.cond_dim,T)

        y_cond = y_cond.unsqueeze(2)                            # [B,D,1,T]
                
        # FiLM Conditioning of freq-bins
        film_params = self.film_generator(pe_high) # [F2,D] -> [F2,2D]
        gamma_high, beta_high = torch.chunk(film_params, chunks=2, dim=-1) # [F2, D]
        gamma_high = rearrange(gamma_high, 'f d -> 1 d f 1') # [1,D,F2,1]
        beta_high = rearrange(beta_high, 'f d -> 1 d f 1')   # [1,D,F2,1]
        spatial_cond = y_cond * gamma_high + beta_high # [B,D,F2,T]
        
        x = torch.cat([x, spatial_cond], dim=1) # [B,2+D,F2,T]

        # Initial convolution
        x = self.init_conv(x) # (bs, c_0, 32, 32)
        skip_connections = [x]
        
        # Encoders
        for encoder in self.encoders:
            x = encoder(x, t_embed) # (bs, c_i, h, w) -> (bs, c_{i+1}, h // 2, w //2)
            skip_connections.append(x)
        
        # Midcoder
        x = self.midcoder(x, t_embed)

        # Decoders
        for decoder in self.decoders:
            skip = skip_connections.pop() # (bs, c_i, h, w)
            if x.shape != skip.shape:
                # shape mismatching due to downsampling
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = x + skip
            x = decoder(x, t_embed) # (bs, c_i, h, w) -> (bs, c_{i-1}, 2 * h, 2 * w)

        # Final convolution
        skip = skip_connections.pop()
        x = x + skip
        x = self.final_conv(x) # (bs, 1, 32, 32)

        # Crop out
        if pad_len:
            x = x[...,:-pad_len]
        return x

"""
input xt (Spectral): [B,2,F2,T]

lr condition y: [B,2,F1,T] -> (Reshape) -> [B,2*F1,T]

------------------
1. Global Spectral Encoder (Conv1D)
[B,2*F1,T] -> [B,D0,T]

2. MOE (Linear Projection)
D0 is a varialbe length depending on LR sampling rate
D1 is a fixed-length dimension for conditioning
[B,D0,T] -> [B,D1,T]

3. Freq-bin Encoder
Add frequency-poisition embedding and dimension
[B,D1,T] -> [B,D1,F2,T] // dimension of F2 is added

** Detailed method **
t_emb : [B,D1,1,1]
Y_cond: [B,D1,1,T]
F_emb:  [B,D1,F,1]

sum result : [B,D1,F,T]

4. Conditioning in the VF estimator input
[xt, y] -> [B,2+D1,F2,T] // throw it back into UNet

"""

from torchinfo import summary
from src.utils.utils import count_model_params

def main():
    # Hyperparameters
    batch_size = 2
    F2 = 432  # High-res FFT bins (NFFT/2)
    F1 = 80   # Low-res LR bins
    T = 256   # Number of time frames
    cond_dim = 384
    feature_enc_layers = 12
    # Instantiate the model
    model = ConvNeXtUNetCond(
        in_channels=2,
        out_channels=2,
        dims=[96, 192, 384, 768],
        depths=[2,2,4,2],
        drop_path=0.0,
        time_dim=256,
        cond_dim=cond_dim,
        feature_enc_layers=feature_enc_layers,
        lr_freq_bins=F1,
        hr_freq_bins=F2,
        cond_dropout_prob=0.1,
        sampling_rates=None,
    )
    
    # Dummy inputs
    x = torch.randn(batch_size, 2, F2, T)
    y = torch.randn(batch_size, 2, F1, T)
    t = torch.randint(0, 1000, (batch_size,))

    # ---
    # for num_block in [3, 6, 9, 12]:
    #     model = ConditioningEncoder(in_freq_bins=F1, cond_dim=384, num_blocks=num_block)    
    #     p = count_model_params(model)
    #     print(p)
    # pdb.set_trace()
    
    # Print model summary
    summary(
        model,
        input_data=(x, t, y),
        depth=3,
        col_names=("input_size", "output_size", "num_params", "kernel_size"),
        verbose=1
    )

if __name__ == "__main__":
    main()

# # --------- without F2 ------------ #
# def main():
#     # Hyperparameters
#     batch_size = 2
#     F2 = 432  # High-res FFT bins (NFFT/2)
#     F1 = 80   # Low-res LR bins
#     F = 512
#     T = 256   # Number of time frames
#     cond_dim = 256

#     # Instantiate the model
#     model = ConvNeXtUNetCond(
#         in_channels=2,
#         out_channels=2,
#         dims=[96, 192, 384, 768],
#         depths=[2,2,4,2],
#         drop_path=0.0,
#         time_dim=cond_dim,
#         cond_dim=cond_dim,
#         lr_freq_bins=F1,
#         hr_freq_bins=F,
#     )
    
#     # Dummy inputs
#     x = torch.randn(batch_size, 2, F, T)
#     y = torch.randn(batch_size, 2, F1, T)
#     t = torch.randint(0, 1000, (batch_size,))

#     # Print model summary
#     summary(
#         model,
#         input_data=(x, t, y),
#         depth=3,
#         col_names=("input_size", "output_size", "num_params", "kernel_size"),
#         verbose=1
#     )

# if __name__ == "__main__":
#     main()
# # --------- without F2 ------------ #


