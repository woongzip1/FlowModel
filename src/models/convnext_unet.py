import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from timm.models.layers import DropPath 

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
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
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
        
class ConvNeXtUNet(ConditionalVectorFieldModel):
    def __init__(self, in_channels=4, out_channels=2,
                 dims=[64,128,256,512], depths=[2,2,2,4],
                 drop_path=0., time_dim=128
                 ):
        super().__init__()
        self.strides = 2**len(dims)
        self.time_embedder = SinusoidalTimeEmbedding(dim=128)
        self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels, dims[0], kernel_size=1),
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
        
    def _pad_frames(self, x):
        num_frames = x.shape[-1]
        pad_len = (self.strides - num_frames % self.strides) % self.strides
        if pad_len:
            x = torch.nn.functional.pad(x, [0,pad_len,0,0])
        assert x.shape[-1] % self.strides == 0, \
            f"After padding, time dim:{x.shape(-1)} must be multiples of {self.strides}"        
        return x, pad_len
        
    def forward(self, x, t, y):
        """
        x : x_t noisy spec [B,2,F,T]
        t : time embedding [B,1] or [B] 
        y : condition lr spectrum [B,2,F,T]
        """
        # Pad logic
        x = torch.cat([x,y], dim=1)
        x, pad_len = self._pad_frames(x)
        t_embed = self.time_embedder(t)
        
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
        
def main():
    from torchinfo import summary
    model = ConvNeXtUNet(in_channels=4, out_channels=2, dims=[64,128,256,512], depths=[2,2,4,4])
    # out = model(torch.randn(5,2,512,100), torch.randn(5))
    summary(
        model,
        input_data=[torch.randn(5,2,512,100), torch.randn(5), torch.randn(5,2,512,100)],
        depth=4,
        col_names=["input_size", "output_size", "num_params"],
        verbose=1
    )
    
if __name__ == "__main__":
    main()