"""
https://gist.github.com/jhauret/0b12892b8770205d999ac0b1f058156f

Implementation of Seanet[1] and StreamingSeanet[2] generators in PyTorch.

[1] Tagliasacchi, M., Li, Y., Misiunas, K., & Roblek, D. (2020).
 SEANet: A multi-modal speech enhancement network.
  https://arxiv.org/abs/2009.02095
[2] Li, Y., Tagliasacchi, M., Rybakov, O., Ungureanu, V., & Roblek, D.
 Real-time speech frequency bandwidth extension.
 https://arxiv.org/pdf/2010.10677.pdf

"""
import math
import torch
import torch.nn as nn
from math import prod

from src.flow.path import ConditionalVectorFieldModel


class SinusoidalTimeEmbedding(nn.Module):
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
            
            # The scaling factor sqrt(2) is often used in learnable Fourier features
            return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)


class GeneratorSeanet(ConditionalVectorFieldModel):
    def __init__(
        self,
        in_channels = 2,
        block_nbr: int = 4,
        blck_channels: tuple = (32, 64, 128, 256, 512),
        blck_strides: tuple = (2, 2, 8, 8),
        weight_norm: bool = True,
        t_embed_dim: int = 128,
    ):
        super().__init__()
        """
        Generator of SEANet.

        Args:
            block_nbr (int): number of encoder/decoder blocks
            blck_channels (tuple): number of channels in each encoder/decoder block
            blck_strides (tuple): stride of each encoder/decoder block
            weight_norm (bool): whether to apply weight normalization
        """

        assert (
            block_nbr == len(blck_channels) - 1
        ), "block_nbr must be equal to len(blck_channels) - 1"
        assert block_nbr == len(
            blck_strides
        ), "block_nbr must be equal to len(blck_strides)"

        self.block_nbr = block_nbr
        self.multiple = prod(blck_strides)  # minimal chunk audio length

        self.time_embedder = SinusoidalTimeEmbedding(dim=t_embed_dim, mode='learnable')

        self.first_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=blck_channels[0],
            kernel_size=7,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )

        self.encoder_blocks = nn.ModuleList(
            [
                EncBlock(
                    out_channels=blck_channels[param + 1],
                    stride=blck_strides[param],
                    weight_norm=weight_norm,
                    t_embed_dim=t_embed_dim,
                )
                for param in range(block_nbr)
            ]
        )

        self.latent_conv = nn.Sequential(
            nn.ELU(),
            conv1d_wrapper(
                in_channels=blck_channels[-1],
                out_channels=128,
                kernel_size=7,
                padding="same",
                bias=False,
                padding_mode="reflect",
                weight_norm=weight_norm,
            ),
            nn.ELU(),
            conv1d_wrapper(
                in_channels=128,
                out_channels=blck_channels[-1],
                kernel_size=7,
                padding="same",
                bias=False,
                padding_mode="reflect",
                weight_norm=weight_norm,
            ),
            nn.ELU(),
        )

        self.decoder_blocks = nn.ModuleList(
            [
                DecBlock(
                    out_channels=blck_channels[-param - 2],
                    stride=blck_strides[-param - 1],
                    weight_norm=weight_norm,
                    t_embed_dim=t_embed_dim,
                )
                for param in range(block_nbr)
            ]
        )

        self.last_conv = conv1d_wrapper(
            in_channels=blck_channels[0],
            out_channels=1,
            kernel_size=7,
            padding="same",
            bias=False,
            padding_mode="reflect",
            weight_norm=weight_norm,
        )

    def forward(self, noisy_speech: torch.Tensor, t: torch.Tensor, lr_speech: torch.Tensor):
        """
        Forward pass of generator.

        Args:
            speech (torch.Tensor): corrupted speech signal [B,C,T]
            t (torch.Tensor): time embedding [B,1]
            lr_speech (torch.Tensor): LR waveform [B,C,T]

        Returns:
            (torch.Tensor): enhanced speech signal [B,C,T]
        """
        if lr_speech is not None:
            noisy_speech = torch.cat([noisy_speech, lr_speech], dim=1) # [B,C,T]
        x = self.first_conv(noisy_speech)
        encoder_outputs = [x]

        t_embed = self.time_embedder(t) # [B,t_embed_dim]

        for block in self.encoder_blocks:
            x = block(x , t_embed=t_embed)
            encoder_outputs.append(x)

        x = self.latent_conv(x)

        for idx, block in enumerate(self.decoder_blocks):
            x = block(x, t_embed=t_embed, encoder_output=encoder_outputs[-idx - 1])

        x = x + encoder_outputs[0]
        x = self.last_conv(x)
        
        if lr_speech is not None:
            x = x + lr_speech
        # x = torch.tanh(x + noisy_speech)

        return x

    def cut_tensor(self, tensor):
        """This function is used to make a tensor divisible by the minimal chunk length"""

        old_len = tensor.shape[2]
        new_len = old_len - old_len % self.multiple
        tensor = torch.narrow(tensor, 2, 0, new_len)

        return tensor


class DecBlock(nn.Module):
    def __init__(self, out_channels, stride, weight_norm, t_embed_dim):
        super().__init__()

        self.residuals = nn.Sequential(
            ResidualUnit(channels=out_channels, dilation=1),
            ResidualUnit(channels=out_channels, dilation=3),
            ResidualUnit(channels=out_channels, dilation=9),
        )
        self.conv_trans = conv_trans1d_wrapper(
            in_channels=2 * out_channels,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride - 1,
            output_padding=stride - 2,
            bias=False,
            weight_norm=weight_norm,
        )
        self.time_adapter = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, out_channels * 2)
        )
    
    def forward(self, x, t_embed, encoder_output=None):
        if encoder_output is not None:
            x = x + encoder_output
        t_embed = self.time_adapter(t_embed).unsqueeze(-1) # [B,C,T]
        x = x + t_embed 
        out = self.residuals(self.conv_trans(x))
        return out


class EncBlock(nn.Module):
    def __init__(self, out_channels: int, stride: int, weight_norm: bool, t_embed_dim: int):
        super().__init__()

        self.residuals = nn.Sequential(
            ResidualUnit(channels=out_channels // 2, dilation=1,),
            ResidualUnit(channels=out_channels // 2, dilation=3,),
            ResidualUnit(channels=out_channels // 2, dilation=9,),
        )
        self.conv = conv1d_wrapper(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=2 * stride,
            stride=stride,
            padding=stride - 1,
            bias=False,
            padding_mode="reflect",
            weight_norm=weight_norm,
        )

        self.time_adapter = nn.Sequential(
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, out_channels // 2)
        )   

    def forward(self, x, t_embed):
        """
        Args:
        - x: [B,C,T]
        - t_embed: [B,t_embed_dim]
        
        - out: [B,C,T]
        """
        t_embed =  self.time_adapter(t_embed).unsqueeze(-1) # [B,C,T]
        x = x + t_embed
        x = self.conv(self.residuals(x))
        return x


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int, weight_norm: bool = True, ):
        super().__init__()

        self.conv1 = conv1d_wrapper(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            dilation=dilation,
            padding="same",
            bias=False,
            padding_mode="reflect",
            weight_norm=weight_norm,
        )
        self.conv2 = conv1d_wrapper(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            padding="same",
            bias=False,
            padding_mode="reflect",
            weight_norm=weight_norm,
        )
        self.elu = nn.ELU()

    def forward(self, x):
        res = x.clone()
        x = self.elu(self.conv1(x)) # [B,C,T]
        x = self.elu(self.conv2(x))
        x = x + res        
        # out = x + self.conv2(self.elu(self.conv1(self.elu(x))))
        return x


def conv1d_wrapper(weight_norm: bool, *args, **kwargs):
    """
    Conv1d wrapper with optional weight normalization.

    Args:
        weight_norm (bool): whether to apply weight normalization
        *args: positional arguments of ConvTranspose1d
        **kwargs: keyword arguments of ConvTranspose1d

    Returns:
        (nn.Module): Conv1d with optional weight normalization
    """
    if weight_norm is True:
        return nn.utils.weight_norm(nn.Conv1d(*args, **kwargs))
    else:
        return nn.Conv1d(*args, **kwargs)


def conv_trans1d_wrapper(weight_norm: bool, *args, **kwargs):
    """
    ConvTranspose1d wrapper with optional weight normalization.

    Args:
        weight_norm (bool): whether to apply weight normalization
        *args: positional arguments of ConvTranspose1d
        **kwargs: keyword arguments of ConvTranspose1d

    Returns:
        (nn.Module): ConvTranspose1d with optional weight normalization
    """
    if weight_norm is True:
        return nn.utils.weight_norm(nn.ConvTranspose1d(*args, **kwargs))
    else:
        return nn.ConvTranspose1d(*args, **kwargs)


if __name__ == "__main__":
    audio = torch.randn(1, 1, 16000)  # Batch, Channel, Length

    # Test Seanet forward pass
    # seanet = GeneratorSeanet(blck_channels=(32, 64, 128, 256, 512), weight_norm=True)
    # seanet = GeneratorSeanet(blck_channels=(32, 64, 128, 256, 512), weight_norm=True)
    seanet = GeneratorSeanet(blck_channels=(64, 128, 256, 512, 1024), weight_norm=True)
    
    
    seanet_total_params = sum(p.numel() for p in seanet.parameters() if p.requires_grad)
    print(f"seanet_total_params: {seanet_total_params*1e-6:.1f}M")
    
    time = torch.rand(1,1) # [B,1]
    seanet_in = seanet.cut_tensor(audio) # [2,2,8,8] -> 256
    seanet_out = seanet(seanet_in, time, seanet_in)
    print(f"seanet_in.shape: {seanet_in.shape}")
    print(f"seanet_out.shape: {seanet_out.shape} \n")
    
    from torchinfo import summary
    summary(
        seanet,
        input_data = [seanet_in, time, seanet_in],
        # verbose=1,
        depth=2,
        col_names=['input_size', 'output_size', 'num_params'],
    )

    # # Test Streaming Seanet forward pass
    # streaming_seanet = GeneratorSeanet(
    #     blck_channels=(8, 16, 32, 64, 128), weight_norm=False
    # )
    # streaming_seanet_total_params = sum(
    #     p.numel() for p in streaming_seanet.parameters() if p.requires_grad
    # )
    # print(f"streaming_seanet_total_params: {streaming_seanet_total_params*1e-6:.1f}M")
    # streaming_seanet_in = seanet.cut_tensor(audio)
    # streaming_seanet_out = seanet(streaming_seanet_in)
    # print(f"streaming_seanet_in.shape: {streaming_seanet_in.shape}")
    # print(f"streaming_seanet_out.shape: {streaming_seanet_out.shape}")