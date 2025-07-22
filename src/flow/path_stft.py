## regarding conditional probability path

## (Alpha, Beta) or Gamma
## ConditionalProbabilityPath

import torch
import torch.nn as nn
import importlib

from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

class ConditionalProbabilityPath(nn.Module, ABC):
    """
    Abstract base class for conditional probability paths
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample_source(self, Z: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Returns x0 in the same format as Z/Y."""

    @abstractmethod
    def sample_xt(self, x0: torch.Tensor, Z: torch.Tensor, Y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns x_t = f(x0, x1=Z, t)."""

    @abstractmethod
    def get_target_vector_field(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        Z: torch.Tensor,
        Y: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Returns u_t(xt|Z,Y)."""

class OriginalCFMPath(ConditionalProbabilityPath):
    def __init__(self, sigma_min:float=1e-4):
        super().__init__()
        self.sigma_min = sigma_min
        
    def sample_source(self, Y):
        # Standard Gaussian
        # x0 ~ N(0,1)
        return torch.randn_like(Y)

    def sample_xt(self, x0, Z, Y, t):
        return t*Z + (1 - t + self.sigma_min*t) * x0

    def get_target_vector_field(self, xt, x0, Z, Y, t):
        return Z - (1 - self.sigma_min) * x0
    
class ReFlowPath(ConditionalProbabilityPath):
    def __init__(self):
        super().__init__()
        
    def sample_source(self, Y):
        # Identical to Y
        return Y

    def sample_xt(self, x0, Z, Y, t):
        return t*Z + (1-t)*x0

    def get_target_vector_field(self, xt, x0, Z, Y, t):
        return Z - x0
    
class DataDependentPriorPath(ConditionalProbabilityPath):
    def __init__(self,
        sampling_rate: int,
        num_freq_bins: int=512,
        alpha: float = 3e-4,
        f_c: float = 4000,
        sigma_max: float = 0.5,
        sigma_min: float = 0.0,
    ):
        super().__init__()
        self.sr = sampling_rate
        self.alpha = alpha
        self.f_c = f_c
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

        # frequency dependent mask
        F = num_freq_bins # consider this is unvalid
        freqs = torch.linspace(0, self.sr/2, F)
        m_k = torch.sigmoid(self.alpha * (freqs - self.f_c))
        sigma_per_freq = m_k * self.sigma_max
        
        # Masks
        # preserve_mask = 1.0 - m_k
        # self.register_buffer("preserve_mask", preserve_mask.view(1, 1, -1, 1)) # [1,1,F,1]
        self.register_buffer("sigma_per_freq", sigma_per_freq.view(1, 1, -1, 1)) # [1,1,F,1]
        
            
    def sample_source(self, Y):
        # x_low_freq = Y * self.preserve_mask
        x_low_freq = Y # Y is LR spec 
        noise = torch.randn_like(Y) * self.sigma_per_freq.to(Y.device)
        x0 = x_low_freq + noise
        return x0
    
    def sample_xt(self, x0, Z, Y, t):
        return t * Z + (1 - t + self.sigma_min*t) * x0
    
    def get_target_vector_field(self, xt: torch.Tensor, x0: torch.Tensor, Z: torch.Tensor, Y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return Z - (1 - self.sigma_min) * x0

def get_path(config):
    class_path = config.get("class_path")
    
    if not class_path:
        raise ValueError("Configuration must contain a 'class_path' key")
    try:
        module_path, class_name = class_path.rsplit(".", 1)
    except ValueError:
        raise ValueError(f"Invalid class_path '{class_path}'. Must contain at least one")
    
    module = importlib.import_module(module_path)
    Class = getattr(module, class_name)
    init_args = config.get("init_args", {})
    return Class(**init_args)
    
    