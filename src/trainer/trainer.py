import torch
import torch.nn as nn

from tqdm import tqdm
from abc import ABC, abstractmethod
from src.flow.path import GaussianConditionalProbabilityPath
from src.flow.losses import flow_matching_loss
from src.models.unet import ConditionalVectorField

## path, unet, loss
MiB = 1024 ** 2

def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size

class Trainer(ABC):
    """
    Abstract base class for Trainer 
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs) -> torch.Tensor:
        """
        Main training loop
        """
        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / MiB:.3f} MiB')
        
        # Set model and optimizers
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f'Epoch {idx}, loss: {loss.item():.3f}')

        # Finish
        self.model.eval()
        
class CFGTrainer(Trainer):
    """
    Concrete class that trains model with classifier free guidance
    """
    def __init__(self, 
                 path: GaussianConditionalProbabilityPath, 
                 model: ConditionalVectorField, 
                 eta: float, **kwargs):
        assert eta > 0 and eta < 1
        super().__init__(model, **kwargs)
        self.eta = eta
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        
        # Step 1: Sample z,y from p_data
        z, y = self.path.p_data.sample(batch_size)
        """
        z - [B, C, H, W]
        y - [B, LabelDim=1]
        """
                
        # Step 2: Set each label to 10 (i.e., null) with probability eta
        eps = torch.rand(batch_size, device=z.device) # eps ~ U[0,1] / thres eta (0.1)
        y[eps < self.eta] = 10

        # Step 3: Sample t and x
        t = torch.rand(batch_size,1,1,1, device=z.device) # [B,1,1,1]
        x = self.path.sample_conditional_path(z, t) # [B,C,H,W] : conditional path

        # Step 4: Regress and output loss
        output = self.model(x, t, y)
        target = self.path.conditional_vector_field(x, z, t)          # [B,C,H,W]        
        loss = flow_matching_loss(predicted_vf=output, target_vf=target)
        return loss        

def main():
    pass

if __name__=="__main__":
    main()