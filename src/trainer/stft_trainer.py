import pdb
import torch
import torch.nn as nn
import os
import shutil

from tqdm import tqdm
from abc import ABC, abstractmethod
from src.flow.path import GaussianConditionalProbabilityPath, DataLoaderConditionalProbabilityPath, LinearAlpha, LinearBeta
from src.flow.losses import flow_matching_loss
from src.flow.path import ConditionalVectorFieldModel

from src.models.seanet import GeneratorSeanet


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
    def __init__(self, model: nn.Module, dataloader: nn.Module):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        
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
    
class STFTTrainer(Trainer):
    def __init__(self,
                 path: DataLoaderConditionalProbabilityPath,
                 model: ConditionalVectorFieldModel,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.path = path
        self.optimizer = None
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.device = None
        
    def save_checkpoint(self, epoch: int, is_best: bool, save_dir: str):
        """
        Saves a checkpoint of the model and optimizer.

        Args:
            epoch (int): The current epoch number.
            is_best (bool): True if this checkpoint has the best validation loss so far.
            save_dir (str): The directory to save the checkpoint in.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }
        
        # Save the recent model checkpoint
        recent_ckpt_path = os.path.join(save_dir, 'recent.pth')
        torch.save(state, recent_ckpt_path)
        
        # If this is the best model, copy the recent checkpoint to best_model.pth
        if is_best:
            best_ckpt_path = os.path.join(save_dir, 'best_model.pth')
            shutil.copyfile(recent_ckpt_path, best_ckpt_path)
            print(f"✅ Best model saved at epoch {epoch+1} with loss {self.best_loss:.6f}")
    
    def load_checkpoint(self, ckpt_path: str):
        """
        Loads a checkpoint to resume training.

        Args:
            ckpt_path (str): Path to the checkpoint file.
        """
        if not os.path.isfile(ckpt_path):
            print(f"⚠️ Checkpoint not found at {ckpt_path}. Starting from scratch.")
            return

        # Ensure the optimizer is initialized before loading its state
        if self.optimizer is None:
            raise RuntimeError("Optimizer must be initialized before loading a checkpoint.")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint.get('best_loss', float('inf')) # Handle older checkpoints
        
        print(f"Checkpoint loaded successfully from {ckpt_path}. Resuming from end of epoch {self.start_epoch}.")
        
    @staticmethod
    def load_model_params(model, ckpt_path, device='cuda'):
        model = model.to(device)
        print(f"Loading '{ckpt_path}...'")
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Model loaded from {ckpt_path}")
        return model
    
    def train(self, num_epochs: int, device: torch.device, lr: float = 1e-3, 
              ckpt_save_dir: str = 'checkpoints', ckpt_load_path: str = None,
              **kwargs) -> None:
        """
        Main training loop
        """
        self.device = device
        
        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / MiB:.3f} MiB')

        # Set model and optimizers
        self.model.to(device)
        self.optimizer = self.get_optimizer(lr)
        
        # Load ckpt
        if ckpt_load_path:
            self.load_checkpoint(ckpt_load_path)
        self.model.train()
        print(f"--- Starting training from epoch {self.start_epoch + 1} ---")

        # Train loop
        for epoch_idx in range(self.start_epoch, num_epochs+1):
            epoch_pbar = tqdm(self.dataloader, desc=f'⚙ Epoch {epoch_idx+1}/{num_epochs}',
                              dynamic_ncols=True, leave=True)
            total_epoch_loss = 0.0
            
            for batch_idx, batch_data in enumerate(epoch_pbar):    
                self.optimizer.zero_grad()
                loss = self.get_train_loss(batch_data, device)
                loss.backward()
                self.optimizer.step()
                
                current_loss = loss.item()
                total_epoch_loss += current_loss
                
                epoch_pbar.set_postfix({'loss': f'{current_loss:.6f}'})
            
            # --- End of Epoch ---
            avg_epoch_loss = total_epoch_loss / len(self.dataloader)
            print(f'Epoch {epoch_idx} completed. Average Loss: {avg_epoch_loss:.6f}')
            
            # Check if the current model is the best
            is_best = avg_epoch_loss < self.best_loss
            if is_best:
                self.best_loss = avg_epoch_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch=epoch_idx, is_best=is_best, save_dir=ckpt_save_dir)

            ## save_epoch_loss()
        
        # --- Finish ---
        self.model.eval()
        print('Training finished')
    
    def get_train_loss(self, batch_data:dict, device:torch.device, **kwargs):
        """
        - batch data
        - device
        
        Returns:
            - torch.Tensor 
        """
        
        ## Sample z,y from p_data
        z = batch_data['hr'].to(device)
        y = batch_data['lr_wave'].to(device)
        batch_size = z.shape[0]
        
        ## Sample t, y, and x
        ## Shape of z
        t = torch.rand([batch_size, 1, 1], device=z.device)
        x, x0 = self.path.sample_conditional_path(z, t, y)
        
        # output
        output = self.model(x, t, y)
        target = self.path.conditional_vector_field(z, x0)
        loss = flow_matching_loss(predicted_vf=output, target_vf=target)
        
        return loss
    

def main():
    from src.utils.utils import load_config
    from data.dataset import make_dataset, prepare_dataloader
    
    config_path = 'configs/config_template.yaml'
    config = load_config(config_path)
    train_loader, val = prepare_dataloader(config)

    alpha = LinearAlpha()
    beta = LinearBeta()
    gaussian_path = DataLoaderConditionalProbabilityPath(
                        p_simple_shape=[16,1,38400], # [C,T] ?
                        alpha=alpha,
                        beta=beta,
    )
    model = GeneratorSeanet(blck_channels=(64, 128, 256, 512, 1024), weight_norm=True)
    trainer = STFTTrainer(
                        path=gaussian_path,
                        model=model,
                        dataloader=train_loader,
                          )
    trainer.train(1, device='cuda')

if __name__=="__main__":
    main()