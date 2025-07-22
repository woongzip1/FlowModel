import pdb
import torch
import torch.nn as nn
import os
import shutil
import wandb


from torch.optim import lr_scheduler
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from abc import ABC, abstractmethod

from src.flow.path_stft import ConditionalProbabilityPath, ReFlowPath, OriginalCFMPath, DataDependentPriorPath
from src.flow.losses import flow_matching_loss

from src.models.convnext_unet import ConvNeXtUNet, ConditionalVectorFieldModel
from src.utils.spectral_ops import InvertibleFeatureExtractor, AmplitudeCompressedComplexSTFT
from src.utils.logger import BaseLogger
from src.utils.utils import t2n, draw_spec, log_spectral_distance

from src.flow.solver import TorchDiffeqSolver, VectorFieldODE

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
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: nn.Module,
                 val_loader: nn.Module,
                 device: torch.device,
                 logger: BaseLogger,
                 ):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.start_epoch = 1
        self.best_loss = float('inf')
        self.optimizer = None
        self.scheduler = None
        self.logger = logger
        
        
    @abstractmethod
    def _train_step(self, **kwargs) -> torch.Tensor:
        # return train step loss
        pass
    
    # @abstractmethod
    def _val_step(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, config):
        return torch.optim.Adam(self.model.parameters(), **config)

    def get_scheduler(self, optimizer, config):
        """
        """
        scheduler_type = config.get('type', 'CosineLR') 
        scheduler_args = config.get('init_args', {})

        if scheduler_type == 'ExponentialLR':
            print("🚀 *** Using ExponentialLR Scheduler ***")
            scheduler = lr_scheduler.ExponentialLR(optimizer, **scheduler_args)
        elif scheduler_type == 'CosineLR':
            print("🚀 *** Using CosineLR with Warmup Scheduler ***")
            scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, **scheduler_args)
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        return scheduler

    def save_checkpoint(self, epoch: int, is_best: bool, save_dir: str, filename=None):
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
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Save the recent model checkpoint
        if filename:
            recent_ckpt_path = os.path.join(save_dir, filename)
        else:
            recent_ckpt_path = os.path.join(save_dir, 'recent.pth')
        print(f"✅ Recent model saved at epoch {epoch}")
        torch.save(state, recent_ckpt_path)
        
        # If this is the best model, copy the recent checkpoint to best_model.pth
        if is_best:
            best_ckpt_path = os.path.join(save_dir, 'best_model.pth')
            shutil.copyfile(recent_ckpt_path, best_ckpt_path)
            print(f"✅ Best model saved at epoch {epoch} with loss {self.best_loss:.6f}")
            
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
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded successfully.")
        
        print(f"🚀 Checkpoint loaded successfully from {ckpt_path}. Resuming from end of epoch {checkpoint['epoch']}.")
   
    @staticmethod
    def load_model_for_inference(model, ckpt_path, device='cuda'):
        model.to(device)
        print(f"Loading model weights from '{ckpt_path}' for inference...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"Model loaded successfully from {ckpt_path}")
        return model

    def validate(self, global_step):
        """
        Runs a full validation loop, calculates, and returns the average validation loss.
        """
        self.model.eval()  # Set model to evaluation mode
        total_val_loss = 0.0
        total_metric = 0.0
        cnt = 0
        val_pbar = tqdm(self.val_loader, desc=f'🔍 Validating...', dynamic_ncols=True)
        
        with torch.no_grad():  # Disable gradient calculations
            for idx, batch in enumerate(val_pbar):
                outdict = self._val_step(batch, idx)
                loss = outdict['loss']
                log_data = outdict['log_payload']
                metric = outdict['lsd']
                
                total_val_loss += loss.item()
                if metric is not None:
                    total_metric += metric
                    cnt += 1
                val_pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})

                if log_data:
                    self.logger.log(log_data, step=global_step)
                

        avg_val_loss = total_val_loss / len(self.val_loader)
        avg_metric = total_metric / cnt if cnt > 0 else 0
                
        self.logger.log({
            "val/loss": avg_val_loss,
            "val/lsd": avg_metric,
        }, step=global_step)
                
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.train()  
        return {"loss": avg_val_loss, "lsd": avg_metric}

    def train(self, 
                num_epochs: int, 
                max_steps: int = 500000,
                optimizer_config: dict = None,
                scheduler_config: dict = None,
                ckpt_save_dir: str = 'ckpts',
                ckpt_load_path: str = None,
                log_step_interval: int = 100,
                val_step_interval: int = 5000,
                **kwargs):
        """
        Main training loop
        """
        # Report model size
        size_b = model_size_b(self.model)
        print(f'Training model with size: {size_b / MiB:.3f} MiB')
        self.model.to(self.device)
        self.optimizer = self.get_optimizer(optimizer_config)
        
        if scheduler_config:
            self.scheduler = self.get_scheduler(self.optimizer, scheduler_config)
            
        if ckpt_load_path:
            self.load_checkpoint(ckpt_load_path)
        
        # Train loop
        global_step = (self.start_epoch - 1) * len(self.train_loader)
        self.model.train()
        print(f"--- Starting training from epoch {self.start_epoch} ---")
        for epoch_idx in range(self.start_epoch, num_epochs+1):
            epoch_pbar = tqdm(self.train_loader, 
                              desc=f'⚙ Epoch {epoch_idx}/{num_epochs}',
                              dynamic_ncols=True, leave=True)
            total_epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(epoch_pbar):
                global_step += 1
                
                self.optimizer.zero_grad()
                loss = self._train_step(batch)
                loss.backward()
                
                # ---- Gradient logging & clipping ---- 
                # grad_norm = clip_grad_norm(self.model.parameters(), max_norm=)
                # if global_step % log_step_interval == 0:
                #     self.logger.log({"grads/total_grad_norm": grad_norm.item()}, step=global_step)
                    
                self.optimizer.step()
                
                if scheduler_config:
                    self.scheduler.step()
                
                total_epoch_loss += loss.item() 
                epoch_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                                
                # --- step loss logging ---
                if global_step % log_step_interval == 0:
                    self.logger.log({"model/loss": loss.item()}, step=global_step)
                    self.logger.log({"charts/lr-adam": self.optimizer.param_groups[0]['lr']}, step=global_step)
                    
                if global_step % val_step_interval == 0:
                    val_results = self.validate(global_step)
                    avg_val_loss = val_results['loss']
                    avg_val_lsd = val_results['lsd']
                    print(f'\nStep {global_step} | Validation Loss: {avg_val_loss:.6f}, Validation LSD: {avg_val_lsd:.4f}\n')
                
                    # --- Checkpointing based on validation loss
                    is_best = avg_val_loss < self.best_loss
                    if is_best:
                        self.best_loss = avg_val_loss
                    self.save_checkpoint(epoch=epoch_idx, is_best=is_best, save_dir=ckpt_save_dir)

                if global_step >= max_steps: # max step end train
                    print(f'\n🏁 Reached max_steps ({global_step}/{max_steps}). Finishing training.')
                    self.save_checkpoint(epoch=epoch_idx, is_best=False, save_dir=ckpt_save_dir, filename=f'step_{global_step}.pth')
                    return 
                                        
            # --- End of epoch ---
            avg_epoch_loss = total_epoch_loss / len(self.train_loader)
            print(f'Epoch {epoch_idx} completed. Average Loss: {avg_epoch_loss:.6f}')
            
            self.logger.log({
                "model/epoch_loss": avg_epoch_loss,
                "charts/epoch": epoch_idx,
            }, step=global_step)
            
            # # --- Checkpointing ---
            # is_best = avg_epoch_loss < self.best_loss
            # if is_best:
            #     self.best_loss = avg_epoch_loss
            # self.save_checkpoint(epoch=epoch_idx, is_best=is_best, save_dir=ckpt_save_dir)

        # --- Finish ---
        self.model.eval()
        print('✅ Training finished!')
    
class STFTTrainer(Trainer):
    def __init__(self,
                 model: ConditionalVectorFieldModel,
                 path: ConditionalProbabilityPath,
                 transform: InvertibleFeatureExtractor,
                 **kwargs):
        super().__init__(model, **kwargs)
        self.path = path
        self.transform = transform
        self.lr_freq_bins = 80
    
    def _preprocess(self, waveform):
        """ waveform: [B,C,T]
        """
        spec = self.transform(waveform) # [B,C,F,T]
        real = torch.view_as_real(spec.squeeze(1)) # [B,F,T,2]
        real = real.permute(0,3,1,2) # -> [B,2,F,T]
        return real[:,:,:-1,:]
    
    def _postprocess(self, spec): 
        # [B,2,F,T] -> [B,T]
        spec = torch.nn.functional.pad(spec, pad=[0,0,0,1], value=0)
        spec = spec.permute(0,2,3,1).contiguous()
        spec = torch.view_as_complex(spec)
        waveform = self.transform.invert(spec)
        return waveform
    
    # def _train_step(self, batch_data:dict, **kwargs):
    #     """
    #     - batch data
    #     - device
        
    #     Returns:
    #         - torch.Tensor 
    #     """
        
    #     ## Sample z,y from p_data
    #     z = batch_data['hr'].to(self.device) # [B,1,T] 
    #     y = batch_data['lr_wave'].to(self.device)
    #     batch_size = z.shape[0]
        
    #     ## z,y -> Z,Y (transform)
    #     Z = self._preprocess(z)
    #     Y = self._preprocess(y)
        
    #     # pdb.set_trace()
    #     ## Sample t, y, and x
    #     ## Shape of z
    #     t = torch.rand([batch_size, 1, 1, 1], device=z.device) # [B,1,1,1]
    #     x0 = self.path.sample_source(Y)
    #     xt = self.path.sample_xt(x0, Z, Y, t)
        
    #     # output
    #     output = self.model(xt, t, Y)
    #     target = self.path.get_target_vector_field(xt, x0, Z, Y, t)
    #     loss = flow_matching_loss(predicted_vf=output, target_vf=target)
        
    #     return loss
    
    def _train_step(self, batch_data:dict, **kwargs):
        # modified for HR reconstruction
        """
        - batch data
        - device
        
        Returns:
            - torch.Tensor 
        """
        
        ## Sample z,y from p_data
        z = batch_data['hr'].to(self.device) # [B,1,T] 
        y = batch_data['lr_wave'].to(self.device)
        batch_size = z.shape[0]
        
        ## z,y -> Z,Y (transform)
        Z = self._preprocess(z)
        Y = self._preprocess(y)
        Y_lr = Y[:,:,:self.lr_freq_bins,:] #[B,2,F1,T]
        Y_hr = Y[:,:,self.lr_freq_bins:,:]
        Z_lr = Z[:,:,:self.lr_freq_bins,:]
        Z_hr = Z[:,:,self.lr_freq_bins:,:]
        
        ## Sample t, y, and x
        ## Shape of z
        t = torch.rand([batch_size, 1, 1, 1], device=z.device) # [B,1,1,1]
        x0 = self.path.sample_source(Y_hr)
        xt = self.path.sample_xt(x0, Z_hr, Y_lr, t)
        
        # output
        output = self.model(xt, t, Y_lr)
        target = self.path.get_target_vector_field(xt, x0, Z_hr, Y_lr, t)
        loss = flow_matching_loss(predicted_vf=output, target_vf=target)
        
        return loss
    
    # def _val_step(self, batch_data:dict, idx:int, **kwargs):
    #     """
    #     Inference per batch and return validaiton loss
    #     """
    #     ## Sample z,y from p_data
    #     z = batch_data['hr'].to(self.device) # [B,1,T] 
    #     y = batch_data['lr_wave'].to(self.device)
    #     batch_size = z.shape[0]
        
    #     ## z,y -> Z,Y (transform)
    #     Z = self._preprocess(z)
    #     Y = self._preprocess(y)
        
    #     # pdb.set_trace()
    #     ## Sample t, y, and x
    #     ## Shape of z
    #     t = torch.rand([batch_size, 1, 1, 1], device=z.device) # [B,1,1,1]
    #     x0 = self.path.sample_source(Y)
    #     xt = self.path.sample_xt(x0, Z, Y, t)
        
    #     # output
    #     # pdb.set_trace()
    #     output = self.model(xt, t, Y)
    #     target = self.path.get_target_vector_field(xt, x0, Z, Y, t)
    #     loss = flow_matching_loss(predicted_vf=output, target_vf=target)
        
    #     ## --- Metric logging
    #     lsd_metric = None
    #     if idx % 100 == 0:
    #         ode_steps = 20
    #         ts_metric = torch.linspace(0, 1, ode_steps+1, device=self.device)
    #         with torch.no_grad():
    #             ode = VectorFieldODE(net=self.model)
    #             solver = TorchDiffeqSolver(ode, method='euler')
    #             x0_batch = self.path.sample_source(Y)
    #             x1_spec_batch = solver.simulate(x0_batch, ts_metric, y=Y)
    #             x1_wave_batch = self._postprocess(x1_spec_batch)
    #             lsd_metric = log_spectral_distance(z[...,:x1_wave_batch.shape[-1]], x1_wave_batch) 
            
    #     ## --- Sample data logging
    #     log_payload = {}
    #     if idx in [100, 400, 800]:
    #         # print(f"INFO: Generating validation sample for batch index {idx}")
    #         # Use the first item in the batch for logging
    #         z_sample = z[0:1]
    #         y_sample = y[0:1]
    #         Y_sample = Y[0:1]

    #         # ODE Solver
    #         ode = VectorFieldODE(net=self.model)
    #         solver = TorchDiffeqSolver(ode, method='euler')
    #         for num_steps in [1, 10, 20, 30]:
    #             ts = torch.linspace(0, 1, num_steps + 1, device=self.device)

    #             # --- Run Inference ---
    #             x0_sample = self.path.sample_source(Y_sample)
    #             x1_spec = solver.simulate(x0_sample, ts, y=Y_sample)
    #             x1_wave = self._postprocess(x1_spec)

    #             z_sample_c = z_sample[...,:x1_wave.shape[-1]]
    #             y_sample_c = y_sample[...,:x1_wave.shape[-1]]
                
    #             # metric = log_spectral_distance(z_sample, x1_wave)
                
    #             # --- Prepare Log Data ---
    #             spec_gt = draw_spec(t2n(z_sample_c), sr=48000, return_fig=True)
    #             spec_cond = draw_spec(t2n(y_sample_c), sr=48000, return_fig=True)
    #             spec_gen = draw_spec(t2n(x1_wave), sr=48000, return_fig=True)
                
    #             # Create the dictionary to be logged
    #             step_logs = {
    #                 # f"val/{idx}/lsd_{num_steps}_steps": metric.item(),
                    
    #                 f"val_samples/{idx}/{num_steps}/audio_ground_truth": wandb.Audio(t2n(z_sample_c), sample_rate=48000),
    #                 f"val_samples/{idx}/{num_steps}/audio_conditional": wandb.Audio(t2n(y_sample_c), sample_rate=48000),
    #                 f"val_samples/{idx}/{num_steps}/audio_generated": wandb.Audio(t2n(x1_wave), sample_rate=48000),
    #                 f"val_samples/{idx}/{num_steps}/spec_ground_truth": wandb.Image(spec_gt),
    #                 f"val_samples/{idx}/{num_steps}/spec_conditional": wandb.Image(spec_cond),
    #                 f"val_samples/{idx}/{num_steps}/spec_generated": wandb.Image(spec_gen),
    #             }
    #             log_payload.update(step_logs)
                
    #     outdict = {
    #         'loss': loss,
    #         'lsd': lsd_metric,
    #         'log_payload': log_payload,
    #     }
    #     return outdict
    
    def _val_step(self, batch_data:dict, idx:int, **kwargs):
        ## modified
        """
        Inference per batch and return validaiton loss
        """
        ## Sample z,y from p_data
        z = batch_data['hr'].to(self.device) # [B,1,T] 
        y = batch_data['lr_wave'].to(self.device)
        batch_size = z.shape[0]
        
        ## z,y -> Z,Y (transform)
        Z = self._preprocess(z)
        Y = self._preprocess(y)
        Y_lr = Y[:,:,:self.lr_freq_bins,:]
        Y_hr = Y[:,:,self.lr_freq_bins:, :]
        Z_lr = Z[:, :, :self.lr_freq_bins, :]   # Ground-truth LR part for reconstruction
        Z_hr = Z[:, :, self.lr_freq_bins:, :]   # Ground-truth HR part for validation loss
        
        # pdb.set_trace()
        ## Sample t, y, and x
        ## Shape of z
        t = torch.rand([batch_size, 1, 1, 1], device=z.device) # [B,1,1,1]
        x0 = self.path.sample_source(Y_hr)
        xt = self.path.sample_xt(x0, Z_hr, Y_lr, t)
        
        # output
        # pdb.set_trace()
        output = self.model(xt, t, Y_lr)
        target = self.path.get_target_vector_field(xt, x0, Z_hr, Y_lr, t)
        loss = flow_matching_loss(predicted_vf=output, target_vf=target)
        
        ## --- Metric logging
        lsd_metric = None
        if idx % 100 == 0:
            ode_steps = 20
            ts_metric = torch.linspace(0, 1, ode_steps+1, device=self.device)
            with torch.no_grad():
                ode = VectorFieldODE(net=self.model)
                solver = TorchDiffeqSolver(ode, method='euler')
                x0_batch = self.path.sample_source(Y_hr)
                x1_spec_batch = solver.simulate(x0_batch, ts_metric, y=Y_lr)
                x1_spec_batch = torch.cat([Y_lr, x1_spec_batch], dim=2)
                x1_wave_batch = self._postprocess(x1_spec_batch)
                lsd_metric = log_spectral_distance(z[...,:x1_wave_batch.shape[-1]], x1_wave_batch) 
            
        ## --- Sample data logging
        log_payload = {}
        if idx in [100, 400, 800]:
            # print(f"INFO: Generating validation sample for batch index {idx}")
            # Use the first item in the batch for logging
            z_sample = z[0:1]
            y_sample = y[0:1]
            Y_sample = Y[0:1]

            # ODE Solver
            ode = VectorFieldODE(net=self.model)
            solver = TorchDiffeqSolver(ode, method='euler')
            for num_steps in [1, 10, 20, 30]:
                ts = torch.linspace(0, 1, num_steps + 1, device=self.device)

                # --- Run Inference ---
                Y_hr = Y_sample[...,self.lr_freq_bins:,:]
                Y_lr = Y_sample[...,:self.lr_freq_bins,:]
                
                x0_sample = self.path.sample_source(Y_hr)
                x1_spec = solver.simulate(x0_sample, ts, y=Y_lr)
                x1_spec = torch.cat([Y_lr, x1_spec], dim=2)
                x1_wave = self._postprocess(x1_spec)

                z_sample_c = z_sample[...,:x1_wave.shape[-1]]
                y_sample_c = y_sample[...,:x1_wave.shape[-1]]
                
                # metric = log_spectral_distance(z_sample, x1_wave)
                
                # --- Prepare Log Data ---
                spec_gt = draw_spec(t2n(z_sample_c), sr=48000, return_fig=True)
                spec_cond = draw_spec(t2n(y_sample_c), sr=48000, return_fig=True)
                spec_gen = draw_spec(t2n(x1_wave), sr=48000, return_fig=True)
                
                # Create the dictionary to be logged
                step_logs = {
                    # f"val/{idx}/lsd_{num_steps}_steps": metric.item(),
                    
                    f"val_samples/{idx}/{num_steps}/audio_ground_truth": wandb.Audio(t2n(z_sample_c), sample_rate=48000),
                    f"val_samples/{idx}/{num_steps}/audio_conditional": wandb.Audio(t2n(y_sample_c), sample_rate=48000),
                    f"val_samples/{idx}/{num_steps}/audio_generated": wandb.Audio(t2n(x1_wave), sample_rate=48000),
                    f"val_samples/{idx}/{num_steps}/spec_ground_truth": wandb.Image(spec_gt),
                    f"val_samples/{idx}/{num_steps}/spec_conditional": wandb.Image(spec_cond),
                    f"val_samples/{idx}/{num_steps}/spec_generated": wandb.Image(spec_gen),
                }
                log_payload.update(step_logs)
                
        outdict = {
            'loss': loss,
            'lsd': lsd_metric,
            'log_payload': log_payload,
        }
        return outdict
    
    

def main():
    from torchinfo import summary
    from src.utils.utils import load_config
    from data.dataset import make_dataset, prepare_dataloader
    
    config_path = 'configs/config_template.yaml'
    config = load_config(config_path)
    train_loader, val = prepare_dataloader(config)

    transform = AmplitudeCompressedComplexSTFT(
                                        window_fn='hann', n_fft=1024, 
                                        sampling_rate=48000, hop_length=256,
                                        alpha=0.3, beta=1, comp_eps=1e-4,)
    
    path = OriginalCFMPath()
    model = ConvNeXtUNet(in_channels=4, out_channels=2, dims=[64,128,256,512], depths=[2,2,4,2])
    summary(
        model,
        input_data=[torch.randn(1,2,512,100), torch.randn(4), torch.randn(1,2,512,100)],
        depth=4,
        col_names=["input_size", "output_size", "num_params"],
        verbose=1
    )
    
    trainer = STFTTrainer(
                        path=path,
                        model=model,
                        train_loader=train_loader,
                        val_loader=train_loader,
                        transform=transform,
                        device='cuda',
                        )
    trainer.train(num_epochs=1)

if __name__=="__main__":
    main()
    
"""
WANDB logger
learning-rate

model/
grads/
validation metrics

train/in_audio
val/in_audio

charts/lr-adam
charts/epoch


"""

    