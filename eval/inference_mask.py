### function that forwards and save functions
### model forward after N-steps and save audio files

import pdb
import os
import torch
import torchaudio
import numpy as np
import argparse
import time
import soundfile as sf

from tqdm import tqdm
from IPython.display import Audio, display

# --- Assumed project structure for imports ---
from data.dataset_idx import prepare_dataloader 
from src.models.seanet import GeneratorSeanet 
from src.flow.path import DataLoaderConditionalProbabilityPath, LinearAlpha, LinearBeta, LinearSigmaBeta
from src.trainer.trainer import WaveTrainer

from src.utils.utils import print_config, draw_2d_heatmap, draw_spec, t2n
from train import load_config

### Forward ODE
from torch import Tensor
from src.flow.solver import ODE, EulerSolver, VectorFieldODE, CFGVectorFieldODE, TorchDiffeqSolver
from src.flow.path import ConditionalVectorFieldModel
from src.flow.path_stft import get_path
from src.models.unetv3 import ConvNeXtUNetFiLM

from src.trainer.stft_trainer_mask import STFTTrainerMask
from src.utils.utils import plot_signals
from src.utils.logger import get_logger
from src.utils.spectral_ops import AmplitudeCompressedComplexSTFT

## models
## dataset
## main utils

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_model_params(model, checkpoint_path, device='cuda'):
    model = model.to(device)
    print(f"Loading '{checkpoint_path}...'")
    ckpt = torch.load(checkpoint_path)
    # import pdb
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    print(f"Model loaded from {checkpoint_path}")
    return model

def setup_model_and_solver(config, ckpt_path, method='euler', device='cuda', guide=1.5):
    """
    Initializes the model, ODE solver, and probability path.
    """
    # Load the model
    model = ConvNeXtUNetFiLM(**config.model).to(device)
    model = STFTTrainerMask.load_model_for_inference(model, ckpt_path, device=device)
    model.eval()

    # Path
    path = get_path(config.path).to(device)
    # ode = VectorFieldODE(net=model)
    ode = CFGVectorFieldODE(net=model, guidance_scale=guide)
    solver = TorchDiffeqSolver(ode, method=method)
    
    # Trainer
    transform = AmplitudeCompressedComplexSTFT(**config.transform)
    trainer = STFTTrainerMask(
                        path=path,
                        model=model,
                        logger=get_logger(config),
                        train_loader=None,
                        val_loader=None,
                        transform=transform,
                        device='cuda',
    )
    
    return model, solver, path, trainer

def run_ode_inference(solver, trainer, lr_audio, num_timesteps=50, sr_value=8, device='cuda'):
    """
    Runs the ODE simulation for a single low-resolution audio input.
    lr_audio: Low-resolution audio tensor, shape [1, 1, L_lr]
    """
    # Ensure input is correctly shaped [Batch, Channel, Length]
    if lr_audio.dim() == 2:
        lr_audio = lr_audio.unsqueeze(0) # Add batch dimension if missing
    
    # Condition
    condition_y = lr_audio.to(device)
    Y = trainer._preprocess(waveform=condition_y)
    
    ## ---- Additional processing for LR concat
    lr_mask, hr_mask = trainer._make_mask(torch.tensor([sr_value]), Y.shape[2], device)
    Y_lr = lr_mask * Y
    Y_hr = hr_mask * Y
    
    x0_sample = trainer.path.sample_source(Y_lr, hr_mask).to(device)
    ts = torch.linspace(0,1,num_timesteps+1)
    with torch.no_grad():
        x1_spec = solver.simulate(x0_sample, ts=ts, y=Y_lr, sr_values=torch.tensor([sr_value]))
    x1_spec = Y_lr + hr_mask * x1_spec
    generated_audio = trainer._postprocess(x1_spec)
    # Return the final generated audio, remove batch and channel dims for saving
    return generated_audio.squeeze(0).squeeze(0).cpu()

def inference_single_file(config, ckpt_path, idx_to_test=0, exp_name='ode_results', 
                          method='euler', num_timesteps=50, sr_value=8, guide=1.5, device='cuda'):
    """
    Runs inference on a single file from the dataset for quick testing and debugging.
    """
    print(f"--- Running inference for single file (index: {idx_to_test}) ---")
    
    # Setup
    _config = config
    _config.dataset.common.sampling_rates = [sr_value]
    model, solver, path, trainer = setup_model_and_solver(config, ckpt_path, method, device, guide)
    _, val_loader = prepare_dataloader(_config)
    val_dataset = val_loader.dataset
    
    # Get a single data point
    outdict= val_dataset[idx_to_test]
    lr_wave = outdict['lr_wave']
    hr_wave = outdict['hr'] 
    name = outdict['filename']

    # Run inference
    print(f"Generating audio for: {name}")
    start_time = time.time()
    generated_wave = run_ode_inference(solver, trainer, lr_wave, num_timesteps, sr_value, device)
    duration = time.time() - start_time
    print(f"Inference took {duration:.2f} seconds.")

    lr_wave = lr_wave[...,:generated_wave.shape[-1]]
    hr_wave = hr_wave[...,:generated_wave.shape[-1]]
    save_dir = os.path.join(config['inference']['dir_speech'], exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f'{save_dir}/lr', exist_ok=True)
    os.makedirs(f'{save_dir}/gen', exist_ok=True)
    os.makedirs(f'{save_dir}/gt', exist_ok=True)
    
    # Save lr file
    lr_file = os.path.join(save_dir, "lr",f"{name}.wav")
    sf.write(lr_file, lr_wave.squeeze().numpy(), config['dataset']['common']['sr'], 'PCM_16')
    print(f"Saved lr audio to: {lr_file}")

    # Save the generated file
    output_file = os.path.join(save_dir, "gen", f"{name}.wav")
    sf.write(output_file, generated_wave.numpy(), config['dataset']['common']['sr'], 'PCM_16')
    print(f"Saved generated audio to: {output_file}")

    # Save ground truth for comparison
    gt_file = os.path.join(save_dir, "gt", f"{name}.wav")
    sf.write(gt_file, hr_wave.squeeze().numpy(), config['dataset']['common']['sr'], 'PCM_16')
    print(f"Saved ground truth audio to: {gt_file}")
    
def inference_full_dataset(config, ckpt_path, exp_name='ode_results', method='euler', num_timesteps=50, device='cuda'):
    """
    Runs inference on the entire validation set and saves the output files.
    This is equivalent to your original GAN inference script.
    """
    print(f"--- Running inference for full dataset ---")
    set_seed(1234)

    # Setup
    model, solver, gaussian_path = setup_model_and_solver(config, ckpt_path, method, device)
    _, val_loader = prepare_dataloader(config)

    # Prepare save directories
    save_base_dir = os.path.join(config['inference']['dir_speech'], exp_name)
    save_gen_dir = os.path.join(save_base_dir, 'generated')
    save_lr_dir = os.path.join(save_base_dir, 'lr')
    save_gt_dir = os.path.join(save_base_dir, 'gt')
    os.makedirs(save_gen_dir, exist_ok=True)
    os.makedirs(save_gt_dir, exist_ok=True)
    
    total_duration = 0
    
    for outdict in tqdm(val_loader):
        lr_wave = outdict['lr_wave']
        hr_wave = outdict['hr'] 
        name = outdict['file_name']
        # The name is usually a tuple, so get the first element
        filename = name[0]

        # Run inference
        start_time = time.time()
        generated_wave = run_ode_inference(solver, gaussian_path, lr_wave, num_timesteps, device)
        total_duration += time.time() - start_time

        # Save lr file
        lr_file = os.path.join(save_lr_dir, f"{filename}.wav")
        sf.write(lr_file, lr_wave.numpy(), config['dataset']['common']['sr'])
        
        # Save generated file
        output_file = os.path.join(save_gen_dir, f"{filename}.wav")
        sf.write(output_file, generated_wave.numpy(), config['dataset']['common']['sr'])
        
        # Save ground truth file
        gt_file = os.path.join(save_gt_dir, f"{filename}.wav")
        sf.write(gt_file, hr_wave.squeeze().numpy(), config['dataset']['common']['sr'])

    avg_duration = total_duration / len(val_loader)
    print("\n--- Inference Complete ---")
    print(f"Saved {len(val_loader)} files to: {save_gen_dir}")
    print(f"Average inference time per file: {avg_duration:.2f} seconds.")

def main():
    print("Initializing Inference Process...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--exp_name", type=str, default='')
    parser.add_argument("--method", type=str, default='midpoint')
    parser.add_argument("--save_gt", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--log_term", type=int, default=10)
    parser.add_argument("--sr", type=int, default=8)
    parser.add_argument("--guide", type=float, default=0.0)
    
    
    args = parser.parse_args()

    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = 'cuda'
    config = load_config(args.config)
    # ckpt_path = "./ckpts/best_model.pth"
    ckpt_path = os.path.join(config.train.ckpt_save_dir, "recent.pth")
    
    for ode_steps in [30]:
        # for idx_to_test in np.arange (0, 2000, 100):
        for idx_to_test in np.arange(0, 800+1, 80):
            inference_single_file(config, ckpt_path, idx_to_test=idx_to_test, 
                                  method=args.method, num_timesteps=ode_steps, 
                                  device=DEVICE, exp_name=os.path.join(f'{args.sr}khz', f'{args.guide}', args.method,f'{ode_steps}'),
                                  guide=args.guide, sr_value=args.sr)
    # inference(config, device=args.device, save_gt=args.save_gt, exp_name=args.exp_name, log_term=args.log_term)

if __name__ == "__main__":
    main()
    
"""
TODO
Inference single file 만 처리함
full dataset 코드는 만지지 않음
"""

# python -m eval.inference --config configs/1-2cfm.yaml --method euler
# python -m eval.inference --config configs/1-3d.yaml --method euler
