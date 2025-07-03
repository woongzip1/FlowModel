### function that forwards and save functions
### model forward after N-steps and save audio files

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
from data.dataset import prepare_dataloader 
from src.models.seanet import GeneratorSeanet 
from src.flow.path import DataLoaderConditionalProbabilityPath, LinearAlpha, LinearBeta, LinearSigmaBeta
from src.trainer.trainer import WaveTrainer 
from src.utils.utils import print_config, draw_2d_heatmap, draw_spec, t2n
from train import load_config

### Forward ODE
from torch import Tensor
from src.flow.solver import ODE, EulerSolver, VectorFieldODE
from src.flow.path import ConditionalVectorFieldModel
from src.models.seanet import GeneratorSeanet
from src.trainer.trainer import WaveTrainer
from src.utils.utils import plot_signals

## models
## dataset
## main utils

def setup_model_and_solver(config, ckpt_path, device):
    """
    Initializes the model, ODE solver, and probability path.
    """
    # Load the model
    model = GeneratorSeanet(**config['model']).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    model.eval()

    # Setup the probability path for defining the ODE
    alpha = LinearAlpha()
    beta = LinearBeta()
    gaussian_path = DataLoaderConditionalProbabilityPath(
        p_simple_shape=[1, config['audio']['segment_length']], # Use segment length from config
        alpha=alpha,
        beta=beta,
    ).to(device)

    # Setup the ODE and Solver
    ode = VectorFieldODE(net=model)
    # If using Classifier-Free Guidance:
    # ode = CFGVectorFieldODE(net=model, guidance_scale=3.0)
    solver = EulerSolver(ode)
    
    return model, solver, gaussian_path

def run_ode_inference(solver, gaussian_path, lr_audio, num_timesteps=50, device='cuda'):
    """
    Runs the ODE simulation for a single low-resolution audio input.
    lr_audio: Low-resolution audio tensor, shape [1, 1, L_lr]
    """
    # Ensure input is correctly shaped [Batch, Channel, Length]
    if lr_audio.dim() == 2:
        lr_audio = lr_audio.unsqueeze(0) # Add batch dimension if missing
    
    condition_y = lr_audio.to(device)

    # 1. Sample initial noise from the prior distribution (Gaussian)
    # The shape should match the target high-resolution audio.
    # Here we assume the target length is known from config.
    # Note: If your model upsamples, the noise shape should match the final output shape.
    x0 = gaussian_path.p_simple.sample(1).to(device)

    # 2. Define the integration timesteps from t=0 to t=1
    ts = torch.linspace(0, 1, num_timesteps).view(1, -1, 1, 1).expand(x0.shape[0], -1, 1, 1).to(device)

    # 3. Solve the ODE
    with torch.no_grad():
        generated_audio = solver.simulate(x0, ts=ts, y=condition_y) # x1 is the result at t=1

    # Return the final generated audio, remove batch and channel dims for saving
    return generated_audio.squeeze(0).squeeze(0).cpu()


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_model_params(model, checkpoint_path, device='cuda'):
    model = model.to(device)
    print(f"Loading '{checkpoint_path}...'")
    ckpt = torch.load(checkpoint_path)
    # import pdb
    model.load_state_dict(ckpt['generator'], strict=False)
    
    print(f"Model loaded from {checkpoint_path}")
    return model

def inference_single_file(config, ckpt_path, idx_to_test=0, num_timesteps=50, device='cuda'):
    """
    Runs inference on a single file from the dataset for quick testing and debugging.
    """
    print(f"--- Running inference for single file (index: {idx_to_test}) ---")
    
    # Setup
    model, solver, gaussian_path = setup_model_and_solver(config, ckpt_path, device)
    val_dataset, _ = prepare_dataloader(config, get_dataset_only=True) # Get dataset object
    
    # Get a single data point
    outdict= val_dataset[idx_to_test]
    lr_wave = outdict['lr_wave']
    hr_wave = outdict['hr'] 
    name = outdict['file_name']

    # Run inference
    print(f"Generating audio for: {name}")
    start_time = time.time()
    generated_wave = run_ode_inference(solver, gaussian_path, lr_wave, num_timesteps, device)
    duration = time.time() - start_time
    print(f"Inference took {duration:.2f} seconds.")

    # Save the generated file
    save_dir = os.path.join(config['inference']['dir_speech'], 'single_test')
    os.makedirs(save_dir, exist_ok=True)
    
    output_file = os.path.join(save_dir, f"{name}_generated.wav")
    sf.write(output_file, generated_wave.numpy(), config['audio']['sampling_rate'])
    print(f"Saved generated audio to: {output_file}")

    # Save ground truth for comparison
    gt_file = os.path.join(save_dir, f"{name}_ground_truth.wav")
    sf.write(gt_file, hr_wave.squeeze().numpy(), config['audio']['sampling_rate'])
    print(f"Saved ground truth audio to: {gt_file}")
    
def inference_full_dataset(config, ckpt_path, exp_name='ode_results', num_timesteps=50, device='cuda'):
    """
    Runs inference on the entire validation set and saves the output files.
    This is equivalent to your original GAN inference script.
    """
    print(f"--- Running inference for full dataset ---")
    set_seed(1234)

    # Setup
    model, solver, gaussian_path = setup_model_and_solver(config, ckpt_path, device)
    _, val_loader = prepare_dataloader(config)

    # Prepare save directories
    save_base_dir = os.path.join(config['inference']['dir_speech'], exp_name)
    save_gen_dir = os.path.join(save_base_dir, 'generated')
    save_gt_dir = os.path.join(save_base_dir, 'ground_truth')
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

        # Save generated file
        output_file = os.path.join(save_gen_dir, f"{filename}.wav")
        sf.write(output_file, generated_wave.numpy(), config['audio']['sampling_rate'])
        
        # Save ground truth file
        gt_file = os.path.join(save_gt_dir, f"{filename}.wav")
        sf.write(gt_file, hr_wave.squeeze().numpy(), config['audio']['sampling_rate'])

    avg_duration = total_duration / len(val_loader)
    print("\n--- Inference Complete ---")
    print(f"Saved {len(val_loader)} files to: {save_gen_dir}")
    print(f"Average inference time per file: {avg_duration:.2f} seconds.")
    

# def inference(config, device='cuda', save_gt=False, exp_name='', log_term=10,):
#     # speech / audio
#     save_base_dir = os.path.join(config['inference']['dir_speech'], exp_name)
#     # save_base_dir = os.path.join(config['inference']['dir_audio'], exp_name)
#     os.makedirs(save_base_dir, exist_ok=True)

#     # dataloader
#     _, val_loader = prepare_dataloader(config)
#     # model
#     config_path = "./ckpts/best_model.pth"
#     model = GeneratorSeanet(**config.model)
#     model = WaveTrainer.load_model_params(model, config_path)
    
#     alpha = LinearAlpha()
#     beta = LinearBeta()
    
#     gaussian_path = DataLoaderConditionalProbabilityPath(
#                         p_simple_shape=[1,40192],
#                         # p_simple_shape=[1,38400],
#                         alpha=alpha,
#                         beta=beta,
#     ).to(device)
    
#     x0 = gaussian_path.p_simple.sample(1)
#     y = outdict['lr_wave'].unsqueeze(0).to(device)
#     z = outdict['hr'].unsqueeze(0)

#     set_seed()
#     ## forward
#     model.eval()
#     bar = tqdm(val_loader)
#     duration_tot = 0
#     with torch.no_grad():
#         for idx, batch in enumerate(bar):
#             waveform, name = batch[0].to(device), batch[1]

#             if idx % log_term == 0: # 10
            
#                 # forward
#                 pred_start = time.time() # tick
#                 audio_gen, _, _ = _forward_pass(waveform, model, config)
#                 duration_tot += time.time() - pred_start # tock
                
#                 # save
#                 output_file = os.path.join(save_base_dir, name[0]+'.wav')
#                 sf.write(output_file, audio_gen[0].squeeze().cpu().numpy(), 24000, 'PCM_16')

#                 if save_gt:
#                     gt_file = os.path.join(save_base_dir, 'gt', name[0]+'.wav')
#                     os.makedirs(os.path.dirname(gt_file), exist_ok=True)
#                     sf.write(gt_file, waveform[0].squeeze().cpu().numpy(), 24000, 'PCM_16')

def main():
    print("Initializing Inference Process...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--exp_name", type=str, default='')
    parser.add_argument("--save_gt", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--log_term", type=int, default=10)
    
    args = parser.parse_args()

    ckpt_path = "./ckpts/best_model.pth"
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = 'cuda'
    config = load_config(args.config)
    inference_single_file(config, ckpt_path, idx_to_test=10, num_timesteps=50, device=DEVICE)
    # inference(config, device=args.device, save_gt=args.save_gt, exp_name=args.exp_name, log_term=args.log_term)

if __name__ == "__main__":
    main()