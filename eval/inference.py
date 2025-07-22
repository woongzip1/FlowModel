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
from data.dataset import prepare_dataloader 
from src.models.seanet import GeneratorSeanet 
from src.flow.path import DataLoaderConditionalProbabilityPath, LinearAlpha, LinearBeta, LinearSigmaBeta
from src.trainer.trainer import WaveTrainer
from src.trainer.stft_trainer import STFTTrainer 

from src.utils.utils import print_config, draw_2d_heatmap, draw_spec, t2n
from train import load_config

### Forward ODE
from torch import Tensor
from src.flow.solver import ODE, EulerSolver, VectorFieldODE, TorchDiffeqSolver
from src.flow.path import ConditionalVectorFieldModel
from src.flow.path_stft import get_path
from src.models.seanet import GeneratorSeanet
from src.models.convnext_unet import ConvNeXtUNet
from src.trainer.trainer import WaveTrainer
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

def setup_model_and_solver(config, ckpt_path, method='euler', device='cuda'):
    """
    Initializes the model, ODE solver, and probability path.
    """
    # Load the model
    model = ConvNeXtUNet(**config.model).to(device)
    model = STFTTrainer.load_model_for_inference(model, ckpt_path, device=device)
    model.eval()

    # Path
    path = get_path(config.path).to(device)
    ode = VectorFieldODE(net=model)
    solver = TorchDiffeqSolver(ode, method=method)
    
    # Trainer
    transform = AmplitudeCompressedComplexSTFT(**config.transform)
    trainer = STFTTrainer(
                        path=path,
                        model=model,
                        logger=get_logger(config),
                        train_loader=None,
                        val_loader=None,
                        transform=transform,
                        device='cuda',
    )
    
    return model, solver, path, trainer

def run_ode_inference(solver, trainer, lr_audio, num_timesteps=50, device='cuda'):
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

    # Source    
    x0 = trainer.path.sample_source(Y).to(device)
    ts = torch.linspace(0, 1, num_timesteps+1)

    # ODE
    with torch.no_grad():
        generated_audio = solver.simulate(x0, ts=ts, y=Y) # x1 is the result at t=1
    generated_audio = trainer._postprocess(generated_audio)

    # Return the final generated audio, remove batch and channel dims for saving
    return generated_audio.squeeze(0).squeeze(0).cpu()

def inference_single_file(config, ckpt_path, idx_to_test=0, exp_name='ode_results', method='euler', num_timesteps=50, device='cuda'):
    """
    Runs inference on a single file from the dataset for quick testing and debugging.
    """
    print(f"--- Running inference for single file (index: {idx_to_test}) ---")
    
    # Setup
    model, solver, path, trainer = setup_model_and_solver(config, ckpt_path, method, device)
    _, val_loader = prepare_dataloader(config)
    val_dataset = val_loader.dataset
    
    # Get a single data point
    outdict= val_dataset[idx_to_test]
    lr_wave = outdict['lr_wave']
    hr_wave = outdict['hr'] 
    name = outdict['filename']

    # Run inference
    print(f"Generating audio for: {name}")
    start_time = time.time()
    generated_wave = run_ode_inference(solver, trainer, lr_wave, num_timesteps, device)
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
    parser.add_argument("--method", type=str, default='euler')
    parser.add_argument("--save_gt", type=bool, default=False)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--log_term", type=int, default=10)
    
    args = parser.parse_args()

    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = 'cuda'
    config = load_config(args.config)
    # ckpt_path = "./ckpts/best_model.pth"
    ckpt_path = os.path.join(config.train.ckpt_save_dir, "best_model.pth")
    
    for ode_steps in [10, 25]:
        # for idx_to_test in np.arange(0, 2000, 100):
        for idx_to_test in np.arange(0, 800+1, 80):
            inference_single_file(config, ckpt_path, idx_to_test=idx_to_test, 
                                  method=args.method, num_timesteps=ode_steps, 
                                  device=DEVICE, exp_name=os.path.join(args.exp_name, args.method,f'{ode_steps}'))
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
