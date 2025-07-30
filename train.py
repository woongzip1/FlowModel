# train.py

import pdb
import torch
import yaml
import random
import argparse
from datetime import datetime

# --- WandB for logging ---
import wandb
from box import Box
from torchinfo import summary

# --- Assumed project structure for imports ---
# from data.dataset import prepare_dataloader 
from data.dataset_idx import prepare_dataloader 

from src.models.seanet import GeneratorSeanet # The model from your core logic
from src.flow.path import DataLoaderConditionalProbabilityPath, DataLoaderConditionalProbabilityPathWithPrior, LinearAlpha, LinearBeta, LinearSigmaBeta
from src.trainer.trainer import WaveTrainer, CFGWaveTrainer, WaveTrainerWithPrior
from src.trainer.stft_trainer import STFTTrainer
from src.trainer.stft_trainer_mask import STFTTrainerMask


from src.utils.utils import print_config  # Assuming you have a print_config utility
from src.utils.spectral_ops import InvertibleFeatureExtractor, AmplitudeCompressedComplexSTFT
from src.utils.logger import BaseLogger, get_logger

from src.models.convnext_unet import ConvNeXtUNet, ConditionalVectorFieldModel
from src.models.convnext_unet_condition import ConvNeXtUNetCond
from src.models.unetv3 import ConvNeXtUNetFiLM
from src.flow.path_stft import get_path



# --- Global Constants ---
DEVICE = f'cuda' if torch.cuda.is_available() else 'cpu'
print(f"INFO: Running on device: {DEVICE}")

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Flow-Matching Model Training Script")
    parser.add_argument('-c', '--config', type=str, default='configs/config_template.yaml', required=True, help="Path to the training configuration file.")
    parser.add_argument('--wandb', type=lambda x: x.lower() == 'true', default=False, help="Set to 'true' to enable WandB logging.")
    return parser.parse_args()

def load_config(config_path: str) -> Box:
    """Loads a YAML config file into a Box object for easy access."""
    with open(config_path, "r") as file:
        return Box(yaml.safe_load(file))

def main():
    """Main training function."""
    # --- 1. Setup and Configuration ---
    args = parse_args()
    config = load_config(args.config)
    
    # For reproducibility
    torch.manual_seed(config.get('seed', 42))
    random.seed(config.get('seed', 42))

    print("--- Configuration ---")
    print_config(config)
    print("---------------------")

    # Initialize wandb
    logger = get_logger(config, args.wandb)
    
    # --- 2. Data Preparation ---
    print("INFO: Preparing dataloaders...")
    train_loader, val_loader = prepare_dataloader(config)

    # --- 3. Model, Path, and Trainer Initialization ---
    print("INFO: Initializing model, path, and trainer...")
    # alpha = LinearAlpha()
    # beta = LinearBeta()
    # beta = LinearSigmaBeta(sigma=1e-4)
    
    # # Initialize the Model
    # model = GeneratorSeanet(**config.model)
    # path = DataLoaderConditionalProbabilityPath(
    #     p_simple_shape=config.path.p_simple_shape, # e.g., [16, 1, 38400]
    #     alpha=alpha,
    #     beta=beta,
    # )
    # # The optimizer is created inside the WaveTrainer
    # trainer = WaveTrainer(
    #     path=path,
    #     model=model,
    #     dataloader=train_loader,
    # )

    ## ---
    # trainer = CFGWaveTrainer(
    #     path=path,
    #     model=model,
    #     dataloader=train_loader,
    #     eta=0.1,
    # )   
    ## --- 
        
    # Prior
    # model = GeneratorSeanet(**config.model)
    # path = DataLoaderConditionalProbabilityPathWithPrior(
    #     p_simple_shape=config.path.p_simple_shape, # e.g., [16, 1, 38400]
    #     alpha=alpha,
    #     beta=beta,
    # )
    # trainer = WaveTrainerWithPrior(
    #     path=path,
    #     model=model,
    #     dataloader=train_loader,
    # )

    ## STFT
    transform = AmplitudeCompressedComplexSTFT(**config.transform)
    path = get_path(config.path)
    # model = ConvNeXtUNet(**config.model)
    # model = ConvNeXtUNetCond(**config.model)
    model = ConvNeXtUNetFiLM(**config.model)
    
    # Dummy input
    # try:
    #     summary(
    #         model,
    #         input_data=[torch.randn(1,2,512,65), torch.rand(1), 
    #                     torch.randn(1,2,512,65), torch.ones(1, dtype=torch.int)*8],
    #         depth=4,
    #         col_names=["input_size", "output_size", "num_params"],
    #         verbose=1
    #     )
    # except:
    #     print("Summary passed")
    
    trainer = STFTTrainerMask(
                        path=path,
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        transform=transform,
                        device=torch.device(DEVICE),
                        logger=logger,
                        )
    
    # --- Start Training ---
    print("INFO: Starting training...")
    trainer.train(
        optimizer_config=config.optimizer,
        scheduler_config=config.scheduler,
        **config.train,
    )
    print("INFO: Training finished.")
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()