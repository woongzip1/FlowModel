# train.py

import torch
import yaml
import random
import argparse
from datetime import datetime

# --- WandB for logging ---
import wandb
from box import Box

# --- Assumed project structure for imports ---
from data.dataset import prepare_dataloader # Using the dataloader function from your core logic
from src.models.seanet import GeneratorSeanet # The model from your core logic
from src.flow.path import DataLoaderConditionalProbabilityPath, LinearAlpha, LinearBeta, LinearSigmaBeta
from src.trainer.trainer import WaveTrainer, CFGWaveTrainer # Your refactored WaveTrainer
from src.utils.utils import print_config  # Assuming you have a print_config utility

# --- Global Constants ---
DEVICE = f'cuda' if torch.cuda.is_available() else 'cpu'
print(f"INFO: Running on device: {DEVICE}")

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Flow-Matching Model Training Script")
    parser.add_argument('--config', type=str, default='configs/config_template.yaml', required=True, help="Path to the training configuration file.")
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

    # Initialize WandB if enabled
    if args.wandb:
        wandb.init(
            project=config.wandb.project_name,
            entity=config.wandb.entity,
            config=config.to_dict(), # wandb expects a dict
            name=config.wandb.run_name,
            notes=config.wandb.get('notes', '')
        )
    
    # --- 2. Data Preparation ---
    print("INFO: Preparing dataloaders...")
    train_loader, val_loader = prepare_dataloader(config)

    # --- 3. Model, Path, and Trainer Initialization ---
    print("INFO: Initializing model, path, and trainer...")
    alpha = LinearAlpha()
    # beta = LinearBeta()
    beta = LinearSigmaBeta(sigma=1e-4)
    
    path = DataLoaderConditionalProbabilityPath(
        p_simple_shape=config.path.p_simple_shape, # e.g., [16, 1, 38400]
        alpha=alpha,
        beta=beta,
    )

    # Initialize the Model
    model = GeneratorSeanet(**config.model)

    # # The optimizer is created inside the WaveTrainer
    # trainer = WaveTrainer(
    #     path=path,
    #     model=model,
    #     dataloader=train_loader,
    # )

    trainer = CFGWaveTrainer(
        path=path,
        model=model,
        dataloader=train_loader,
        eta=0.1,
    )

    # --- Start Training ---
    print("INFO: Starting training...")
    trainer.train(
        num_epochs=config.train.num_epochs,
        device=torch.device(DEVICE),
        lr=config.optimizer.learning_rate,
        ckpt_save_dir=config.train.ckpt_save_dir,
        ckpt_load_path=config.train.get('ckpt_load_path', None),
    )

    print("INFO: Training finished.")
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()