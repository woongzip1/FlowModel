import os
import pdb
import torch
import torchaudio

from eval.visqol.visqol_wrapper import ViSQOL 

MODE = "speech"
MODE = "audio"

def main():
    # Define the path to the folder where ViSQOL was built
    VISQOL_FOLDER_PATH = os.path.expanduser("~/tools/visqol")
    
    # Load files
    sr = 48000
    ref_audio, _ = torchaudio.load("../es01.wav")
    deg_audio, _ = torchaudio.load("../es01_noise.wav")

    print(f"Audio loaded. Sample Rate: {sr}, Shape: {ref_audio.shape}")
    print(f"Creating ViSQOL object (mode='{MODE}')...")
    try:
        visqol_calculator = ViSQOL(visqol_folder=VISQOL_FOLDER_PATH,
                                     mode=MODE,
                                    )
        print("ViSQOL object created successfully.")
    except AssertionError as e:
        print(f"Failed to create ViSQOL object: {e}")
        print("Please check if VISQOL_FOLDER_PATH is correct and if the ViSQOL build was successful.")
        return

    # The wrapper expects a batch in the shape [B, C, T], so we add a dimension with .unsqueeze(0).
    print("Calculating ViSQOL score...")
    score = visqol_calculator(ref_audio.unsqueeze(0), deg_audio.unsqueeze(0), sr)
    
    print("-" * 30)
    print(f"✅ ViSQOL (MOS-LQO) score: {score:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
    
    