import sys
import pdb

import random
import torch
import torchaudio
import torch.nn as nn
import numpy as np

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.utils.utils import read_tsv_firstcol, load_config, _worker_init_fn


def make_dataset(config, mode:str):
    return Dataset(
        **config.dataset.common,
        **config.dataset[mode],
        mode=mode,
    )

def prepare_dataloader(config):
    # config = load_config(config_path)
    train_dataset = make_dataset(config, 'train')
    val_dataset = make_dataset(config, 'val')

    # collator
    collator = WaveformCollator(
        target_sr=config.dataset.common.sr,
        sampling_rates_probs=config.collator.sampling_rates_probs
    )

    # Optional ratio split
    if config.dataset.ratio < 1:
        train_size = int(config.dataset.ratio * len(train_dataset))
        _, train_dataset = random_split(train_dataset, [len(train_dataset) - train_size, train_size])

    dl_args = dict(config.dataloader)
    dl_args['worker_init_fn'] = _worker_init_fn
    dl_args['collate_fn'] = collator
    
    train_loader = DataLoader(train_dataset, shuffle=True, **dl_args)
    
    val_loader_args = config.dataloader
    val_loader_args['worker_init_fn'] = _worker_init_fn
    val_loader_args['collate_fn'] = collator
    val_loader_args['batch_size'] = 1
    val_loader = DataLoader(val_dataset, shuffle=False, **val_loader_args)

    return train_loader, val_loader

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 tsv_path: str,
                 num_samples=24000, #16384
                 sr=24000,
                 sampling_rates=[8,16,24], # no use
                 mode="train"):
        self.num_samples, self.sr, self.mode = num_samples, sr, mode
        self.sampling_rates=sampling_rates
        
        self.wb_paths = read_tsv_firstcol(tsv_path)
        print(len(self.wb_paths), 'samples loaded!')

    def __len__(self):
        return len(self.wb_paths)

    def _pad(self, wav, N=80):
        pad = (N - wav.shape[-1] % N) % N
        return torch.nn.functional.pad(wav, (0,pad))

    def _ensure(self, wav, L, repeat=True):
        # if short: repeat, else: crop
        if wav.shape[-1] < L and repeat: 
            wav = torch.nn.functional.pad(wav, (0, 4000)) # offset
            reps = (L + wav.shape[-1] - 1) // wav.shape[-1]          # ceil(L / wav.shape[-1])
            wav = wav.repeat(1, reps)[..., :L]   # repeat
        elif wav.shape[-1] < L and not repeat:
            pad = L - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad))        
        elif wav.shape[-1] > L:        
            wav = wav[..., :L]
        return wav

    def __getitem__(self, idx):
        wb_path = self.wb_paths[idx]
        y, sr = torchaudio.load(wb_path)
        if y.size(0) > 1:
            # print("Stereo file detected..")
            y = y.mean(dim=0, keepdim=True)
    
        # gain & normalize        
        gain = np.random.uniform(-1,-6) if self.mode=='train' else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]]) # peak normalize
        
        # resample
        if sr != self.sr:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sr)
        
        if self.mode=="train":
            target_signal_len = self.num_samples
            current_signal_len = y.shape[-1]
            # print(target_signal_len)
            if current_signal_len <= target_signal_len:
                y = self._ensure(y, target_signal_len)
            else:
                s = np.random.randint(0, current_signal_len - target_signal_len)
                y = y[..., s:s+target_signal_len]
        elif self.mode in ['val']:
            # y = self._ensure(y, 48000*4, repeat=False)
            y = y[...,:48000*4]
        else:
            sys.exit(f"unsupported mode! (train/val)") 
         
        outdict = {
            'hr': y,
            'filename': Path(wb_path).stem,
        }

        return outdict

class WaveformCollator:
    def __init__(self, 
                 target_sr=48000, 
                 sampling_rates_probs={8: 0.7, 12: 0.1, 16: 0.1, 24: 0.1}):
        """
        Initializes the collator.
        Args:
            target_sr (int): The high-resolution sample rate (e.g., 48000).
            sampling_rates_probs (dict): A dictionary mapping sample rates (in kHz) to their sampling probabilities.
                                         Example: {8: 0.7, 12: 0.1, 16: 0.1, 24: 0.1}
        """
        self.target_sr = target_sr
        self.sampling_rates = list(sampling_rates_probs.keys())  # [8, 12, 16, 24]
        self.probs = list(sampling_rates_probs.values()) # [0.7, 0.1, 0.1, 0.1]

    def _apply_lpf(self, hr_wave, low_sr_khz):
        """
        Applies a low-pass filter by downsampling and then upsampling the waveform.
        This correctly simulates the anti-aliasing filter effect.
        """
        original_len = hr_wave.shape[-1]
        target_sr_hz = low_sr_khz * 1000
        
        # Downsample to the target low sample rate
        lr_wave_resampled = torchaudio.functional.resample(
            hr_wave, orig_freq=self.target_sr, new_freq=target_sr_hz
        )
        # Upsample back to the original high sample rate to match lengths
        lr_wave_upsampled = torchaudio.functional.resample(
            lr_wave_resampled, orig_freq=target_sr_hz, new_freq=self.target_sr
        )
        
        lr_wave_upsampled = lr_wave_upsampled[..., :original_len]
        return lr_wave_upsampled

    def __call__(self, batch):
        """
        Processes a batch of data items from the Dataset.
        """
        # 1. Choose one low_sr for the entire batch based on given probabilities
        low_sr_khz = random.choices(self.sampling_rates, self.probs, k=1)[0]
        
        # 2. Stack HR waveforms from the batch items
        # Assuming dataset returns {'hr': tensor_shape_[1, T], ...}
        hr_waves = torch.stack([item['hr'].squeeze(0) for item in batch])

        # 3. Create LR versions by applying the LPF
        lr_waves = self._apply_lpf(hr_waves, low_sr_khz)
        
        # 4. Return a dictionary that matches the trainer's expectation
        return {
            'hr': hr_waves.unsqueeze(1),       # Shape: [B, 1, T]
            'lr_wave': lr_waves.unsqueeze(1), # Shape: [B, 1, T]
            'low_sr': [low_sr_khz] * len(batch) # [B]
        }

    """
    HR -> (Downsample) -> LR -> (Upsample) -> STFT extraction
    """
    def _get_lr_wav(self, wave: torch.Tensor, target_sr: int=8000, orig_sr: int=24000,) -> torch.Tensor:
        """
        input -  [B, T] 
        output - [B, T]; HF lost wave signal
        """
        ## apply random low pass filter
        wave_lr = torchaudio.functional.resample(waveform=wave, orig_freq=orig_sr, new_freq=target_sr,)
        wave_lr = torchaudio.functional.resample(waveform=wave_lr, orig_freq=target_sr, new_freq=orig_sr,)[...,:wave.shape[-1]]
        return wave_lr
    
    def _get_stft(self, wave: torch.Tensor) -> torch.Tensor:
        """
        B-dimension can be ommited
        input -  [B, T]
        output - [B,1,F,T,2]
        """
        # hop_length=256
        # target_signal_len = wave.shape[-1] // hop_length * hop_length # make multiple\
        # wave = wave[...,:target_signal_len]
        
        pad_len = 1024-1
        wave = torch.nn.functional.pad(wave, pad=[0,pad_len])
        spec = torch.stft(wave, n_fft=1024, hop_length=256, 
                          window=torch.hann_window(1024),
                          return_complex=True, center=False, ) # need to care abourrt alignment
        spec = torch.view_as_real(spec)
        return spec

def main():
    config_path = 'configs/config_template.yaml'
    config = load_config(config_path)
    train_dataset = make_dataset(config, 'train')
    # train, val = prepare_dataloader(config)
    # pdb.set_trace()
    
    print(len(train_dataset))
    for i in train_dataset:
        print(len(i))
        wb, lr_spec, name, = i
        print(wb.shape, lr_spec.shape, name,)
        pdb.set_trace()
        # break

if __name__ == "__main__":
    main()
"""
##### python split_vctk.py --root /home/woongzip/Dataset_48khz/GT/VCTK --out_dir ./
##### python split_vctk.py --root /home/woongzip/dataset_real/GT/VCTK --out_dir ./
## find /home/woongzip/dataset_real/GT/MUSDB18_split/test -name mixture*.wav > mixture.val

### LibriTTS
## find /home/woongzip/Dataset_24khz/LibriTTS/train-clean-100 /home/woongzip/Dataset_24khz/LibriTTS/train-clean-360 /home/woongzip/Dataset_24khz/LibriTTS/train-other-500 -name "*.wav" > libri-train.txt
## find /home/woongzip/Dataset_24khz/LibriTTS/train-clean-100 /home/woongzip/Dataset_24khz/LibriTTS/train-clean-360 /home/woongzip/Dataset_24khz/LibriTTS/train-other-500 -name "*.wav" > libri-train.txt
## find /home/woongzip/Dataset_24khz/LibriTTS/train-clean-100 /home/woongzip/Dataset_24khz/LibriTTS/train-clean-360 /home/woongzip/Dataset_24khz/LibriTTS/train-other-500 -name "*.wav" > libri-train.txt

### Audio Dataset (NSynth)
# find /home/woongzip/Dataset/nsynth-train -name "*.wav" > nsynth-train.txt
# find /home/woongzip/Dataset/nsynth-val -name "*.wav" > nsynth-val.txt
# find /home/woongzip/Dataset/nsynth-test -name "*.wav" > nsynth-testin.txt

find /ssd/woongzip/dataset_real/GT/MUSDB18_split -name "*.wav" > audio_48.txt
find /ssd/woongzip/dataset_real/GT/MUSDB18_split -name "*vocals_*.wav" > vocals_48.txt

find /ssd/woongzip/dataset_real/GT/VCTK -name "*.wav" > speech_48.txt
"""