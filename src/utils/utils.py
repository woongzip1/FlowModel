import os
import yaml

import torch
import librosa
import random
import numpy as np

from box import Box
from matplotlib import pyplot as plt

## t2n n2t

def t2n(x: torch.Tensor) -> np.ndarray :
    return x.detach().cpu().squeeze().numpy()  

## Utility functions for dataset construction
## supports global multisampling [12, 16, 20 kbps]
def get_audio_paths(paths: list, file_extensions=['.wav', '.flac']):
    """ Get list of all audio paths """
    audio_paths = []
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        for root, dirs, files in os.walk(path):
            audio_paths += [os.path.join(root, file) for file in files 
                            if os.path.splitext(file)[-1].lower() in file_extensions] 
    audio_paths.sort(key=lambda x: os.path.split(x)[-1])
    
    return audio_paths

def get_filename(path):
    return os.path.splitext(os.path.basename(path))

def read_tsv_firstcol(tsv_path):
    with open(tsv_path) as f:
        return [line.split('\t',1)[0].strip() for line in f if line.strip()]

## Load config
def load_config(config_path):
    with open(config_path, "r") as file:
        return Box(yaml.safe_load(file))

def _worker_init_fn(worker_id):
    base_seed = torch.initial_seed()
    seed = (base_seed + worker_id) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

## Visualize data
def draw_spec(x,
              figsize=(7,4), title='', n_fft=2048,
              win_len=1024, hop_len=512, sr=16000, cmap='inferno',
              window='hann',
              vmin=-50, vmax=40, use_colorbar=False,
              ylim=None,
              title_fontsize=10,
              label_fontsize=8,
                return_fig=False,
                save_fig=False, save_path=None):
    fig = plt.figure(figsize=figsize)
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop_len, win_length=win_len, window=window)
    stft = 20 * np.log10(np.clip(np.abs(stft), a_min=1e-8, a_max=None))

    r=5
    # stft[...,100-r:100+r] = -50
    
    plt.imshow(stft,
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
               origin='lower', extent=[0, len(x) / sr, 0, sr//2])

    if use_colorbar:
        plt.colorbar()

    plt.xlabel('Time (s)', fontsize=label_fontsize)
    plt.ylabel('Frequency (Hz)', fontsize=label_fontsize)

    if ylim is None:
        ylim = (0, sr / 2)
    plt.ylim(*ylim)

    plt.title(title, fontsize=title_fontsize)
    plt.tight_layout()

    if save_fig and save_path:
        plt.savefig(f"{save_path}.png")
    
    if return_fig:
        plt.close()
        return fig
    else:
        # plt.close()
        plt.show()
        return stft
    
def draw_2d_heatmap(spectrum: torch.tensor, cmap='inferno', vmin=None, vmax=None, title=None,
                    figsize=(7,4), ylim=None, 
                    save_fig=False, save_path='save.png', show_fig=True,
                    sr=24000, use_colorbar=True):
    # spectrum [F,T]
    assert spectrum.squeeze().dim()==2, \
        f'shape of input must be [F,T], input:{spectrum.squeeze().dim()}'
        
    # note: dB scale -> 20 log10 (x)
    # spectrum = 20 * torch.clip(spectrum, min=1e-8,) 
    spectrum = spectrum.squeeze().detach().cpu().numpy()
    
    # plot heatmap
    fig = plt.figure(figsize=figsize)
    im = plt.imshow(spectrum,
               aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
               origin='lower', 
               extent=[0, spectrum.shape[-1], 0, sr//2]
               )
    
    if use_colorbar:
        from matplotlib.ticker import FuncFormatter
        cbar = plt.colorbar(im)
        formatter = FuncFormatter(lambda x, pos: f'{x:3.1f}')
        cbar.ax.yaxis.set_major_formatter(formatter)
    plt.xlabel('Frame', fontsize=8)
    plt.ylabel('Frequency', fontsize=8)

    if title:
        plt.title(title)
    if ylim:
        plt.ylim(ylim)
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_path)
        return 
    if show_fig:
        plt.show()
    else:
        plt.close()
    # plt.show()
    return fig

def plot_signals(x, x_hat, range=[10000,15000],figsize=(10,2), diff=0.01):
    plt.figure(figsize=figsize)
    plt.plot(x, label='gt')
    plt.plot(x_hat+diff, label='s')
    plt.xlim(range)
    plt.legend()
    plt.show()

def print_config(config, indent=0):
    for k, v in config.items():
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            print_config(v, indent + 4)
        else:
            print(" " * indent + f"{k}: {v}")
            
## ---- Metrics adopted from AP-BWE ---- ####
def log_spectral_distance(ref_audio, syn_audio):
    spec_pred = torch.log10(stft(syn_audio.squeeze(1))[0].square().clamp(min=1e-8)) 
    spec_ref = torch.log10(stft(ref_audio.squeeze(1))[0].square().clamp(min=1e-8)) 
    lsd = (spec_pred - spec_ref).square().mean(dim=1).sqrt().mean()
    return lsd

def lsd_high(ref_audio, syn_audio, cutoff_freq, n_fft=2048, hop_length=512, sr=24000):
    """Calculates the Log-Spectral Distance for the high-frequency components."""
    # Perform STFT and get the log-magnitude spectrograms
    # The [0] index is used to get the magnitude from stft's output tuple.
    spec_pred = torch.log10(stft(syn_audio, n_fft, hop_length)[0].square().clamp(1e-8))
    spec_ref = torch.log10(stft(ref_audio, n_fft, hop_length)[0].square().clamp(1e-8))

    # Convert cutoff frequency in Hz to the corresponding frequency bin index
    # The frequency of the k-th bin is k * (sr / n_fft)
    # So, the index k is cutoff_freq * n_fft / sr
    
    cutoff_bin = int(cutoff_freq * n_fft / sr)

    # Slice the spectrograms to keep only the high-frequency parts
    spec_pred_hf = spec_pred[cutoff_bin:, :]
    spec_ref_hf = spec_ref[cutoff_bin:, :]

    # Calculate LSD on the high-frequency components
    lsd = (spec_pred_hf - spec_ref_hf).square().mean(dim=1).sqrt().mean()
    return lsd

def stft(audio, n_fft=2048, hop_length=512):
    hann_window = torch.hann_window(n_fft).to(audio.device)
    stft_spec = torch.stft(audio, n_fft, hop_length, window=hann_window, return_complex=True)
    stft_mag = torch.abs(stft_spec)
    stft_pha = torch.angle(stft_spec)
    return stft_mag, stft_pha
