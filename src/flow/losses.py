## FlowMatchingLoss (MSE)

import torch
import torch.nn.functional as F
import numpy as np

from src.utils.utils import draw_2d_heatmap

def flow_matching_loss(predicted_vf: torch.Tensor, target_vf: torch.Tensor) -> torch.Tensor:
    """
    Flow matching loss; L2 loss between estimated and target vf
    """
    loss = F.mse_loss(predicted_vf, target_vf)
    
    return loss

def masked_flow_matching_loss(
    predicted_vf: torch.Tensor, 
    target_vf: torch.Tensor, 
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Flow matching loss; L2 loss between estimated and target vf
    
    predicted_vf [B,2,F,T]
    target_vf [B,2,F,T]
    mask [B,1,F,T] (hr preserving)
    """
    import pdb
    error = (predicted_vf - target_vf) ** 2
    mask_expanded = mask.expand_as(error)
    masked_error = error * mask
    
    # calculate loss at only generated HF region
    B = predicted_vf.shape[0] # Batch size
    # error : [B,2,F,T]
    # mask: [B,1,F,1]
    
    # [B,2FT]
    # error_sum = masked_error.reshape(B,-1).sum(dim=1)
    # mask_sum = mask_expanded.reshape(B,-1).sum(dim=1).clamp_min(eps)
    error_sum = masked_error.sum(dim=(1,2,3))
    mask_sum = mask_expanded.sum(dim=(1,2,3)).clamp_min(eps)
    
    mse_per_sample = error_sum / mask_sum
    loss = mse_per_sample.mean() # mean over Batch
    
    return loss

def unwrap(p, discont=None, dim=-1, *, period=2*np.pi):
    nd = p.ndim
    dd = torch.diff(p, dim=dim)
    if discont is None:
        discont = period/2
    slice1 = [slice(None, None)]*nd     # full slices
    slice1[dim] = slice(1, None)
    slice1 = tuple(slice1)
    interval_high = period / 2
    boundary_ambiguous = True
    interval_low = -interval_high
    ddmod = torch.remainder(dd - interval_low, period) + interval_low
    if boundary_ambiguous:
        ddmod[(ddmod == interval_low) & (dd > 0)] = interval_high
    ph_correct = ddmod - dd
    ph_correct[abs(dd) < discont] = 0.
    up = p.clone()
    up[slice1] = p[slice1] + ph_correct.cumsum(dim)
    return up

def compute_instantaneous_frequency(complex_spec):
    phi = torch.angle(complex_spec)
    delta_phi = torch.remainder(torch.diff(phi, dim=2), 2 * np.pi)
    uw_delta_phi = unwrap(delta_phi, dim=2)
    psi = torch.remainder(
        (uw_delta_phi[..., :-1] + uw_delta_phi[..., 1:]) / 2., 2 * np.pi)

    return psi

def compute_phase_loss(pred_psi, targ_psi, w=None, mask=None, debug=False):
    if w is None:
        t_pad_targ_psi = F.pad(targ_psi, (1, 1, 0, 0))
        f_pad_targ_psi = F.pad(targ_psi, (0, 0, 1, 1))
        s_tm1 = torch.remainder(t_pad_targ_psi[..., :-2] - targ_psi, 2 * np.pi).abs()
        s_tp1 = torch.remainder(t_pad_targ_psi[..., 2:] - targ_psi, 2 * np.pi).abs()
        s_fm1 = torch.remainder(f_pad_targ_psi[:, :-2] - targ_psi, 2 * np.pi).abs()
        s_fp1 = torch.remainder(f_pad_targ_psi[:, 2:] - targ_psi, 2 * np.pi).abs()
        s = s_tm1 + s_tp1 + s_fm1 + s_fp1
        
        w = (1 + s).mean() / (1 + s)
        w = w * mask[...,1:-1] # mask out silences
        w = w.clip(max=3)
        
    # w = torch.ones_like(w) * mask[...,1:-1] # without w
    temp = ((1 - torch.cos(pred_psi - targ_psi)) * w)
    loss = ((1 - torch.cos(pred_psi - targ_psi)) * w).mean()
    
    if debug:
        draw_2d_heatmap(w, title='mask w', vmax=3)
        draw_2d_heatmap(mask, title='mag mask')
        draw_2d_heatmap(temp, title='masked region', vmax=2)
        pass
    
    return loss, temp

def stft_loss(
    answer,
    predict,
    fft_sizes=(1024, 2048, 512),
    hop_sizes=(128, 256, 64),
    win_lengths=(512, 1024, 256),
    f1_start_bin=(80, 160, 40),
    window=torch.hann_window,
    debug=False,
):
    loss = 0
    mag_loss_sum = 0
    sc_loss_sum = 0
    phase_loss_sum = 0
    
    for i in range(len(fft_sizes)):
        
        ansStft = torch.view_as_real(
            torch.stft(
                answer.squeeze(1),
                n_fft=fft_sizes[i],
                hop_length=hop_sizes[i],
                win_length=win_lengths[i],
                window=window(win_lengths[i], device=answer.device),
                return_complex=True,
            )
        )
        predStft = torch.view_as_real(
            torch.stft(
                predict.squeeze(1),
                n_fft=fft_sizes[i],
                hop_length=hop_sizes[i],
                win_length=win_lengths[i],
                window=window(win_lengths[i], device=predict.device),
                return_complex=True,
            )
        )
        
        ansStft = ansStft[...,f1_start_bin[i]:,:,:]
        predStft = predStft[...,f1_start_bin[i]:,:,:]
        
        ansStftMag = ansStft[..., 0] ** 2 + ansStft[..., 1] ** 2
        predStftMag = predStft[..., 0] ** 2 + predStft[..., 1] ** 2

        magMin = 1e-6
        mask = (ansStftMag > magMin)
        # print(mask.shape, 'mask shape', torch.min(mask), torch.max(mask))
        
        # mag spec
        ansStftMag = torch.sqrt(ansStftMag + magMin)
        predStftMag = torch.sqrt(predStftMag + magMin)

        # log mag spec
        ansStftMag_log = torch.log10(torch.clip(ansStftMag, min=1e-7))
        predStftMag_log = torch.log10(torch.clip(predStftMag, min=1e-7))

        mag_loss = F.mse_loss(predStftMag_log, ansStftMag_log)
        converge_loss = (torch.norm(predStftMag - ansStftMag, p="fro") /
                         (torch.norm(ansStftMag, p="fro") + 1))

        if debug:
                draw_2d_heatmap((predStftMag_log-ansStftMag_log).pow(2), vmin=0, vmax=None, title='Mag loss')
                draw_2d_heatmap((predStftMag-ansStftMag).pow(2) / (ansStftMag + 1).pow(2), vmin=None, vmax=None, title='SC loss')
                pass
            
        # --  phase loss
        pred_if = compute_instantaneous_frequency(torch.view_as_complex(predStft))
        targ_if = compute_instantaneous_frequency(torch.view_as_complex(ansStft))
        # draw_2d_heatmap(pred_if)
        # draw_2d_heatmap(targ_if)
        phase_loss, temp= compute_phase_loss(pred_if, targ_if, mask=mask, debug=debug)
        # -----

        mag_loss_sum += mag_loss
        sc_loss_sum += converge_loss 
        phase_loss_sum += phase_loss
        
        ## ---- take care of these
        phase_loss_sum = 0

    loss = (mag_loss_sum + sc_loss_sum + phase_loss_sum)/len(fft_sizes)
    return loss

    
def main():
    
    predicted_vf = torch.randn(5,1,32,32) # [B,C,H,W]
    target_vf = torch.randn(5,1,32,32)
    
    # Loss over batch
    square_error_per_sample = torch.sum((predicted_vf-target_vf)**2, dim=(1,2,3))
    loss = torch.mean(square_error_per_sample)
    
    print(loss) # without normalization 
    print(loss / (32*32)) # with normalization
    print(F.mse_loss(predicted_vf, target_vf)) # with normalization
    return

if __name__=="__main__":
    main()
    
    