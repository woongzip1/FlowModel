## FlowMatchingLoss (MSE)

import torch
import torch.nn.functional as F

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
    
    