## FlowMatchingLoss (MSE)

import torch
import torch.nn.functional as F

def flow_matching_loss(predicted_vf: torch.Tensor, target_vf: torch.Tensor) -> torch.Tensor:
    """
    Flow matching loss; L2 loss between estimated and target vf
    """
    loss = F.mse_loss(predicted_vf, target_vf)
    
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
    
    