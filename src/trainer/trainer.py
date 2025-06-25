import torch
# from flow_matching.paths import GaussianPath
# from flow_matching.losses import conditional_flow_loss

class Trainer:
    def __init__(self, model, optimizer, config, ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        # self.path = GaussianPath()
        
    def train_step(self, batch):
        # batch should contain the original data x0
        x0 = batch
        
        # 1. Sample random time t ~ U[0, 1]
        # Adding a small epsilon to avoid t=0 issues if any
        t = torch.rand(x0.shape[0], device=x0.device) * (1 - 1e-4) + 1e-4

        # 2. Sample random noise
        epsilon = torch.randn_like(x0)

        # 3. Compute x_t and the target vector field u_t using the path object
        x_t, u_t = self.path.get_xt_and_ut(x0, t, epsilon)

        # 4. Get model prediction for the vector field
        v_pred = self.model(x_t, t)

        # 5. Calculate loss
        loss = conditional_flow_loss(v_pred, u_t)
        
        # 6. Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, train_loader):
        # train loop
        