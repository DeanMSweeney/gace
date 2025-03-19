import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons, make_circles

class Flow(nn.Module): 
    '''
    Defines the architecture for flow matching
    '''

    def __init__(self, dim: int = 2, h: int = 64): # fully connected neural network for learning velocity field 
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h),  
            nn.ELU(),              
            nn.Linear(h, h),       
            nn.ELU(),             
            nn.Linear(h, h),      
            nn.ELU(),              
            nn.Linear(h, dim)       
        )
    
    def forward(self, t: Tensor, xt: Tensor) -> Tensor:
        '''
        Concatenates time and position vectors for forward pass through neural network  
        '''

        combined_input = torch.cat((t, xt), -1)

        return self.net(combined_input)
    
    def step(self, xt: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor: 
        '''
        Performs midpoint integration to sample transform x0 to x1 (i.e., sample from p1, starting from p0)
        '''
      
        t_start = t_start.view(1, 1).expand(xt.shape[0], 1)
        midpoint_time = t_start + (t_end - t_start) / 2 # evalute the midpoint 
        intermediate_value = self(xt=xt, t=t_start) # compute the velocity at this point using the trained neural network 
        
        updated_xt = xt + intermediate_value * (t_end - t_start) / 2 # move xt using computed velocity 
        midpoint_output = self(t=midpoint_time, xt=updated_xt) # compute vt again 
        
        return xt + (t_end - t_start) * midpoint_output # compute xt

def sample_from():
    '''
    Defines sampling procedures from x1 and x0
    '''

    # Generate data for a dummy distribution (testing purposes only)
    rv1 = torch.rand(1000) * 4 - 2 
    rv2 = torch.rand(1000) * 4 - 2 
    mask = (torch.floor(rv1) + torch.floor(rv2)) % 2 == 0  

    x1 = torch.stack((rv1[mask], rv2[mask]), dim=1)
    x0 = torch.randn_like(x1)

    return x0, x1

# Training loop
def train():
    '''
    Defines the training procedure for the neural network
    '''
    for _ in range(1000000):
        x0, x1 = sample_from()
        t = torch.rand(len(x1), 1)
        xt = (1 - t) * x0 + t * x1

        # Compute loss 
        loss = loss_fn(flow(t=t, xt=xt), x1 - x0)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Loss: {loss.item()}")

######################################################
# Model training 
######################################################
# Initialize model and optimizer
flow = Flow()
optimizer = torch.optim.Adam(flow.parameters(), 1e-3)
loss_fn = nn.MSELoss()
train()


######################################################
# Model sampling
######################################################
'''
x = torch.randn(300, 2)
n_steps = 8
fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
time_steps = torch.linspace(0, 1.0, n_steps + 1)

axes[0].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
axes[0].set_title(f't = {time_steps[0]:.2f}')
axes[0].set_xlim(-2.0, 2.0)
axes[0].set_ylim(-2.0, 2.0)

for i in range(n_steps):
    x = flow.step(xt=x, t_start=time_steps[i], t_end=time_steps[i + 1])
    axes[i + 1].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
    axes[i + 1].set_title(f't = {time_steps[i + 1]:.2f}')

plt.tight_layout()
plt.show()
'''