#!/usr/bin/env python
"""\
Physics-informed neural network example (published on TechShare)

This script is the physics-informed case.

Author: Henning Sauerland (ERD-ACL), henning.sauerland@hitachi-eu.com
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def analytical(x):
    """Defines the analytical solution"""
    return x**2+1

class FCN(nn.Module):
    "Defines a fully connected network (3 layers, 32 nodes)"
    def __init__(self):
        super().__init__()
        
        # define the layers
        activation = nn.Tanh()
        self.layers = nn.Sequential(
            nn.Linear(1, 32),
            activation,
            nn.Linear(32, 32),
            activation,
            nn.Linear(32, 32),
            activation,
            nn.Linear(32, 32),
            activation,
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # forward pass
        return self.layers(x)

# evaluate the analytical solution
x = torch.linspace(-1,1,500).view(-1,1)
y = analytical(x).view(-1,1)

# extract training data
x_data = x[0:250:50]
y_data = y[0:250:50]

# define sampling points for enforcing physics loss
x_physics = torch.linspace(-1,1,30).view(-1,1).requires_grad_(True)

# train standard neural network
torch.manual_seed(123)
model = FCN()
optimizer = torch.optim.Adam(model.parameters(),lr=5e-4)

losses1 = []
losses2 = []
for i in range(20000):
    optimizer.zero_grad()
    
    # compute data loss
    y_pred = model(x_data)
    loss1 = torch.mean((y_pred-y_data)**2) #mean squared error
    
    # compute the physics loss
    y_pred_physics = model(x_physics)
    dx  = torch.autograd.grad(y_pred_physics, x_physics, torch.ones_like(y_pred_physics), create_graph=True)[0] # dy/dx
    dx2 = torch.autograd.grad(dx, x_physics, torch.ones_like(dx), create_graph=True)[0] # d^2y/dx^2
    dx3 = torch.autograd.grad(dx2, x_physics, torch.ones_like(dx2), create_graph=True)[0] # d^3y/dx^3

    physics = dx3 # residual differential equation
    loss2 = torch.mean(physics**2)
    
    # aggregated loss
    loss = loss1 + loss2
    loss.backward()
    
    optimizer.step()
    losses1.append(loss1)
    losses2.append(loss2)

    # plot the result as training progresses
    if i % 500 == 0: print(str(i).ljust(5), " Loss: ",loss.item())

# plot training loss history
fig = plt.figure(constrained_layout=False, figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)
h = ax.plot(losses1, label = "Data loss")
h = ax.plot(losses2, label = "Physics loss")
ax.set_yscale('log')
ax.set_title('Training Loss')
ax.set_xlabel('epoch')
plt.legend()
plt.show()      

# plot results
y_pred = model(x).detach()
plt.figure()
plt.plot(x, y, label="Exact solution", color="black",)
plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
plt.plot(x, y_pred, color="tab:blue", linewidth=4, alpha=0.8, label="PINN prediction")
plt.scatter(x_physics.detach(), 0.9*torch.ones_like(x_physics), color="tab:green", alpha=0.4, 
                    label='Physics loss sampling locations')
plt.legend()
plt.show()  

