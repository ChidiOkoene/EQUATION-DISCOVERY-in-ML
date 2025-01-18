#!/usr/bin/env python
"""\
Physics-informed neural network example (published on TechShare)

This script is the data-driven reference model.

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
    
# train standard neural network
torch.manual_seed(123)
model = FCN()
optimizer = torch.optim.Adam(model.parameters(),lr=5e-4)

losses = []
for i in range(2000):
    optimizer.zero_grad()
    y_pred = model(x_data)
    
    loss = torch.mean((y_pred-y_data)**2) # mean squared error
    loss.backward()
    optimizer.step()
    losses.append(loss)
    
    # plot the result as training progresses
    if i % 500 == 0: print(str(i).ljust(5), " Loss: ",loss.item())

# plot training loss history
fig = plt.figure(constrained_layout=False, figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)
h = ax.plot(losses, label = "Data loss")
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
plt.plot(x, y_pred, color="tab:blue", linewidth=4, alpha=0.8, label="Neural network prediction")
plt.legend()
plt.show()
