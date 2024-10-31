
import numpy as np

def gen_data1(n_samples=100, x0=3, y0=1/2, dt=0.1):
    # Initialize time vector
    t = np.linspace(0, dt * (n_samples - 1), n_samples)
    
    # Initialize state arrays
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    
    # Set initial conditions
    x[0] = x0
    y[0] = y0
    
    # Simulate the system using Euler's method
    for i in range(1, n_samples):
        dxdt = -2 * x[i - 1]  # dx/dt = -2x
        dydt = y[i - 1]       # dy/dt = y
        
        # Update states
        x[i] = x[i - 1] + dxdt * dt
        y[i] = y[i - 1] + dydt * dt

    return t, x, y
