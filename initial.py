import numpy as np
import torch

def initial_f():
    N = 64
    M = 127
    k = 0.3
    alpha = 0.3
    vmax = 8.0
    dt = 0.1
    t_end = 100.0
    L = 2 * np.pi / k

    # Grid definition
    x = torch.linspace(0, L, N - 2)
    v = torch.linspace(-vmax, vmax, M)
    dx = x[1] - x[0]
    dv = v[1] - v[0]

    # Calculate number of time steps
    N_steps = round(t_end / dt)

    # Add ghost nodes in X
    dx = dx.reshape(1)
    L = L*torch.ones(1)
    x = torch.cat((-dx, x, L + dx))
    X, V = np.meshgrid(x, v)
    X = X.T
    V = V.T

    # Initial conditions - Maxwellian in V and perturbed in X
    f = np.exp(-V ** 2 / 2) / np.sqrt(2 * np.pi) * (1.0 + alpha * np.cos(k * X))
    return x, v, f, N, M, dt, dx, dv
