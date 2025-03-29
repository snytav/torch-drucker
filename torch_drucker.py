import numpy as np
import matplotlib.pyplot as plt
import torch


def Vlasov_Poisson_Landau_damping():
    # Define problem parameters
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
    f = torch.from_numpy(f)


    # Apply periodic values at ghost nodes
    f[-1, :] = f[1, :]
    f[0, :] = f[-2, :]

    # Open maximized figure window
    plt.figure(figsize=(12, 8))

    # Start main calculation procedure
    T = 0
    while T <= N_steps:
        # Plot EDF f(x,v,t) in phase space
        if T % 100 == 0:
            plt.pcolormesh(x[1:-1], v[1:-1], f[1:-1, 1:-1].T, shading='auto', cmap='hot')
            plt.axis('square')
            plt.clim(0, 0.0001)
            plt.colorbar()
            plt.title(f'Nonlinear Landau damping, t = {T * dt:.2f}', fontsize=18)
            plt.draw()
            plt.pause(0.01)

        # X-coordinate shift at half time step
        for J in range(1, M - 1):
            SGN = torch.sign(v[J])
            SHIFT = torch.abs(v[J]) * 0.5 * dt / dx
            II = int(SHIFT)
            SHIFT -= II

            # Make circular shift
            f_temp = np.roll(f[:, J], SGN.astype(int) * II)
            I = np.arange(1, N-1)
            Dxf = np.zeros(N)
            Dxf[1:-1] = SHIFT * (f_temp[1:-1] + SGN * (f_temp[2:] - f_temp[:-2]) * (1.0 - SHIFT) / 4.0)

            # Apply periodic border conditions for Dxf
            Dxf[-1] = Dxf[1]
            Dxf[0] = Dxf[-2]

            # New distribution function after shift
            f[1:-1, J] = f_temp[1:-1] + Dxf[I - SGN.astype(int)] - Dxf[I]

        # Apply periodic boundaries in X-coordinate
        f[-1, :] = f[1, :]
        f[0, :] = f[-2, :]

        # Electrical field strength from exact solution of Poisson's equation
        rho = torch.trapz(torch.from_numpy(f),
            torch.from_numpy(v), axis=1)
        E1 = torch.cumulative_trapezoid(rho,
            torch.from_numpy(x))
        E1 = torch.cat((torch.zeros(1), E1), dim=0)
        E  = E1 - torch.from_numpy(x)
        E -= torch.mean(E)

        # V-coordinate shift at full time step
        for I in range(1, N - 1):
            SGN = np.sign(E[I])
            SHIFT = abs(E[I]) * dt / dv
            JJ = int(SHIFT)
            SHIFT -= JJ

            f_temp = np.zeros(M)
            if SGN > 0:
                f_temp[0:JJ] = 0.0
                f_temp[JJ:M] = f[I, 0:(M - JJ)]
            else:
                f_temp[0] = 0.0
                f_temp[1:(M - JJ)] = f[I, (JJ + 1):M]
                f_temp[(M - JJ):M] = 0.0

            J = range(1, M - 1)
            Dvf = np.zeros(M)
            Dvf[1:-1] = SHIFT * (f_temp[1:-1] + SGN * (f_temp[2:] - f_temp[:-2]) * (1.0 - SHIFT))

        T += 1


Vlasov_Poisson_Landau_damping()