import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

from matplotlib import cm
import torch



def Cheng_Knorr_Sonnerdrucker():
    # Problem parameters
    inter_type = 'linear'  # Change to 'spline' for cubic spline scheme
    N = 10
    M = N
    k = 0.5
    alpha = 0.05
    vmax = 2 * np.pi
    CFL = 3.8
    t_end = 15.0
    L = 2 * np.pi / k

    # Grid definition
    #x = np.linspace(0, L, N)
    x = torch.linspace(0, L, N)
    #v = np.linspace(-vmax, vmax, M)
    v = torch.linspace(-vmax, vmax, M)
    dx = x[1] - x[0]
    dv = v[1] - v[0]
    X, V = np.meshgrid(x.numpy(), v.numpy())
    X = X.T
    V = V.T

    # Initial conditions
    f = np.exp(-V ** 2 / 2) / np.sqrt(2 * np.pi) * (1.0 + alpha * np.cos(k * X)) * V ** 2
    f = torch.from_numpy(f)
    plt.figure()
    #plt.surf(X, V, f)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, V, f.numpy(), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf)
    plt.savefig('initial.png')

    # Apply periodic borders and zero at |v| beyond vmax
    f[-1, :] = f[1, :]
    f[0, :] = f[-2, :]
    f[0, :] = 0
    f[-1, :] = 0

    # Open maximized figure window
    plt.figure(figsize=(10, 10))
    plt.ion()

    # Preallocate memory
    E = torch.zeros_like(x)

    # Start main calculation procedure
    time = 0
    while time < t_end:
        # Estimate time step using CFL condition
        dt = CFL / (vmax / dx + torch.max(torch.abs(E)) / dv)
        if dt > t_end - time:
            dt = t_end - time

        # Plot EDF f(x,v,t) in phase space
        plt.pcolor(x[1:-1], v[1:-1], f[1:-1, 1:-1].T)
        plt.colorbar()
        plt.axis('square')
        plt.draw()
        plt.pause(0.01)

        # X-coordinate shift at half time step
        x_SHIFT = X - V * 0.5 * dt
        x_SHIFT = np.where(x_SHIFT <= 0, x_SHIFT + L, x_SHIFT)
        x_SHIFT = np.where(x_SHIFT >= L, x_SHIFT - L, x_SHIFT)

        interp_func = interp2d(X.flatten(), V.flatten(), f.flatten(), kind=inter_type)
        f = interp_func(x_SHIFT.flatten(), V.flatten()).T

        # Apply periodic boundaries in X-coordinate
        f[N - 1, :] = f[1, :]
        f[0, :] = f[N - 2, :]

        # Electrical field strength from exact solution of Poisson's equation
        E = np.cumtrapz(np.trapz(f, v, axis=1), x) - x
        E -= np.mean(E)
        E = np.zeros_like(E)

        # V-coordinate shift at full time step
        Vsh = V - E[:, np.newaxis] * dt
        Vsh = np.where((Vsh < vmax) & (Vsh >= -vmax), Vsh, 0)

        interp_func = interp2d(X.flatten(), V.flatten(), f.flatten(), kind=inter_type)
        f = interp_func(X.flatten(), Vsh.flatten()).T

        # Boundary conditions
        f[:, 0] = 0
        f[:, -1] = 0
        f[-1, :] = f[1, :]
        f[0, :] = f[-2, :]

        # X-coordinate shift at half time step
        x_SHIFT = X - V * 0.5 * dt
        x_SHIFT = np.where(x_SHIFT <= 0, x_SHIFT + L, x_SHIFT)
        x_SHIFT = np.where(x_SHIFT >= L, x_SHIFT - L, x_SHIFT)

        interp_func = interp2d(X.flatten(), V.flatten(), f.flatten(), kind=inter_type)
        f = interp_func(x_SHIFT.flatten(), V.flatten()).T

        # Apply periodic boundaries in X-coordinate
        f[N - 1, :] = f[1, :]
        f[0, :] = f[N - 2, :]

        # Next time step
        time += dt

    # Final EDF plot
    plt.pcolor(x[1:-1], v[1:-1], f[1:-1, 1:-1].T)
    plt.colorbar()
    plt.title('Two stream instability', fontsize=18)
    plt.axis('square')
    plt.show()


Cheng_Knorr_Sonnerdrucker()