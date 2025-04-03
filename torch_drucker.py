import numpy as np
import matplotlib.pyplot as plt
import torch
from draw_surf import surf,contour

#https://discuss.pytorch.org/t/implementation-of-function-like-numpy-roll/964/6
def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)

def timestep(x,v,f_in,T,N,M,dt,dx,dv):
    f = torch.clone(f_in)
    # Plot EDF f(x,v,t) in phase space
    if T % 100 == -1:
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
        f_temp = roll(f[:, J], SGN.int() * II, 0)
        I = torch.arange(1, N - 1).to(f_temp.device)
        Dxf = torch.zeros(N).to(f_temp.device)
        Dxf[1:-1] = SHIFT * (f_temp[1:-1] + SGN * (f_temp[2:] - f_temp[:-2]) * (1.0 - SHIFT) / 4.0)

        # Apply periodic border conditions for Dxf
        Dxf[-1] = Dxf[1]
        Dxf[0] = Dxf[-2]

        # New distribution function after shift
        f[1:-1, J] = f_temp[1:-1] + Dxf[I - SGN.int()] - Dxf[I]

    # Apply periodic boundaries in X-coordinate
    f[-1, :] = f[1, :]
    f[0, :] = f[-2, :]

    # Electrical field strength from exact solution of Poisson's equation
    rho = torch.trapz(f, v, axis=1)
    E1 = torch.cumulative_trapezoid(rho, x)
    E1 = torch.cat((torch.zeros(1).to(E1.device), E1), dim=0)
    E = E1 - x
    E -= torch.mean(E)

    # V-coordinate shift at full time step
    for I in range(1, N - 1):
        SGN = torch.sign(E[I])
        SHIFT = abs(E[I]) * dt / dv
        JJ = int(SHIFT)
        SHIFT -= JJ

        f_temp = torch.zeros(M)
        if SGN > 0:
            f_temp[0:JJ] = 0.0
            f_temp[JJ:M] = f[I, 0:(M - JJ)]
        else:
            f_temp[0] = 0.0
            f_temp[1:(M - JJ)] = f[I, (JJ + 1):M]
            f_temp[(M - JJ):M] = 0.0

        J = range(1, M - 1)
        Dvf = torch.zeros(M)
        Dvf[1:-1] = SHIFT * (f_temp[1:-1] + SGN * (f_temp[2:] - f_temp[:-2]) * (1.0 - SHIFT))

    return f

def loss(model,x,v,f):
    lf = 0.0
    for i,xi in enumerate(x):
        for j,vi in enumerate(v):
            xt = torch.cat((xi.reshape(1),vi.reshape(1)))
            y = model(xt)
            lf += torch.abs(f[i,j] - y)
    return lf

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
    X, Y = np.meshgrid(x,v)


    # surf(X,Y,f,'Initial distribution')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    f  = torch.from_numpy(f)
    f  = f.to(device)
    v  = v.to(device)
    x  = x.to(device)
    dx = dx.to(device)



    # Apply periodic values at ghost nodes
    f[-1, :] = f[1, :]
    f[0, :] = f[-2, :]

    # Open maximized figure window
    #plt.figure(figsize=(12, 8))

    # Start main calculation procedure
    T = 0
    f.requires_grad=True
    from NN import PDEnet
    model = PDEnet(50)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)

    if device == torch.device('cuda'):
       model = model.cuda()


    hist = np.zeros(N_steps)

    while T <= N_steps:
        optimizer.zero_grad()
        f1 = timestep(x,v,f,T,N,M,dt,dx,dv)


        if T % 10 == 0:
            contour(X,Y,f.detach().numpy(),'t = '+ str(T*dt) )

        T += 1
        df = torch.max(torch.abs(f1-f))
        f = f1
        lf = loss(model,x,v,f)
        lf.backward(retain_graph=True)
        optimizer.step()
        print(T,lf.item())
        hist[T] = lf.item()
    plt.figure()
    plt.plot(np.linspace(),hist)
    plt.savefig('train_history.png')



    np.savetxt('v_final.txt',f.numpy(),'%25.15e')


Vlasov_Poisson_Landau_damping()