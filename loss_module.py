import torch
from torch.autograd.functional import jacobian



def loss_pde(df,device,t,x,v,model):
    # df = torch.subtract(f1,f)
    # df = df.reshape(t.shape[0],df.shape[0],df.shape[1])
    lf = 0.0
    for i,ti in enumerate(t):
        for j,xi in enumerate(x):
            for k,vi in enumerate(v):
                input_point = torch.Tensor([ti,xi, vi]).to(device)
                input_point.requires_grad_()
                jac = jacobian(model.forward, input_point, create_graph=True)
                df_dt = jac[0][0][0]
                qq = 0
                lf += torch.pow(df_dt - df[i][j][k],2.0)
    return lf


if __name__ == '__main__':
    from pinn_pde_solve import pde_solve
    from initial import initial_f
    x, v, f, N, M, dt, dx, dv = initial_f()
    f = torch.from_numpy(f)

    from time_module import timestep
    T = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = f.to(device)
    v = v.to(device)
    x = x.to(device)
    dx = dx.to(device)
    from NN import PDEnet3D
    model = PDEnet3D(100)
    model.to(device)
    f1 = timestep(x, v, f, T, N, M, dt, dx, dv)
    df = torch.subtract(f1,f)
    df = df.reshape(1, df.shape[0], df.shape[1])

    f_NN = pde_solve(df,f1,device,T*torch.ones(1),x,v,model)

    #f_out = timestep(x, v, f_in, T, N, M, dt, dx, dv)