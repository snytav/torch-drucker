import torch
from torch.autograd.functional import jacobian
from NN import PDEnet3D

def loss_pde(f,f1,device,t,x,v,model):
    df = torch.subtract(f1,f)
    df = df.reshape(t.shape[0],df.shape[0],df.shape[1])
    lf = 0.0
    for i,ti in enumerate(t):
        for j,xi in enumerate(x):
            for k,vi in enumerate(v):
                input_point = torch.Tensor([ti,xi, vi])
                input_point.requires_grad_()
                jac = jacobian(model.forward, input_point, create_graph=True)
                df_dt = jac[0][0][0]
                qq = 0
                lf += torch.pow(df_dt - df[i][j][k],2.0)
    return lf

def pde_solve(f,f1,device,t,x,v):

    model = PDEnet3D(50)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    lf = loss_pde(f,f1,device,t,x,v,model)
    n = 0
    while lf.item() > 0.1:
        optimizer.zero_grad()
        lf = loss_pde(f, f1, device, t, x, v, model)
        lf.backward(retain_graph=True)
        optimizer.step()

        print(n,lf.item())
        n = n + 1


