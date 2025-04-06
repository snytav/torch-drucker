import torch
from torch.autograd.functional import jacobian
from NN import PDEnet3D

def loss_pde(f,f1,device,t,x,v):
    df = torch.subtract(f1 - f)
    lf = 0.0
    for i,ti in enumerate(t):
        for j,xi in enumerate(x):
            for k,vi in enumerate(v):
                input_point = torch.Tensor([ti,xi, yi]).double()
                input_point.requires_grad_()

                lf += torch.power()

def pde_solve(f,f1,device,t,x,v):

    model = PDEnet3D(50)
    jac = jacobian(model.forward, input_point, create_graph=True)