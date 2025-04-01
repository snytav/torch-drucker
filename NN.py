import torch
import torch.nn as nn
# нейронная сеть для решения уравнения Пуассона
class PDEnet(nn.Module):
    def __init__(self,N):
        super(PDEnet,self).__init__()
        self.N = N
        fc1 = nn.Linear(2,self.N) # первый слой

        fc2 = nn.Linear(self.N, 1) # второй слой
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self,x):
        x = x.reshape(1, 2)
        y = self.fc1(x)
        y = torch.sigmoid(y)
        y = self.fc2(y.reshape(1, self.N))
        return y
