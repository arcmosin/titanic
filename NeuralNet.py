import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self,input_dim):
        super(NeuralNet,self).__init__()

        self.net=nn.Sequential(
            nn.Linear(input_dim,2),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        self.criterion=nn.BCELoss(size_average=True)

    def forward(self,x):
        return self.net(x).squeeze(1)

    def cal_loss(self,pred,label):
        return self.criterion(pred,label)
