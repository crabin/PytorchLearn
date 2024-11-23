

import torch

import torch.nn.functional as F
import torch.nn as nn


def batch_norm(is_training, x, gamma, beta, moving_mean, moving_var, eps=1e-5, momentum=0.9):
    if not is_training:
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        mean = x.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

        x_hat = (x - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * x_hat + beta
    return Y, moving_mean, moving_var
    
class BatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm2d, self).__init__()
        shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.register_buffer('moving_mean', torch.zeros(shape))
        self.register_buffer('moving_var', torch.ones(shape))

    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
            
        y, self.moving_mean, self.moving_var = batch_norm(self.training,
            x, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return y


