import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.flatten(x, self.dim)
        return x
