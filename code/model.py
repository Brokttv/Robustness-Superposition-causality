import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class Model(nn.Module):
    def __init__(self, n: int, m: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n, m) * 0.01)
        self.b = nn.Parameter(torch.zeros(m))
 
    def forward(self, x):
        encoder = x @ self.w.T      # (batch, m) -> (batch, n)
        return F.relu(encoder @ self.w + self.b)  # (batch, n) -> (batch, m)
