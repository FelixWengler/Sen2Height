import torch

def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()