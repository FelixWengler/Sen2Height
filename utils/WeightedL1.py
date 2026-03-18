import torch
import torch.nn as nn

class BinWeightedL1(nn.Module):
    def __init__(self, edges=(0, 5, 10, 15, 22, 30, 35, 40, 60),
        weights=(0.8, 0.8, 0.916, 0.8, 0.8, 1.134, 1.619, 1.939)):
        super().__init__()
        self.register_buffer("edges", torch.tensor(edges, dtype=torch.float32))
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred, target):
        # target: [B,1,H,W]
        bin_idx = torch.bucketize(target, self.edges, right=False) - 1
        bin_idx = torch.clamp(bin_idx, 0, self.weights.numel() - 1)
        w = self.weights[bin_idx]
        return (w * (pred - target).abs()).mean()

#    def __init__(self, edges=(0, 5, 10, 15, 22, 30, 35, 40, 60),
                 #weights=(0.8, 0.8, 0.916, 0.8, 0.8, 1.134, 1.619, 1.939)):