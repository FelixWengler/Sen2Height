import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.InstanceNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.InstanceNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

        # Safe default: keep attention soft at start
        self.psi[0].bias.data.fill_(0.1)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Optional: residual skip to prevent catastrophic suppression
        return x * psi + x


class Sentinel2UNet(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Standard encoder (3x3)
        self.enc1a = UNetBlock(in_channels, 64, kernel_size=3)
        self.enc2a = UNetBlock(64, 128, kernel_size=3)
        self.enc3a = UNetBlock(128, 256, kernel_size=3)

        # Large-kernel encoder (7x7)
        self.enc1b = UNetBlock(in_channels, 64, kernel_size=7)
        self.enc2b = UNetBlock(64, 128, kernel_size=7)
        self.enc3b = UNetBlock(128, 256, kernel_size=7)

        # Bottleneck
        self.bottleneck = UNetBlock(256, 512)

        # Attention gates
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64)

        # Output layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Parallel encoders
        e1a = self.enc1a(x)
        e1b = self.enc1b(x)
        e1 = e1a + e1b

        e2a = self.enc2a(self.pool(e1a))
        e2b = self.enc2b(self.pool(e1b))
        e2 = e2a + e2b

        e3a = self.enc3a(self.pool(e2a))
        e3b = self.enc3b(self.pool(e2b))
        e3 = e3a + e3b

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder with attention
        d3 = self.up3(b)
        e3 = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)
