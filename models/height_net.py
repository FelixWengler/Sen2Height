import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # For residual connection if channel dims change
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        identity = self.residual_conv(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Residual connection
        return F.relu(out)

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
        self.psi[0].bias.data.fill_(0.1)  # Soft start

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi + x  # Residual skip

class Sentinel2ResUNet(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Multiscale encoders
        self.enc1a = ResidualUNetBlock(in_channels, 64, kernel_size=3)
        self.enc1b = ResidualUNetBlock(in_channels, 64, kernel_size=7)
        self.enc2a = ResidualUNetBlock(64, 128, kernel_size=3)
        self.enc2b = ResidualUNetBlock(64, 128, kernel_size=7)
        self.enc3a = ResidualUNetBlock(128, 256, kernel_size=3)
        self.enc3b = ResidualUNetBlock(128, 256, kernel_size=7)

        # Bottleneck
        self.bottleneck = ResidualUNetBlock(256, 512)

        # Attention gates
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualUNetBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualUNetBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualUNetBlock(128, 64)

        # Output
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Multiscale encoders
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
