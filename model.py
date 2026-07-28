"""
model.py
Standalone 2D U-Net for binary segmentation of calvarial defect regions.
Input : (B, 1, H, W) single-channel CT image, values in [0, 1]
Output: (B, 1, H, W) raw logits (apply sigmoid for probability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net with configurable feature sizes.
    Default features=(32, 64, 128, 256) → ~7.8M params, efficient on Apple MPS.
    Use features=(64, 128, 256, 512) for a larger ~31M-param model.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: tuple = (32, 64, 128, 256)):
        super().__init__()

        f = features
        self.inc    = DoubleConv(in_channels, f[0])
        self.down1  = Down(f[0], f[1])
        self.down2  = Down(f[1], f[2])
        self.down3  = Down(f[2], f[3])
        self.bottom = Down(f[3], f[3] * 2)

        self.up1    = Up(f[3] * 2, f[3])
        self.up2    = Up(f[3],     f[2])
        self.up3    = Up(f[2],     f[1])
        self.up4    = Up(f[1],     f[0])
        self.outc   = nn.Conv2d(f[0], out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bottom(x4)

        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss. logits are raw (no sigmoid applied yet)."""
    probs   = torch.sigmoid(logits)
    probs   = probs.view(-1)
    targets = targets.view(-1).float()
    intersection = (probs * targets).sum()
    return 1.0 - (2.0 * intersection + eps) / (probs.sum() + targets.sum() + eps)


def focal_bce_loss(logits: torch.Tensor, targets: torch.Tensor,
                   gamma: float = 2.0, alpha: float = 0.75) -> torch.Tensor:
    """
    Focal binary cross-entropy.
    alpha weights the positive class (set > 0.5 for foreground that is rare).
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction='none')
    pt  = torch.exp(-bce)
    weight = alpha * targets.float() + (1 - alpha) * (1 - targets.float())
    focal  = weight * (1 - pt) ** gamma * bce
    return focal.mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor,
                  dice_w: float = 0.5, focal_w: float = 0.5) -> torch.Tensor:
    return dice_w * dice_loss(logits, targets) + focal_w * focal_bce_loss(logits, targets)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = UNet()
    print(f'UNet parameters: {count_parameters(model):,}')
    x = torch.randn(2, 1, 512, 512)
    y = model(x)
    print(f'Input : {x.shape}  →  Output : {y.shape}')
