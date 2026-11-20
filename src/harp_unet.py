from __future__ import annotations

import torch
from torch import nn


class DSConv(nn.Module):
    """Depthwise-separable convolution used by the lightweight decoder."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class MSCAG(nn.Module):
    """Multi-scale channel attention gate for encoder features."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.dilated = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.standard = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        multi_scale = self.dilated(inputs) + self.standard(inputs) + self.pointwise(inputs)
        return multi_scale * self.channel_attention(multi_scale)


class EncoderBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.residual = nn.Conv2d(input_channels, output_channels, 1, bias=False) if input_channels != output_channels else nn.Identity()
        self.dsconv = DSConv(input_channels, output_channels)
        self.activation = nn.ReLU(inplace=True)
        self.attention = MSCAG(output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.activation(self.residual(inputs) + self.dsconv(inputs))
        return self.attention(features)


class HARPUNet(nn.Module):
    """Lightweight 256x256 binary pothole segmentation network."""

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.pool = nn.MaxPool2d(2)
        self.enc1 = EncoderBlock(3, channels[0])
        self.enc2 = EncoderBlock(channels[0], channels[1])
        self.enc3 = EncoderBlock(channels[1], channels[2])
        self.enc4 = EncoderBlock(channels[2], channels[3])
        self.bottleneck = DSConv(channels[3], channels[3] * 2)
        self.up4 = nn.ConvTranspose2d(channels[3] * 2, channels[3], 2, stride=2)
        self.dec4 = DSConv(channels[3] * 2, channels[3])
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
        self.dec3 = DSConv(channels[2] * 2, channels[2])
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
        self.dec2 = DSConv(channels[1] * 2, channels[1])
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
        self.dec1 = DSConv(channels[0] * 2, channels[0])
        self.mask_head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        dec4 = self.dec4(torch.cat((self.up4(bottleneck), enc4), dim=1))
        dec3 = self.dec3(torch.cat((self.up3(dec4), enc3), dim=1))
        dec2 = self.dec2(torch.cat((self.up2(dec3), enc2), dim=1))
        dec1 = self.dec1(torch.cat((self.up1(dec2), enc1), dim=1))
        return self.mask_head(dec1)
