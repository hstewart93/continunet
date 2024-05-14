"""UNet model for semantic segmentation."""

import torch
import torch.nn as nn


class DoubleConvolutionalBlock(nn.Module):
    """Double convolutional block with batch normalization and ReLU activation."""

    def __init__(self, in_channels, out_channels):
        super(DoubleConvolutionalBlock, self).__init__()
        self.double_convolutional_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_convolutional_block(x)


class DownsamplingBlock(nn.Module):
    """Downsampling block with max pooling and double convolutional block."""

    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DownsamplingBlock, self).__init__()
        self.downsampling_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConvolutionalBlock(in_channels, out_channels),
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, x):
        return self.downsampling_block(x)


class UpsamplingBlock(nn.Module):
    """Upsampling block with transposed convolution and double convolutional block."""

    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(UpsamplingBlock, self).__init__()
        self.upsampling_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2),
            DoubleConvolutionalBlock(in_channels, out_channels),
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, decoder_input, encoder_output):
        x = self.upsampling_block(decoder_input)
        return torch.cat([x, encoder_output], dim=1)


class UNet(nn.Module):
    """UNet model for semantic segmentation."""

    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList(
            [
                DoubleConvolutionalBlock(in_channels, 64),
                DownsamplingBlock(64, 128),
                DownsamplingBlock(128, 256),
                DownsamplingBlock(256, 512),
                DownsamplingBlock(512, 1024),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                UpsamplingBlock(1024, 512),
                UpsamplingBlock(512, 256),
                UpsamplingBlock(256, 128),
                UpsamplingBlock(128, 64),
            ]
        )
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        encoder_outputs = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            encoder_outputs.append(x)
        encoder_outputs.pop()
        for decoder_block in self.decoder:
            x = decoder_block(x, encoder_outputs.pop())
        x = self.output(x)
        return torch.sigmoid(x)
