import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    # L: two convolutions instead of one can allow the network to learn
    # more complex features without changing resolution.

    def __init__(self, in_channels, out_channels, mid_channels=None, activation='relu'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if activation == 'relu':
            activation = nn.ReLU(inplace=True)
        if activation == 'gelu':
            activation = nn.GELU()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            activation,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    # L: (encoder step)

    def __init__(self, in_channels, out_channels, scale_factor=2, activation='relu'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(scale_factor),
            DoubleConv(in_channels, out_channels, activation=activation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""
    # L: (decoder step)

    def __init__(self, in_channels, out_channels, scale_factor=2, activation='relu'):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels, activation=activation)

    def forward(self, x):
        out = self.up(x)
                        
        return self.conv(out)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    """
    L:
    Feature encoder: transforms into a compact representation
    """
    def __init__(self, in_channels=1, channels_latent=64, activation='relu'):
        super().__init__()
        
        self.inc = DoubleConv(in_channels, channels_latent, activation=activation) #maps raw input to latent space
        self.down1 = Down(channels_latent, 2*channels_latent, scale_factor=2, activation=activation) #reduce size x2
        self.down2 = Down(2*channels_latent, 4*channels_latent, scale_factor=2, activation=activation) #reduce size x2 again
        self.down3 = Down(4*channels_latent, 8*channels_latent, scale_factor=2, activation=activation) #reduce size x2 again
        
    def forward(self, image):        
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        return x4 #returns very compressed representation of the input
    
class Decoder(nn.Module):
    """
    L:
    Decoder: maps latent space back to physical space 
    """
    def __init__(self, channels_latent, out_channels, activation='relu'):
        super().__init__()
                
        self.up1 = Up(8*channels_latent, 4*channels_latent, scale_factor=2, activation=activation) #from latent to higher resolution
        self.up2 = Up(4*channels_latent, 2*channels_latent, scale_factor=2, activation=activation) #higher again
        self.up3 = Up(2*channels_latent, channels_latent, scale_factor=2, activation=activation) #restore original resolution
        self.outc = OutConv(channels_latent, out_channels) #project to desired num. of output channels

    def forward(self, z):
        x = self.up1(z)
        x = self.up2(x)
        x = self.up3(x)
        out = self.outc(x)

        return out
    
if (__name__ == '__main__'):
    
    
    x = torch.zeros((10, 12, 8, 8)) #batch=10, 12 input channels, 8x8 spatial
            
    enc = Encoder(in_channels=12, channels_latent=64, activation='gelu')
    dec = Decoder(channels_latent=64, out_channels=12, activation='gelu')

    z = enc(x) #latent represenation
    y = dec(z) #reconstruction

    print(z.shape) #(10, 512, 1, 1)
    print(y.shape) #(10, 12, 8, 8)
    