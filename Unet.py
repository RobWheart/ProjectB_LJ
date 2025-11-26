import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 3D UNet for Partial Volume Change Regression  (-1 ~ +1)
# ============================================================
class DoubleConv(nn.Module):
    """(Conv3D → BN → ReLU) × 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool3d + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Upsampling block: ConvTranspose3d + concat + DoubleConv"""
    def __init__(self, in_channels, skip_channels, out_channels):
        """
        in_channels: number of channels from previous decoder layer
        skip_channels: number of channels from encoder skip connection
        out_channels: number of channels after convolution
        """
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(skip_channels + out_channels, out_channels)

    def forward(self, x1, x2):
        # Upsample
        x1 = self.up(x1)
        # Pad if needed
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2,
            diffZ // 2, diffZ - diffZ // 2
        ])
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1×1×1 convolution to map to desired output channels"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Standard 3D U-Net for regression of voxel-wise partial volume change
    Input:  [B, 2, D, H, W]  (before/after)
    Output: [B, 1, D, H, W]  (change map, range [-1, 1])
    """
    def __init__(self, in_channels=2, out_channels=1, base_c=32):
        super().__init__()
        # Encoder
        self.inc   = DoubleConv(in_channels, base_c)         #  2 →  32
        self.down1 = Down(base_c, base_c * 2)                # 32 →  64
        self.down2 = Down(base_c * 2, base_c * 4)            # 64 → 128
        self.down3 = Down(base_c * 4, base_c * 8)            #128 → 256
        self.down4 = Down(base_c * 8, base_c * 8)            #256 → 256

        # Decoder
        self.up1 = Up(base_c * 8, base_c * 8, base_c * 4)    # 256,256→128
        self.up2 = Up(base_c * 4, base_c * 4, base_c * 2)    # 128,128→64
        self.up3 = Up(base_c * 2, base_c * 2, base_c)        # 64,64→32
        self.up4 = Up(base_c,     base_c,     base_c)        # 32,32→32

        # Output head + activation
        self.outc = OutConv(base_c, out_channels)            # 32 → 1
        self.activation = nn.Tanh()                          # map to [-1, 1]

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final output (regression map)
        x = self.outc(x)
        x = self.activation(x)
        return x


