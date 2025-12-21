import torch
import torch.nn as nn

# ============================
# 1. U-Net Generator
# ============================
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(UNetGenerator, self).__init__()
        self.encoder = nn.ModuleList([
            self._block(in_channels, features, 4, 2, 1, batch_norm=False),
            self._block(features, features * 2, 4, 2, 1),
            self._block(features * 2, features * 4, 4, 2, 1),
            self._block(features * 4, features * 8, 4, 2, 1),
            self._block(features * 8, features * 8, 4, 2, 1),
            self._block(features * 8, features * 8, 4, 2, 1),
        ])

        self.bottleneck = self._block(features * 8, features * 8, 4, 2, 1, batch_norm=False)

        self.decoder = nn.ModuleList([
            self._upblock(features * 8, features * 8),
            self._upblock(features * 16, features * 8),
            self._upblock(features * 16, features * 8),
            self._upblock(features * 16, features * 4),
            self._upblock(features * 8, features * 2),
            self._upblock(features * 4, features),
        ])

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(features * 2, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.tanh = nn.Tanh()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def _upblock(self, in_channels, out_channels, dropout=0.0):
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for idx, up in enumerate(self.decoder):
            x = up(x)
            x = torch.cat((x, skips[idx]), dim=1)
        return self.tanh(self.final(x))

# ============================
# 2. PatchGAN Discriminator
# ============================
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, features=64):  # 改為1
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            self._block(in_channels * 2, features, 4, 2, 1, batch_norm=False),
            self._block(features, features * 2, 4, 2, 1),
            self._block(features * 2, features * 4, 4, 2, 1),
            self._block(features * 4, features * 8, 4, 1, 1),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, batch_norm=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)  # x, y 都是 [B, 1, H, W]，cat後 [B, 2, H, W]
        return self.model(x)