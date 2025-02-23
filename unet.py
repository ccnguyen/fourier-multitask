import torch, sys, pdb
import torch.nn as nn
import citorch

class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, norm_layer, momentum=0.01):
        super().__init__()
        if norm_layer is nn.Identity:
            bias = True
        else:
            bias = False

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias),
            norm_layer(out_ch, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=bias),
            norm_layer(out_ch, momentum=momentum),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class DownsampleBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.block = ConvBlock(in_ch, out_ch, norm_layer=norm_layer)
        self.downsample = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.block(x)
        y = x
        x = self.downsample(x)
        return x, y


class UpsampleBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.block = ConvBlock(in_ch, out_ch, norm_layer=norm_layer)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, y):
        x = self.upsample(x)
        x = torch.cat([x, y], dim=1)
        x = self.block(x)
        return x


class UNet(nn.Module):

    def __init__(self, channels, n_layers: int, norm_layer=None, fourier=False):
        super().__init__()

        # assert len(channels) == n_layers + 2, "The number of layers and the provided channels don't match."

        self.downblocks = nn.ModuleList()
        self.upblocks_seg = nn.ModuleList()

        self.upblocks_dep = nn.ModuleList()
        self.n_layers = n_layers

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        for i in range(n_layers):
            block = DownsampleBlock(channels[i], channels[i + 1], norm_layer)
            self.downblocks.append(block)

        bottom_in = channels[n_layers]
        bottom_out = channels[n_layers + 1]
        self.bottom_block = ConvBlock(bottom_in, bottom_out, norm_layer=norm_layer)

        for i in range(n_layers):
            block = UpsampleBlock(channels[i + 1] + channels[i + 2], channels[i + 1], norm_layer)
            self.upblocks_seg.append(block)
            block = UpsampleBlock(channels[i + 1] + channels[i + 2], channels[i + 1], norm_layer)
            self.upblocks_dep.append(block)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.fourier = fourier


    def forward(self, x):
        features = []
        for i in range(self.n_layers):
            x, y = self.downblocks[i](x)
            features.append(y)
        x = self.bottom_block(x) # [1, 12, 25, 25]

        if self.fourier:
            x = citorch.stack_complex(x, torch.zeros_like(x))
            x = citorch.fft(x)
            x = torch.where(x > -100.0, x, torch.zeros_like(x))
            x = citorch.ifft(x)
            x = x[..., 0]

        dep = x
        seg = x
        for i in range(self.n_layers - 1, -1, -1):
            dep = self.upblocks_dep[i](dep, features[i])
            seg = self.upblocks_seg[i](seg, features[i])
        return seg, dep, x