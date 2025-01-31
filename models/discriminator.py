import torch
import torch.nn as nn
from models.downsample import Downsample


class DiscriminatorBaseBlock(nn.Module):
    def __init__(self, in_features, out_features, do_downsample=True, do_instancenorm=True):
        super().__init__()

        self.do_downsample = do_downsample
        self.do_instancenorm = do_instancenorm

        self.conv = nn.Conv2d(in_features, out_features, kernel_size=4, stride=1, padding=1)
        self.leakyrelu = nn.LeakyReLU(0.2, True)

        if do_instancenorm:
            self.instancenorm = nn.InstanceNorm2d(out_features)

        if do_downsample:
            self.downsample = Downsample(out_features)
        
    def forward(self, x):
        x = self.conv(x)
        if self.do_instancenorm:
            x = self.instancenorm(x)
        x = self.leakyrelu(x)
        if self.do_downsample:
            x = self.downsample(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.block1 = DiscriminatorBaseBlock(in_channels, features, do_downsample=True, do_instancenorm=False)
        self.block2 = DiscriminatorBaseBlock(features, features * 2, do_downsample=True, do_instancenorm=True)
        self.block3 = DiscriminatorBaseBlock(features * 2, features * 4, do_downsample=True, do_instancenorm=True)
        self.block4 = DiscriminatorBaseBlock(features * 4, features * 8, do_downsample=False, do_instancenorm=True)
        self.conv = nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv(x)
        return x

    def set_requires_grad(self, requires_grad=False):
        for param in self.parameters():
            param.requires_grad = requires_grad

if __name__ == "__main__":
    x = torch.randn((5, 3, 256, 256))
    print(x.shape)
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.shape)