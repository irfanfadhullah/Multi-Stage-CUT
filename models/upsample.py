import torch
import torch.nn as nn

class Upsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.ConvTranspose2d(features, features, kernel_size=4, stride=2, padding=3)
        ]
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)