import torch
import torch.nn as nn
from models.downsample import Downsample
from models.upsample import Upsample

class ResnetBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3),
            nn.InstanceNorm2d(features),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, kernel_size=3),
            nn.InstanceNorm2d(features)
        )

    def forward(self, x):
        return x + self.model(x)
    
class GeneratorBaseBlock(nn.Module):
    def __init__(self, in_features, out_features, do_upsample=True, do_downsample=False):
        super().__init__()

        self.do_upsample = do_upsample
        self.do_downsample = do_downsample

        if self.do_upsample:
            self.upsample = Upsample(in_features)
        
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1)
        self.instancenorm  = nn.InstanceNorm2d(out_features)
        self.relu = nn.ReLU(True)
        if self.do_downsample:
            self.downsample = Downsample(out_features)

    def forward(self, x):
        if self.do_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.instancenorm(x)
        x = self.relu(x)
        if self.do_downsample:
            x = self.downsample(x)
        return x
    
    def fordward_hook(self, x):
        if self.do_upsample:
            x = self.upsample(x)
        x_hook = self.conv(x)
        x = self.instancenorm(x_hook)
        x = self.relu(x)
        if self.do_downsample:
            x = self.downsample(x)
        return x_hook, x

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64, residuals=9):
        super().__init__()
        self.residuals = residuals

        self.reflectionpad = nn.ReflectionPad2d(3)
        self.block1 = nn.Sequential(
                        nn.Conv2d(in_channels, features, kernel_size=7),
                        nn.InstanceNorm2d(features),
                        nn.ReLU(True)
                        )

        self.downsampleblock2 = GeneratorBaseBlock(features, features * 2, do_upsample=False, do_downsample=True)
        self.downsampleblock3 = GeneratorBaseBlock(features * 2, features * 4, do_upsample=False, do_downsample=True)

        self.resnetblocks4 = nn.Sequential(*[ResnetBlock(features * 4) for _ in range(residuals)])

        self.upsampleblock5 = GeneratorBaseBlock(features * 4, features * 2, do_upsample=True, do_downsample=False)
        self.upsampleblock6 = GeneratorBaseBlock(features * 2, features, do_upsample=True, do_downsample=False)

        self.block7 = nn.Sequential(
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(features, in_channels, kernel_size=7),
                        nn.Tanh(),
                        )

    def append_sample_feature(self, feature, return_ids, return_feats, mlp_id=0, num_patches=256, patch_ids=None):
        B, H, W = feature.shape[0], feature.shape[2], feature.shape[3]
        feature_reshape = feature.permute(0, 2, 3, 1).flatten(1, 2) # B, F, C
        if patch_ids is not None:
            patch_id = patch_ids[mlp_id]
        else:
            patch_id = torch.randperm(feature_reshape.shape[1])
            patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
        x_sample = feature_reshape[:, patch_id, :].flatten(0, 1)

        return_ids.append(patch_id)
        return_feats.append(x_sample)

    def forward(self, x, encode_only=False, num_patches=256, patch_ids=None):
        if not encode_only:
            x = self.reflectionpad(x)
            x = self.block1(x)
            x = self.downsampleblock2(x)
            x = self.downsampleblock3(x)
            x = self.resnetblocks4(x)
            x = self.upsampleblock5(x)
            x = self.upsampleblock6(x)
            x = self.block7(x)
            return x

        else:
            return_ids = []
            return_feats = []
            mlp_id = 0

            x = self.reflectionpad(x)
            self.append_sample_feature(x, return_ids, return_feats, mlp_id=mlp_id, num_patches=num_patches, patch_ids=patch_ids)
            mlp_id += 1

            x = self.block1(x)
            
            x_hook, x = self.downsampleblock2.fordward_hook(x)
            self.append_sample_feature(x_hook, return_ids, return_feats, mlp_id=mlp_id, num_patches=num_patches, patch_ids=patch_ids)
            mlp_id += 1

            x_hook, x = self.downsampleblock3.fordward_hook(x)
            self.append_sample_feature(x_hook, return_ids, return_feats, mlp_id=mlp_id, num_patches=num_patches, patch_ids=patch_ids)
            mlp_id += 1

            for resnet_layer_id, resnet_layer in enumerate(self.resnetblocks4):
                x = resnet_layer(x)
                if resnet_layer_id in [0, 4]:
                    self.append_sample_feature(x, return_ids, return_feats, mlp_id=mlp_id, num_patches=num_patches, patch_ids=patch_ids)
                    mlp_id += 1

            return return_feats, return_ids


if __name__ == "__main__":

    x = torch.randn((5, 3, 256, 256))
    print(x.shape)
    G = Generator()
    x = G(x)
    print(x.shape)

    x = torch.randn((5, 3, 256, 256))
    print(x.shape)
    G = Generator()
    feat_k_pool, sample_ids = G(x, encode_only=True, patch_ids=None)
    feat_q_pool, _ = G(x, encode_only=True, patch_ids=sample_ids)
    print(len(feat_k_pool))
