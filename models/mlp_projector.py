import torch
import torch.nn as nn
from models.downsample import Downsample
from models.upsample import Upsample
from models.generator import Generator

class MLP_Module(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.mlp = nn.Sequential(
            *[
                nn.Linear(input_nc, output_nc),
                nn.ReLU(),
                nn.Linear(output_nc, output_nc)
            ]
        )
    
    def forward(self, x):
        return self.mlp(x)
    

class Projector_Head(nn.Module):
    def __init__(self, in_channels = 3, features = 64, residuals = 9):
        super().__init__()
        self.mlp_1 = MLP_Module(3, 256)
        self.mlp_2 = MLP_Module(128, 256)
        self.mlp_3 = MLP_Module(256, 256)
        self.mlp_4 = MLP_Module(256, 256)
        self.mlp_5 = MLP_Module(256, 256)

    def forward(self, features):
        return_features = []

        for idx, feature in enumerate(feature):
            mlp = getattr(self, f'mlp_{idx+1}')
            feature = mlp(feature)
            norm = feature.pow(2).sum(1, keepdim = True).pow(1/2)
            feature = feature.div(norm + 1e-7)
            return_features.append(feature)

        return return_features
    
if __name__=="__main__":
    x = torch.randn((5, 3, 256, 256))
    print(x.shape)
    G = Generator()
    H = Projector_Head()
    feat_k_pool, sample_ids = G(x, encode_only = True, patch_ids = None)
    feat_q_pool, _ = G(x, encode_only=True, patch_ids = sample_ids)
    print(len(feat_q_pool))
    return_features = H(feat_q_pool)
    print(len(return_features))

