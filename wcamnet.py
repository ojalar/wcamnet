import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import SqueezeExcitation

class SEBlock(nn.Module):
    # Implementation of a Standard Residual SquuezeExcitation block
    def __init__(self, in_features, out_features, squeeze, downsample = True):
        super().__init__()
        # set stride based on downsampling
        stride = 2 if downsample else 1

        # if modification is output size is needed
        if in_features != out_features or downsample:
            self.ds = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride = stride),
                nn.BatchNorm2d(out_features)
                )    
        else:
            self.ds = None

        # configure the CNN parts of the block
        self.rb = nn.Sequential(
                nn.Conv2d(in_features, out_features, 3, stride = stride, padding = 1),
                nn.BatchNorm2d(out_features),
                nn.ReLU(),
                nn.Conv2d(out_features, out_features, 3, padding = 1),
                nn.BatchNorm2d(out_features)
                )
        # configure the SqueezeExcitation layer
        self.se = SqueezeExcitation(in_features, in_features // squeeze)
        self.relu = nn.ReLU() 

    def forward(self, x):
        if self.ds is None:
            identity = x
        else:
            identity = self.ds(x)
        x = self.rb(x)
        x = self.se(x)
        x += identity
        x = self.relu(x)

        return x


class WCAMNet(nn.Module):
    # Implementation of the WCamNet architecture
    def __init__(self, grad = False, squeeze = 8, fc2 = 1024, fc3 = 1024):
        super().__init__()
        # initialise DINOv2 backbone and set gradients based on init config
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        for param in self.dinov2.parameters():
            param.requires_grad = grad
        
        # initialise HD branch
        f1 = 64
        f2 = 128
        f3 = 256
        fc1 = f3 + 768
        self.hd_branch = self._make_hd_block(f1, f2, f3)
       
        # initialise SqueezeExcitation blocks
        self.se1 = SEBlock(fc1, fc2, squeeze)
        self.se2 = SEBlock(fc2, fc3, squeeze)
        
        # average pooling and linear layer for final output
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(fc3, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # features from HD branch
        hd_features = self.hd_branch(x)
        # patch tokens from DINOv2
        dinov2_res = self.dinov2.forward_features(x)
        bs = x.size()[0]
        dim = x.size()[-1] // 14
        dinov2_features = dinov2_res["x_norm_patchtokens"]
        dinov2_features = torch.transpose(dinov2_features, 1, 2)
        dinov2_features = dinov2_features.view((bs, 768, dim, dim))
        # combine HD branch and DINOv2 features
        features = torch.cat((hd_features, dinov2_features), 1)
        # processing through SqueezeExcitation blocks
        features = self.se1(features)
        features = self.se2(features)
        # average pooling, linear layer and sigmoid activation
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        pred = self.linear(features)
        return self.sigmoid(pred)

    def _make_hd_block(self, f1, f2, f3):
        # method for creating the HD branch based on the chosen feature sizes
        block = nn.Sequential(
                nn.Conv2d(3, f1, 7, padding=3, stride=2),
                nn.BatchNorm2d(f1),
                nn.ReLU(),
                nn.Conv2d(f1, f2, 3, padding=1),
                nn.BatchNorm2d(f2),
                nn.ReLU(),
                nn.Conv2d(f2, f3, 3, padding=1),
                nn.BatchNorm2d(f3),
                nn.ReLU(),
                nn.MaxPool2d(7, stride=7)
                )

        return block

