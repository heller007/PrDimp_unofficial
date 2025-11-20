# models/heads/cls_feature_head.py
"""
Classification Feature Head for PrDiMP / DiMP
--------------------------------------------
Implements the shallow feature projection layer f_θ described in:

- Bhat et al., DiMP (ICCV 2019)
- Danelljan et al., PrDiMP (CVPR 2020)

This module transforms backbone features into classification features
used by the initializer, optimizer, and the convolutional model filter.

Requirements from paper:
- Convolutional feature projection
- No spatial resolution change
- Produces more discriminative features for classification
"""

import torch
import torch.nn as nn

def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)

class ClassificationFeatureHead(nn.Module):
    def __init__(self, in_channels=1024, out_channels=512, use_bn=True):
        """
        Args:
            in_channels: channels coming from backbone (layer3 = 1024 for ResNet50)
            out_channels: channels used by the classifier filter (typically 256–512)
            use_bn: whether to use BatchNorm (helps optimization stability)
        """
        super().__init__()

        layers = []
        layers.append(conv3x3(in_channels, out_channels))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(conv3x3(out_channels, out_channels))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.head(x)
