# models/predictor/initializer.py
"""
Model Initializer for PrDiMP / DiMP

Implements the filter initialization step described in:
- Bhat et al., "Learning Discriminative Model Prediction" (DiMP), ICCV 2019
- Danelljan et al., "Probabilistic Regression for Visual Tracking" (PrDiMP), CVPR 2020

The initializer takes classification features from template images and
produces the initial convolutional filter w0 of size (1, C, k, k).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelInitializer(nn.Module):
    def __init__(self,
                 in_channels=512,
                 filter_size=4,   # k = 4 (DiMP/PrDiMP default)
                 hidden_channels=None):
        """
        Args:
            in_channels: number of channels entering the initializer (after CLS feature head)
            filter_size: spatial resolution of the learned filter (k x k)
            hidden_channels: optional hidden layer channels
        """
        super().__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        # The learned projection block (1 or 2 conv layers)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # Final 1×1 conv to shape output channel dimension
        self.to_filter = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)

        self.filter_size = filter_size
        self.in_channels = in_channels

    def forward(self, feat_patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_patches: tensor (N, C, H, W)
              These are classification features from template patches.

        Returns:
            w0: initial filter of shape (1, C, k, k)
        """

        # Apply learned projection
        x = self.proj(feat_patches)      # (N, hidden, H, W)
        x = self.to_filter(x)            # (N, C, H, W)

        # Adaptive pool to desired filter size
        x = F.adaptive_avg_pool2d(x, (self.filter_size, self.filter_size))  # (N, C, k, k)

        # Average across samples (DiMP uses multiple template samples)
        w0 = x.mean(dim=0, keepdim=True)  # (1, C, k, k)

        return w0
