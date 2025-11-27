# models/backbone/resnet50.py
"""
ResNet-50 backbone module for PrDiMP50.

Usage:
    from models.backbone.resnet50 import ResNet50Backbone
    model = ResNet50Backbone(pretrained=True, out_layer='layer3', train_layers=['layer3','layer4'])
    feats = model(images)  # returns dict with requested layers, and model.get_feature(feats) for single output

Notes:
  - By default returns features from layer3 (stride 16) which is commonly used in DiMP/PrDiMP.
  - You can optionally enable a 1x1 projection neck to change channel dimension.
  - If pretrained=True, weights are taken from torchvision (ImageNet).
  - To see design choices in the PrDiMP paper, refer to the uploaded file:
    /mnt/data/Danelljan_Probabilistic_Regression_for_Visual_Tracking_CVPR_2020_paper.pdf
"""
from typing import Optional, List, Dict
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Backbone(nn.Module):
    def __init__(self,
                 pretrained: bool = True,
                 out_layer: str = 'layer3',
                 train_layers: Optional[List[str]] = None,
                 proj_channels: Optional[int] = None,
                 use_norm: bool = False):
        """
        Args:
            pretrained: whether to load torchvision pretrained weights (ImageNet).
            out_layer: which layer to treat as primary output: 'layer1','layer2','layer3','layer4'
            train_layers: list of layer names to keep trainable. If None -> freeze all except last block.
                          Example: ['layer3','layer4'] to train the deeper layers.
            proj_channels: if set, apply a 1x1 conv to project output channels to this value.
            use_norm: whether to l2-normalize features (optional).
        """
        super().__init__()
        assert out_layer in ('layer1', 'layer2', 'layer3', 'layer4')
        self.out_layer = out_layer
        self.use_norm = use_norm

        # Load torchvision resnet50
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet50(weights=weights)

        # Build the stem and layers as modular blocks
        self.stem = nn.Sequential(
            resnet.conv1,  # 7x7 conv, stride 2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool  # -> /4
        )
        self.layer1 = resnet.layer1  # stride 4 -> output stride 4
        self.layer2 = resnet.layer2  # output stride 8
        self.layer3 = resnet.layer3  # output stride 16 (commonly used)
        self.layer4 = resnet.layer4  # output stride 32

        # optional projection 1x1 conv after the chosen output
        self.proj = None
        if proj_channels is not None:
            # get in_channels from layer3 by default (1024). Use mapping
            in_ch_map = {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
            in_ch = in_ch_map[self.out_layer]
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, proj_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(proj_channels),
                nn.ReLU(inplace=True)
            )

        # set trainable layers
        if train_layers is None:
            # default: freeze everything except layer4
            train_layers = ['layer4']
        # convert to set for membership checks
        train_layers_set = set(train_layers)

        # Freeze or unfreeze parameters
        for name, param in self.named_parameters():
            require_grad = False
            # keep grad if any of train_layers is a prefix in the param name
            for tl in train_layers_set:
                if name.startswith(tl) or name.startswith(f'stem') or name.startswith('layer4'):
                    # allow training of the chosen layers (and keep 'layer4' trainable by default)
                    if tl in name or name.startswith(tl):
                        require_grad = True
            param.requires_grad = require_grad

        # Note: If user wants to fine-tune batchnorm stats set eval mode appropriately in training loop.

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with keys 'layer1','layer2','layer3','layer4' corresponding to feature maps.
        """
        out = {}
        x = self.stem(x)            # down to stride 4
        x = self.layer1(x)
        out['layer1'] = x
        x = self.layer2(x)
        out['layer2'] = x
        x = self.layer3(x)
        out['layer3'] = x
        x = self.layer4(x)
        out['layer4'] = x

        # apply optional projection to the chosen output
        if self.proj is not None:
            out[self.out_layer] = self.proj(out[self.out_layer])

        # optional L2-norm
        if self.use_norm:
            for k, v in out.items():
                # normalization along channel dim
                out[k] = nn.functional.normalize(v, p=2, dim=1)

        return out

    def get_feature(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Convenience: returns the chosen output tensor
        """
        return feats[self.out_layer]

# ----------------------
# Quick unit test
# ----------------------
if __name__ == "__main__":
    # quick sanity check
    model = ResNet50Backbone(pretrained=False, out_layer='layer3', train_layers=['layer3', 'layer4'], proj_channels=None)
    model.eval()
    dummy = torch.randn(2, 3, 320, 320)
    feats = model(dummy)
    for k in ('layer1','layer2','layer3','layer4'):
        print(k, feats[k].shape)
    chosen = model.get_feature(feats)
    print("Chosen output (out_layer=%s) shape: %s" % (model.out_layer, tuple(chosen.shape)))
