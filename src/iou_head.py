# models/heads/iou_head.py
"""
IoU Head (IoU-Net style) for PrDiMP / BBR scoring.

Usage:
    iou_head = IoUHead(feat_dim=512, pooled_size=(7,7), hidden_dim=256)
    # feature_map: (B, C, Hf, Wf)  -- features for the batch of images (search patches)
    # boxes: either (B, K, 4) in center format (cx,cy,w,h) or (B, 4) for one box per image
    scores = iou_head(feature_map, boxes, image_size=(H_img, W_img))
    # scores shape: (B, K) or (B,) depending on input
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


def cxcywh_to_x1y1x2y2(boxes):
    # boxes: (...,4) cx,cy,w,h -> x1,y1,x2,y2
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=-1)


def ensure_2d_boxes(boxes):
    """
    Accept boxes as (B,4) or (B,K,4) and convert to (B,K,4).
    """
    if boxes.dim() == 2:
        return boxes.unsqueeze(1)  # (B,1,4)
    return boxes  # (B,K,4)


class IoUHead(nn.Module):
    def __init__(self,
                 feat_dim: int = 512,
                 pooled_size: Tuple[int, int] = (7, 7),
                 hidden_dim: int = 256,
                 mlp_layers: int = 2):
        """
        Args:
            feat_dim: channel dim of input feature maps (after cls head)
            pooled_size: output spatial size from ROIAlign (Hpool, Wpool)
            hidden_dim: hidden dimension for MLP head
            mlp_layers: number of linear layers after global pooling
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.pooled_size = pooled_size

        # small conv to reduce channels / process pooled patch if desired
        self.conv_pool = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # alternative: flatten the pooled patch and use MLP
        mlp = []
        input_dim = hidden_dim
        for i in range(mlp_layers - 1):
            mlp.append(nn.Linear(input_dim, input_dim))
            mlp.append(nn.ReLU(inplace=True))
        # final linear to single scalar
        mlp.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*mlp)

        # initialize last layer bias to small negative value (optional)
        try:
            nn.init.constant_(self.mlp[-1].bias, 0.0)
        except Exception:
            pass

    def forward(self,
                feat: torch.Tensor,
                boxes: torch.Tensor,
                image_size: Tuple[int, int]):
        """
        Args:
            feat: (B, C, Hf, Wf) feature maps for each image in batch (e.g., search patches)
            boxes: (B,4) or (B,K,4) box coordinates in **image pixel** coords of the input images
                   Accepts **center format (cx,cy,w,h)** by default.
            image_size: (H_img, W_img) the spatial size of the image patch that produced feat
                       (i.e., the size of the crop before it was passed through backbone)
        Returns:
            scores: (B, K) IoU scores (float). If K==1 returns shape (B,)
        Notes:
            - The function uses torchvision.ops.roi_align. The `spatial_scale` parameter
              is computed as feat_w / img_w (or feat_h / img_h). We set spatial_scale using
              width ratio to be consistent.
            - boxes are provided in original image pixel coordinates (not normalized).
        """
        # feat dims
        B, C, Hf, Wf = feat.shape
        H_img, W_img = image_size

        # ensure boxes shape is (B, K, 4)
        boxes_in = ensure_2d_boxes(boxes)  # (B, K, 4)
        Bbox, K, _ = boxes_in.shape

        assert Bbox == B, "Boxes batch size must match feature batch size"

        # convert boxes from cxcywh -> x1y1x2y2
        if boxes_in.shape[-1] == 4:
            boxes_xyxy = cxcywh_to_x1y1x2y2(boxes_in)  # (B,K,4)
        else:
            boxes_xyxy = boxes_in

        # Build roi_align input: list of boxes as (batch_idx, x1, y1, x2, y2)
        # torchvision.ops.roi_align expects boxes in (x1,y1,x2,y2) in the coordinate system of the INPUT
        # We will use spatial_scale so we can pass boxes in image pixel coords and set spatial_scale = Wf / W_img (or Hf/H_img)
        spatial_scale_w = float(Wf) / float(W_img)
        spatial_scale_h = float(Hf) / float(H_img)
        # torchvision roi_align accepts a single spatial_scale scalar. We average the two to be robust.
        spatial_scale = (spatial_scale_w + spatial_scale_h) / 2.0

        # Construct boxes list for roi_align: flatten batch and boxes
        rois = []
        for b in range(B):
            for k in range(K):
                x1, y1, x2, y2 = boxes_xyxy[b, k].tolist()
                rois.append([b, x1, y1, x2, y2])
        rois = torch.tensor(rois, device=feat.device, dtype=feat.dtype)

        # apply roi_align: input is feat (B,C,Hf,Wf); rois shape (num_rois,5)
        pooled = roi_align(feat, rois, output_size=self.pooled_size, spatial_scale=spatial_scale, aligned=True)
        # pooled shape: (B*K, C, Hp, Wp)

        # process pooled with conv + global pool -> (B*K, hidden_dim, 1,1)
        x = self.conv_pool(pooled)  # (B*K, hidden_dim, 1, 1)
        x = x.view(x.size(0), -1)   # (B*K, hidden_dim)

        # MLP -> scalar
        out = self.mlp(x).view(B, K)  # (B, K)

        # If K==1, optionally squeeze
        if out.size(1) == 1:
            return out[:, 0]  # (B,)
        return out
