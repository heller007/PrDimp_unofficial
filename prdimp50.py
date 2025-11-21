# models/prdimp/prdimp50.py
"""
Full PrDiMP50 Model Assembly
----------------------------

Connects:
  - Backbone (ResNet-50)
  - Classification Feature Head
  - Model Initializer
  - Model Optimizer (Steepest-Descent)
  - Probabilistic Regression (TCR + BBR)
  - IoU Head (Box scoring)

This module builds the full PrDiMP50 architecture as described in:
  * Bhat et al., DiMP ICCV 2019
  * Danelljan et al., PrDiMP CVPR 2020
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

# Components
from src.resnet50 import ResNet50Backbone
from src.cls_feature_head import ClassificationFeatureHead
from src.initializer import ModelInitializer
from src.optimizer import SteepestDescentOptimizer
from src.iou_head import IoUHead
from src.regression import (
    grid_density_from_scores, 
    kl_loss_grid,
    sample_gaussian_boxes,
    mc_kl_loss,
    label_pdf_box_gaussian
)


class PrDiMP50(nn.Module):
    def __init__(self,
                 backbone_out_layer="layer3",
                 cls_feat_dim=512,
                 filter_size=4,
                 n_opt_iter=5,
                 feat_proj_channels=512,
                 iou_hidden_dim=256):

        super().__init__()

        # Backbone (stride-16 default layer3)
        self.backbone = ResNet50Backbone(
            pretrained=True,
            out_layer=backbone_out_layer,
            train_layers=["layer3", "layer4"]
        )

        # Classification Feature Head
        self.cls_head = ClassificationFeatureHead(
            in_channels=1024,      # ResNet50 layer3 output channels
            out_channels=cls_feat_dim,
            use_bn=True
        )

        # Initializer → w0
        self.initializer = ModelInitializer(
            in_channels=cls_feat_dim,
            filter_size=filter_size
        )

        # Optimizer → produces [w0, w1, ..., wT]
        self.optimizer = SteepestDescentOptimizer(
            in_channels=cls_feat_dim,
            filter_size=filter_size,
            n_iter=n_opt_iter,
            reg_lambda=1e-3
        )

        # IoU Head (for BBR)
        self.iou_head = IoUHead(
            feat_dim=cls_feat_dim,
            pooled_size=(7, 7),
            hidden_dim=iou_hidden_dim
        )

        self.filter_size = filter_size
        self.cls_feat_dim = cls_feat_dim
        self.n_opt_iter = n_opt_iter

    # --------------------------------------------------------
    # Feature Extraction
    # --------------------------------------------------------
    def extract_classification_features(self, images):
        """
        images: (B,3,H,W)
        returns cls_features: (B, C, Hf, Wf)
        """
        feats = self.backbone(images)
        feat = self.backbone.get_feature(feats)    # e.g. (B,1024,20,20)
        cls_feat = self.cls_head(feat)             # (B,512,20,20)
        return cls_feat

    # --------------------------------------------------------
    # Filter Initialization + Optimization
    # --------------------------------------------------------
    def compute_filter_iterates(self, template_cls_feats, label_maps):
        """
        template_cls_feats: (N,C,Hf,Wf) template samples
        label_maps: (N,1,Hf,Wf)
        Returns:
            list of w_i filters
        """
        w0 = self.initializer(template_cls_feats)
        w_iters = self.optimizer(w0, template_cls_feats, label_maps)
        return w_iters

    # --------------------------------------------------------
    # Apply filter to features → score map
    # --------------------------------------------------------
    def apply_filter(self, search_cls_feat, w):
        """
        search_cls_feat: (B,C,Hf,Wf)
        w: filter (1,C,k,k)
        Returns:
            score_map: (B,1,Hf,Wf)
        """
        B, C, Hf, Wf = search_cls_feat.shape

        # Convolution per batch element (fast)
        return F.conv2d(
            search_cls_feat, 
            w, 
            padding=self.filter_size // 2
        )   # (B,1,Hf,Wf)

    # --------------------------------------------------------
    # Forward Pass for TRAINING
    # --------------------------------------------------------
    def forward_train(self,
                      template_images,
                      search_images,
                      template_label_maps,
                      search_gt_boxes):
        """
        Full training forward pass:
            - Extract features
            - Compute filter iterates
            - Compute TCR loss
            - Compute BBR loss

        Args:
            template_images: (N,C,H,W) template frames
            search_images:   (B,C,H,W) search frames
            template_label_maps: (N,1,Hf,Wf)
            search_gt_boxes: (B,4) in center format (cx,cy,w,h)
        """
        # FIX: label maps must be 4D (N,1,H,W)
        if template_label_maps.dim() == 5:
            # remove middle singular dims until 4D
            while template_label_maps.dim() > 4:
                template_label_maps = template_label_maps.squeeze(2)

        # 1. Extract classification features
        tpl_feat = self.extract_classification_features(template_images)   # (N, C, Hf, Wf)
        srch_feat = self.extract_classification_features(search_images)    # (B, C, Hf, Wf)

        # 2. Compute filter iterates
        filter_iters = self.compute_filter_iterates(tpl_feat, template_label_maps)

        # 3. TCR Loss (average over iterates)
        tcr_losses = []
        for w in filter_iters:
            score_map = self.apply_filter(srch_feat, w)
            # build label distribution for search images
            B, _, Hf, Wf = score_map.shape

            # create Gaussian label pdf on grid
            xs = torch.arange(Wf, device=score_map.device).float()
            ys = torch.arange(Hf, device=score_map.device).float()
            xs, ys = torch.meshgrid(ys, xs, indexing="ij")

            label_pdf = torch.zeros((B,1,Hf,Wf), device=score_map.device)
            sigma = 1.0  # can tune

            for i in range(B):
                cx, cy, _, _ = search_gt_boxes[i]
                cx = cx / search_images.shape[3] * Wf
                cy = cy / search_images.shape[2] * Hf
                g = torch.exp(-((xs - cy)**2 + (ys - cx)**2)/(2*sigma*sigma))
                g = g / g.sum()
                label_pdf[i,0] = g

            tcr_losses.append(kl_loss_grid(score_map, label_pdf))

        tcr_loss = sum(tcr_losses) / len(tcr_losses)

        # 4. BBR Loss (Monte Carlo)
        # Sample K boxes
        samples, log_q = sample_gaussian_boxes(search_gt_boxes, sigma_factor=0.1, K=32)

        # Evaluate IoU scores via IoU Head
        B = search_images.size(0)
        img_h, img_w = search_images.shape[2], search_images.shape[3]
        scores = self.iou_head(srch_feat, samples, (img_h, img_w))  # (B,K)

        # Compute label log-pdf for samples
        log_p_label = label_pdf_box_gaussian(search_gt_boxes, samples)

        # MC-KL loss
        bbr_loss = mc_kl_loss(scores, log_q, log_p_label)

        total_loss = tcr_loss + bbr_loss

        return {
            "loss_total": total_loss,
            "loss_tcr": tcr_loss,
            "loss_bbr": bbr_loss
        }

    # --------------------------------------------------------
    # Forward Pass for INFERENCE
    # --------------------------------------------------------
    def predict_score_map(self, search_image, filter_w):
        """Used by tracker"""
        feat = self.extract_classification_features(search_image)
        score = self.apply_filter(feat, filter_w)
        return score
