# models/probabilistic/regression.py
"""
Probabilistic Regression Module for PrDiMP

Implements:
  - Grid softmax density (for Target Center Regression)
  - KL loss on grid
  - Monte Carlo sampling based KL (for bounding box regression)
  - Helper utilities for proposal generation

Reference:
  Danelljan et al., "Probabilistic Regression for Visual Tracking", CVPR 2020
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import math


# -------------------------------------------------------
# 1. GRID SOFTMAX (for Target Center Regression)
# -------------------------------------------------------

def grid_density_from_scores(score_map: torch.Tensor, eps=1e-8):
    """
    Compute softmax density p(y|x) over spatial grid.

    Args:
        score_map: (B, 1, H, W)
    Returns:
        p: normalized density (B, 1, H, W)
    """
    B, _, H, W = score_map.shape
    flat = score_map.view(B, -1)
    m = flat.max(dim=1, keepdim=True)[0]
    exp_flat = torch.exp(flat - m)
    Z = exp_flat.sum(dim=1, keepdim=True) + eps
    p_flat = exp_flat / Z
    return p_flat.view(B, 1, H, W)


def kl_loss_grid(score_map: torch.Tensor, label_pdf_grid: torch.Tensor, eps=1e-8):
    """
    KL divergence for TCR:
       KL(p_label || p_pred) = sum p_label * (log p_label - log p_pred)
    Up to constant, used during training.

    Args:
        score_map: (B,1,H,W) unnormalized logits
        label_pdf_grid: (B,1,H,W) normalized label density
    Returns:
        scalar loss
    """
    B = score_map.size(0)
    flat_scores = score_map.view(B, -1)
    flat_label = label_pdf_grid.view(B, -1)

    # log Z = logsumexp(scores)
    logZ = torch.logsumexp(flat_scores, dim=1)  # (B,)

    # ∑ label * s
    s_term = (flat_label * flat_scores).sum(dim=1)  # (B,)

    loss = logZ - s_term
    return loss.mean()


# -------------------------------------------------------
# 2. MONTE CARLO KL FOR BBR
# -------------------------------------------------------

def mc_kl_loss(samples_scores: torch.Tensor,
               log_q: torch.Tensor,
               log_p_label: torch.Tensor,
               eps=1e-8):
    """
    Monte Carlo estimator of KL divergence for bounding box regression.

    Follows Eq. 10 in PrDiMP paper:
        E_q [ exp(s - log q) ] and weighted term.

    Args:
        samples_scores: (B, K)     s_theta(b_k)
        log_q: (B, K)             log q(b_k | data)
        log_p_label: (B, K)       log p_label(b_k) ground-truth density

    Returns:
        scalar loss
    """
    B, K = samples_scores.shape

    # term1 = log( (1/K) * sum exp(s_k - log q_k) )
    a = samples_scores - log_q
    a_max = a.max(dim=1, keepdim=True)[0]
    log_sum = torch.log(torch.exp(a - a_max).sum(dim=1) + eps) + a_max.squeeze(1)
    term1 = log_sum - math.log(K)

    # term2 = (1/K) * sum s_k * p_label(b_k)/q(b_k)
    w = torch.exp(log_p_label - log_q)
    term2 = (samples_scores * w).sum(dim=1) / K

    return (term1 - term2).mean()


# -------------------------------------------------------
# 3. SAMPLING HELPERS
# -------------------------------------------------------

def sample_gaussian_boxes(gt_boxes, sigma_factor=0.1, K=64):
    """
    Sample K bounding boxes around ground truth (center xywh).

    Args:
        gt_boxes: (B, 4) in (cx,cy,w,h)
    Returns:
        samples: (B, K, 4)
        log_q: (B, K)
    """
    B = gt_boxes.size(0)
    device = gt_boxes.device
    cx, cy, w, h = gt_boxes.unbind(dim=1)

    # sample dx, dy ~ N(0, sigma^2 * (w+h))
    sigma_xy = sigma_factor * (w + h) / 2.0
    dx = torch.randn(B, K, device=device) * sigma_xy.unsqueeze(1)
    dy = torch.randn(B, K, device=device) * sigma_xy.unsqueeze(1)

    # sample dw, dh multiplicative noise
    sigma_scale = sigma_factor
    dw = torch.randn(B, K, device=device) * sigma_scale
    dh = torch.randn(B, K, device=device) * sigma_scale

    # create samples
    samples_cx = cx.unsqueeze(1) + dx
    samples_cy = cy.unsqueeze(1) + dy
    samples_w = w.unsqueeze(1) * torch.exp(dw)
    samples_h = h.unsqueeze(1) * torch.exp(dh)

    samples = torch.stack([samples_cx, samples_cy, samples_w, samples_h], dim=2)

    # compute log_q (proposal density)
    # dx, dy: Gaussian; dw, dh: log-normal -> approx Gaussian in log-space
    var_xy = sigma_xy.unsqueeze(1)**2 + 1e-8
    var_scale = sigma_scale**2 + 1e-8

    log_q = (
        -0.5 * ((dx**2) / var_xy)
        -0.5 * ((dy**2) / var_xy)
        -0.5 * ((dw**2) / var_scale)
        -0.5 * ((dh**2) / var_scale)
    )

    return samples, log_q


# -------------------------------------------------------
# 4. LABEL DENSITY FOR BBR
# -------------------------------------------------------

def label_pdf_box_gaussian(gt_boxes, samples, sigma_factor=0.1):
    """
    Compute log p_label(b) for sampled boxes.

    Gaussian in (cx,cy) and log-scale in w,h.

    Args:
        gt_boxes: (B,4)
        samples: (B,K,4)
    Returns:
        log_p_label: (B,K)
    """
    B, K, _ = samples.shape
    device = samples.device
    cx_gt, cy_gt, w_gt, h_gt = gt_boxes.unbind(dim=1)
    cx_s, cy_s, w_s, h_s = samples.unbind(dim=2)

    sigma_xy = sigma_factor * (w_gt + h_gt) / 2.0
    var_xy = sigma_xy.unsqueeze(1)**2 + 1e-8

    # center distance term
    dxy = (cx_s - cx_gt.unsqueeze(1))**2 + (cy_s - cy_gt.unsqueeze(1))**2
    term_xy = -0.5 * dxy / var_xy

    # scale term: Gaussian in log(w) and log(h)
    lw_gt = torch.log(w_gt + 1e-8).unsqueeze(1)
    lh_gt = torch.log(h_gt + 1e-8).unsqueeze(1)
    lw_s = torch.log(w_s + 1e-8)
    lh_s = torch.log(h_s + 1e-8)

    var_scale = (sigma_factor**2) + 1e-8
    term_scale = -0.5 * ((lw_s - lw_gt)**2 + (lh_s - lh_gt)**2) / var_scale

    return term_xy + term_scale
