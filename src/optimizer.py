# models/predictor/optimizer.py
"""
Optimizer module for DiMP / PrDiMP: Steepest-descent with Gauss-Newton step-size.

Implements a practical, vectorized optimizer that:
 - accepts an initial filter w0 (1, C, k, k)
 - training feature samples X (N, C, H, W)
 - corresponding label maps Y (N, 1, H, W) or (N, H, W)
 - optional spatial mask M (N, 1, H, W) as sample weights
 - performs n_iter updates: w <- w - alpha * grad
 - computes alpha via approximate Gauss-Newton denominator using Hv product

Notes & design choices:
 - Uses F.unfold to avoid Python loops for gradient computation
 - Works with arbitrary filter_size (k)
 - Returns list of iterates as tensors on same device as input
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SteepestDescentOptimizer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 filter_size: int = 4,
                 n_iter: int = 5,
                 reg_lambda: float = 1e-3,
                 eps: float = 1e-8,
                 precond: Optional[torch.Tensor] = None,
                 use_sample_weights: bool = False):
        """
        Args:
            in_channels: C (channels of feature map and filter)
            filter_size: k (filter spatial size)
            n_iter: number of steepest-descent iterations
            reg_lambda: diagonal Tikhonov regularizer on filter parameters
            eps: numerical stability constant
            precond: optional 1D tensor of shape (C,) for diagonal preconditioner
            use_sample_weights: whether `mask` argument contains sample weights
        """
        super().__init__()
        self.in_channels = in_channels
        self.filter_size = filter_size
        self.n_iter = n_iter
        self.reg_lambda = reg_lambda
        self.eps = eps
        self.use_sample_weights = use_sample_weights

        if precond is not None:
            assert precond.ndim == 1 and precond.shape[0] == in_channels
            # store as buffer
            self.register_buffer('precond', precond.view(1, in_channels, 1, 1))
        else:
            self.precond = None

    @staticmethod
    def _ensure_label_shape(labels: torch.Tensor):
        # Accept labels as (N, H, W) or (N,1,H,W) -> unify to (N,1,H,W)
        if labels.ndim == 3:
            return labels.unsqueeze(1)
        return labels

    def forward(self,
            w_init: torch.Tensor,
            feat_samples: torch.Tensor,
            label_maps: torch.Tensor,
            mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Run optimizer.

        Args:
            w_init: (1, C, k, k) initial filter
            feat_samples: (N, C, H, W) classification features from templates
            label_maps: (N, 1, H_label, W_label) label maps (e.g., gaussian target maps)
            mask: optional (N, 1, H_label, W_label) spatial weights (same shape as label_maps); if None, ones used.

        Returns:
            List of filters [w0, w1, ..., wT] each of shape (1, C, k, k)
        """
        device = feat_samples.device
        dtype = feat_samples.dtype

        # Basic checks
        assert w_init.ndim == 4, "w_init must be (1, C, k, k)"
        N, C, H_feat, W_feat = feat_samples.shape
        assert C == self.in_channels, f"feat_samples channels {C} != in_channels {self.in_channels}"

        # Ensure labels shaped (N,1,H_label,W_label) -> convert to device/dtype
        labels = self._ensure_label_shape(label_maps).to(dtype=dtype, device=device)  # (N,1,H_label,W_label)

        # If mask not provided, use ones (match labels shape initially)
        if mask is None:
            mask = torch.ones_like(labels, device=device, dtype=dtype)
        else:
            mask = mask.to(dtype=dtype, device=device)

            # If label spatial size does not match the expected unfolded grid, resize labels & mask
            H_label, W_label = labels.shape[-2], labels.shape[-1]

            # compute unfold output spatial size (number of locations L) for the given filter_size & padding used below
            pad = self.filter_size // 2
            # output spatial dims produced by F.unfold with stride=1
            out_h = H_feat + 2 * pad - self.filter_size + 1
            out_w = W_feat + 2 * pad - self.filter_size + 1

            # If the provided label maps are not already the unfolded spatial size, resize them to (out_h, out_w)
            if (H_label != out_h) or (W_label != out_w):
                # Resize by bilinear (labels are soft heatmaps); align_corners=False for stability
                labels = F.interpolate(labels, size=(out_h, out_w), mode='bilinear', align_corners=False)
                mask = F.interpolate(mask, size=(out_h, out_w), mode='bilinear', align_corners=False)

                # If labels represent a probability map, renormalize per sample so sums to 1
                lab_flat = labels.view(N, -1)
                lab_sum = lab_flat.sum(dim=1, keepdim=True)
                # avoid divide-by-zero
                lab_sum = lab_sum + (lab_sum == 0.).float()
                labels = (lab_flat / lab_sum).view(N, 1, out_h, out_w)

            # Now flatten labels/mask (their flattened length L matches unfold L)
            labels_flat = labels.view(N, -1)  # (N, L)
            mask_flat = mask.view(N, -1)      # (N, L)

        # Precompute unfolded feature patches for efficient gradient and Hv computations.
        pad = self.filter_size // 2
        feat_unf = F.unfold(feat_samples, kernel_size=self.filter_size, padding=pad)  # (N, C*k*k, L)
        N2, CK2, L = feat_unf.shape
        assert N2 == N, "Unfolded batch size mismatch"

        # Transpose to (N, L, C*k*k)
        feat_unf_t = feat_unf.transpose(1, 2).contiguous()  # (N, L, CK2)

        # Convert initial filter to vector form
        w = w_init.clone().to(device=device, dtype=dtype)  # (1, C, k, k)
        w_vec = w.view(1, -1)  # (1, CK2)

        iterates = [w.clone()]

        # Preconditioner expansion
        if self.precond is not None:
            pc = self.precond.view(1, -1).repeat(1, self.filter_size * self.filter_size)  # (1, CK2)
        else:
            pc = None

        # Main loop
        for it in range(self.n_iter):
            # compute score maps s_flat (N, L)
            w_vec_n = w_vec.expand(N, -1)  # (N, CK2)
            s_flat = torch.bmm(feat_unf_t, w_vec_n.unsqueeze(2)).squeeze(2)  # (N, L)

            # residuals
            resid = mask_flat * (s_flat - labels_flat)  # (N, L)

            # gradient: feat_unf (N, CK2, L) * resid (N, L) -> (N, CK2), sum over L
            resid_exp = resid.unsqueeze(1)  # (N,1,L)
            grad_vec = torch.bmm(feat_unf, resid_exp.transpose(1, 2)).squeeze(2)  # (N, CK2)
            grad_vec_sum = grad_vec.sum(dim=0, keepdim=True) / max(1, N)  # (1, CK2)
            grad_vec_sum = grad_vec_sum + self.reg_lambda * w_vec  # regularization

            # numerator
            grad_flat = grad_vec_sum.view(-1)
            numer = (grad_flat * grad_flat).sum()

            # approximate Hv
            grad_for_bmm = grad_vec_sum.t().unsqueeze(0).expand(N, -1, -1)  # (N, CK2, 1)
            h = torch.bmm(feat_unf_t, grad_for_bmm).squeeze(2)  # (N, L)

            h_exp = h.unsqueeze(1)  # (N,1,L)
            Hv_per_sample = torch.bmm(feat_unf, h_exp.transpose(1, 2)).squeeze(2)  # (N, CK2)
            Hv = Hv_per_sample.sum(dim=0, keepdim=True) / max(1, N)  # (1, CK2)
            Hv = Hv + self.reg_lambda * w_vec

            denom = (grad_vec_sum.view(-1) * Hv.view(-1)).sum() + self.eps
            alpha = numer / denom
            if not torch.isfinite(alpha):
                alpha = torch.tensor(0.0, device=device, dtype=dtype)

            # preconditioning
            if pc is not None:
                grad_update = grad_vec_sum / pc
            else:
                grad_update = grad_vec_sum

            # update
            w_vec = w_vec - alpha.view(1, 1) * grad_update
            w = w_vec.view(1, C, self.filter_size, self.filter_size).clone()
            iterates.append(w.clone())

        return iterates


