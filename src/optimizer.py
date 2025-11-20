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
            label_maps: (N, 1, H, W) label maps (e.g., gaussian target maps)
            mask: optional (N, 1, H, W) spatial weights (same shape as label_maps); if None, ones used.

        Returns:
            List of filters [w0, w1, ..., wT] each of shape (1, C, k, k)
        """
        device = feat_samples.device
        dtype = feat_samples.dtype

        # basic checks
        assert w_init.ndim == 4, "w_init must be (1, C, k, k)"
        N, C, H, W = feat_samples.shape
        assert C == self.in_channels, f"feat_samples channels {C} != in_channels {self.in_channels}"

        labels = self._ensure_label_shape(label_maps).to(dtype=dtype, device=device)  # (N,1,H,W)
        if mask is None:
            mask = torch.ones_like(labels, device=device, dtype=dtype)
        else:
            mask = mask.to(dtype=dtype, device=device)
        # Ensure shapes
        assert labels.shape == mask.shape, "labels and mask must have same shape (N,1,H,W)"

        # Precompute unfolded feature patches for efficient gradient and Hv computations.
        # F.unfold returns (N, C*k*k, L) where L = H*W (because padding preserves size here)
        pad = self.filter_size // 2
        # Extract local patches for every spatial location
        feat_unf = F.unfold(feat_samples, kernel_size=self.filter_size, padding=pad)  # (N, C*k*k, L)
        # We'll use these repeatedly
        N, CK2, L = feat_unf.shape  # CK2 = C * k * k, L = H*W
        assert CK2 == C * (self.filter_size ** 2)

        # to make matrix multiplies easier, transpose to (N, L, C*k*k)
        feat_unf_t = feat_unf.transpose(1, 2).contiguous()  # (N, L, C*k*k)

        # Precompute mask & labels flattened
        labels_flat = labels.view(N, -1)  # (N, L)
        mask_flat = mask.view(N, -1)      # (N, L)

        # Convert initial filter to vector form
        w = w_init.clone().to(device=device, dtype=dtype)  # (1, C, k, k)
        w_vec = w.view(1, -1)  # (1, C*k*k)

        # Prepare list of iterates
        iterates = [w.clone()]

        # For convenience, precompute preconditioner vector if present
        if self.precond is not None:
            # precond shape (1, C, 1, 1) -> expand to match filter vector shape
            pc = self.precond.view(1, -1).repeat(1, self.filter_size * self.filter_size)  # (1, C*k*k)
        else:
            pc = None

        # main loop: n_iter iterations
        for it in range(self.n_iter):
            # 1) compute score maps s = conv(feat, w)
            # Convolution at every spatial location corresponds to dot(feat_unf_t, w_vec^T)
            # feat_unf_t: (N, L, CK2); w_vec.T: (CK2, 1) -> produces (N, L, 1)
            # compute via batched matmul
            # w_vec currently shape (1, CK2) -> expand to (N, CK2)
            w_vec_n = w_vec.expand(N, -1)  # (N, CK2)
            # compute s_flat (N, L)
            s_flat = torch.bmm(feat_unf_t, w_vec_n.unsqueeze(2)).squeeze(2)  # (N, L)

            # 2) residual r = mask * (s - y)
            resid = mask_flat * (s_flat - labels_flat)  # (N, L)

            # 3) gradient g = sum_i F.unfold(x_i) * resid_i  -> in vector form
            # feat_unf: (N, CK2, L); resid: (N, L)
            # Multiply feat_unf by resid and sum over L
            # Expand resid to shape (N, 1, L) for broadcasting
            resid_exp = resid.unsqueeze(1)  # (N,1,L)
            grad_vec = torch.bmm(feat_unf, resid_exp.transpose(1, 2)).squeeze(2)  # (N, CK2)
            # Sum across samples to get overall gradient (1, CK2)
            grad_vec_sum = grad_vec.sum(dim=0, keepdim=True) / max(1, N)  # (1, CK2)
            # add regularization term: + reg_lambda * w_vec
            grad_vec_sum = grad_vec_sum + self.reg_lambda * w_vec  # (1, CK2)

            # 4) compute numerator = ||grad||^2
            grad_flat = grad_vec_sum.view(-1)
            numer = (grad_flat * grad_flat).sum()

            # 5) compute approximate Hv = J * grad where J approximates Jacobian (Gauss-Newton)
            # We approximate Hv via: for each sample i:
            #   conv_i = F.fold( feat_unf_i^T * (feat_unf_i * grad) )  --> but vectorized:
            # compute per-sample inner products: for sample i, compute v_i = feat_unf_t_i @ grad_vec_sum.T  -> (L,1)
            # then map back to filter-space via feat_unf * v_i summed over L.
            # Implement in vectorized fashion:
            #  - step A: per-sample responses h_i = feat_unf_t_i (N,L,CK2) @ grad_vec_sum.T (CK2,1) -> (N,L)
            #  - step B: multiply feat_unf (N,CK2,L) with h_i (N,1,L) and sum over L -> (N,CK2)
            # Finally average over N and add reg term.

            # step A
            # grad_vec_sum.T is (CK2,1) -> make it (N, CK2, 1) for bmm
            grad_for_bmm = grad_vec_sum.t().unsqueeze(0).expand(N, -1, -1)  # (N, CK2, 1)
            # feat_unf: (N, CK2, L) -> transpose to (N, L, CK2) is feat_unf_t, we already have it
            # so compute h = feat_unf_t @ grad (N, L, CK2) @ (N, CK2,1) -> (N,L,1) -> squeeze
            h = torch.bmm(feat_unf_t, grad_for_bmm).squeeze(2)  # (N, L)

            # step B: compute Hv contribution per sample: feat_unf (N, CK2, L) * h (N,1,L) -> sum over L -> (N,CK2)
            h_exp = h.unsqueeze(1)  # (N,1,L)
            Hv_per_sample = torch.bmm(feat_unf, h_exp.transpose(1, 2)).squeeze(2)  # (N, CK2)
            Hv = Hv_per_sample.sum(dim=0, keepdim=True) / max(1, N)  # (1, CK2)
            # add regularization contribution
            Hv = Hv + self.reg_lambda * w_vec  # (1, CK2)

            # compute denom = grad^T Hv
            denom = (grad_vec_sum.view(-1) * Hv.view(-1)).sum() + self.eps

            # compute alpha = numer / denom
            alpha = numer / denom
            # if alpha is NaN or inf, clamp it
            if not torch.isfinite(alpha):
                alpha = torch.tensor(0.0, device=device, dtype=dtype)

            # Optional preconditioning: divide gradient by preconditioner
            if pc is not None:
                grad_update = grad_vec_sum / pc
            else:
                grad_update = grad_vec_sum

            # Update w_vec
            w_vec = w_vec - alpha.view(1, 1) * grad_update  # (1, CK2)

            # reshape back to (1, C, k, k)
            w = w_vec.view(1, C, self.filter_size, self.filter_size).clone()
            iterates.append(w.clone())

        return iterates
