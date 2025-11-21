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
        Steepest descent optimizer used in DiMP/PrDiMP.
        This version includes correct label/mask resizing to match unfold grid.
        """

        device = feat_samples.device
        dtype  = feat_samples.dtype

        # Basic shapes
        N, C, H_feat, W_feat = feat_samples.shape
        k = self.filter_size
        pad = k // 2

        # compute unfolded output spatial size
        out_h = H_feat + 2*pad - k + 1
        out_w = W_feat + 2*pad - k + 1
        L = out_h * out_w

        def _reshape_to_n1hw(tensor: torch.Tensor, name: str) -> torch.Tensor:
            tensor = tensor.to(device=device, dtype=dtype)

            # Iteratively squeeze singleton dims (except batch) until ≤4 dims
            while tensor.dim() > 4:
                squeezed = False
                for dim in range(1, tensor.dim()):
                    if tensor.size(dim) == 1:
                        tensor = tensor.squeeze(dim)
                        squeezed = True
                        break
                if not squeezed:
                    raise ValueError(f"{name} shape {tuple(tensor.shape)} cannot be reduced to 4D (N,1,H,W)")

            if tensor.dim() == 2:            # (H,W)
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.dim() == 3:          # (N,H,W)
                tensor = tensor.unsqueeze(1)
            elif tensor.dim() != 4:
                raise ValueError(f"{name} must become 4D after squeezing; got {tuple(tensor.shape)}")

            if tensor.size(1) != 1:
                tensor = tensor[:, :1]

            batch = tensor.size(0)
            if batch != N:
                if batch == 1:
                    tensor = tensor.expand(N, -1, -1, -1)
                else:
                    raise ValueError(f"{name} batch {batch} != feature batch {N}")

            return tensor.contiguous()
        
        labels = _reshape_to_n1hw(label_maps, "labels")

        if mask is None:
            mask = torch.ones_like(labels, device=device, dtype=dtype)
        else:
            mask = _reshape_to_n1hw(mask, "mask")

        # ------ Resize labels and mask to (out_h, out_w) ------
        H_label, W_label = labels.shape[-2], labels.shape[-1]
        if (H_label != out_h) or (W_label != out_w):
            labels = F.interpolate(labels, size=(out_h, out_w), mode="bilinear", align_corners=False)
            mask   = F.interpolate(mask,   size=(out_h, out_w), mode="bilinear", align_corners=False)

            # Renormalize labels if they are PDFs
            flat = labels.view(N, -1)
            s = flat.sum(dim=1, keepdim=True)
            s = s + (s == 0).float()  # avoid /0
            labels = (flat / s).view(N, 1, out_h, out_w)

        # Now create the flattened versions
        labels_flat = labels.view(N, L)  # (N, L)
        mask_flat   = mask.view(N, L)    # (N, L)

        # Unfold features
        feat_unf = F.unfold(feat_samples, kernel_size=k, padding=pad)  # (N, CK2, L)
        _, CK2, L_unf = feat_unf.shape
        assert L_unf == L, f"L mismatch: unfold produced {L_unf}, expected {L}"

        feat_unf_t = feat_unf.transpose(1, 2).contiguous()  # (N, L, CK2)

        # Initialize w
        w = w_init.to(device=device, dtype=dtype)
        w_vec = w.view(1, -1)  # (1, CK2)

        iterates = [w.clone()]

        for it in range(self.n_iter):

            # score = X @ w
            w_vec_n = w_vec.expand(N, -1)              # (N, CK2)
            s_flat  = torch.bmm(feat_unf_t, w_vec_n.unsqueeze(2)).squeeze(2)  # (N, L)

            # residual
            resid = mask_flat * (s_flat - labels_flat)    # (N, L)

            # gradient = X^T (resid)
            resid_exp = resid.unsqueeze(1)      # (N,1,L)
            grad_vec = torch.bmm(feat_unf, resid_exp.transpose(1,2)).squeeze(2)  # (N, CK2)
            grad_vec = grad_vec.mean(dim=0, keepdim=True)                         # (1, CK2)
            grad_vec = grad_vec + self.reg_lambda * w_vec

            # numerator
            numer = (grad_vec.view(-1)**2).sum()

            # approximate Hessian-vector product
            gv = grad_vec.t().unsqueeze(0).expand(N, -1, -1)   # (N,CK2,1)
            h = torch.bmm(feat_unf_t, gv).squeeze(2)           # (N,L)
            h_exp = h.unsqueeze(1)                             # (N,1,L)
            Hv = torch.bmm(feat_unf, h_exp.transpose(1,2)).squeeze(2)  # (N,CK2)
            Hv = Hv.mean(dim=0, keepdim=True)
            Hv = Hv + self.reg_lambda * w_vec

            denom = (grad_vec.view(-1) * Hv.view(-1)).sum() + self.eps
            alpha = numer / denom
            if not torch.isfinite(alpha):
                alpha = torch.tensor(0.0, device=device, dtype=dtype)

            # update
            w_vec = w_vec - alpha * grad_vec
            w = w_vec.view(1, C, k, k).clone()
            iterates.append(w.clone())

        return iterates


