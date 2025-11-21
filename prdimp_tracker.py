# tracking/prdimp_tracker.py
"""
PrDiMP50 Tracker Module
-----------------------

Implements the test-time tracking loop for the PrDiMP model:
 - Initialization: extract template patch, compute w0, optimize to get final filter
 - Tracking: search region cropping, score map prediction, TCR center estimation,
             BBR refinement with IoU head, update bbox

Requires:
  - PrDiMP50 model (models/prdimp/prdimp50.py)
  - GOT10K-style dataset loader's crop_and_resize logic for consistent patch extraction
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from PIL import Image

from prdimp50 import PrDiMP50
from torchvision.transforms.functional import to_tensor, to_pil_image
from src.regression import (
    sample_gaussian_boxes,
    label_pdf_box_gaussian,
    mc_kl_loss
)
from src.iou_head import cxcywh_to_x1y1x2y2


# -------------------------------------------------------------
# Utility: crop-and-resize (same as dataset loader logic)
# -------------------------------------------------------------
def crop_and_resize(img: Image.Image,
                    center: Tuple[float, float],
                    size: Tuple[float, float],
                    out_size: Tuple[int, int]) -> Image.Image:
    cx, cy = center
    w, h = size

    left = cx - w / 2
    top = cy - h / 2
    right = cx + w / 2
    bottom = cy + h / 2

    img_w, img_h = img.size

    crop_left = int(np.floor(left))
    crop_top = int(np.floor(top))
    crop_right = int(np.ceil(right))
    crop_bottom = int(np.ceil(bottom))

    canvas = Image.new("RGB", (int(w), int(h)), (0, 0, 0))

    src_x1 = max(0, crop_left)
    src_y1 = max(0, crop_top)
    src_x2 = min(img_w, crop_right)
    src_y2 = min(img_h, crop_bottom)

    if src_x1 < src_x2 and src_y1 < src_y2:
        patch = img.crop((src_x1, src_y1, src_x2, src_y2))
        dst_x1 = src_x1 - crop_left
        dst_y1 = src_y1 - crop_top
        canvas.paste(patch, (int(dst_x1), int(dst_y1)))

    resized = canvas.resize((out_size[1], out_size[0]), Image.BILINEAR)
    return resized


# -------------------------------------------------------------
# Main Tracker Class
# -------------------------------------------------------------
class PrDiMPTracker:
    def __init__(self,
                 model: PrDiMP50,
                 device="cuda",
                 template_size=(128, 128),
                 search_size=(320, 320),
                 template_context=2.0,
                 search_context=5.0):
        """
        Args:
            model: PrDiMP50 neural model
            template_size: size of template crop
            search_size: size of search crop
            template_context: context factor for template
            search_context: context factor for search window
        """
        self.model = model.to(device)
        self.device = device
        self.template_size = template_size
        self.search_size = search_size
        self.template_context = template_context
        self.search_context = search_context

        self.state = None  # (cx, cy, w, h)
        self.filter_w = None  # final optimized filter

    # ---------------------------------------------------------
    # Initialization on first frame
    # ---------------------------------------------------------
    def initialize(self, image: Image.Image, box_xywh: Tuple[float, float, float, float]):
        """
        Args:
            image: PIL Image (first frame)
            box_xywh: (xmin, ymin, w, h) target box in first frame
        """
        x, y, w, h = box_xywh
        cx = x + w / 2
        cy = y + h / 2
        self.state = torch.tensor([cx, cy, w, h], dtype=torch.float32).to(self.device)

        # Extract template patch
        crop_w = max(w, h) * self.template_context
        crop_h = crop_w
        template_patch = crop_and_resize(image, (cx, cy), (crop_w, crop_h), self.template_size)
        template_tensor = to_tensor(template_patch).unsqueeze(0).to(self.device)

        # Classification features
        tpl_feat = self.model.extract_classification_features(template_tensor)

        # Build Gaussian label map for template
        B, C, Hf, Wf = tpl_feat.shape
        xs = torch.arange(Wf, device=self.device).float()
        ys = torch.arange(Hf, device=self.device).float()
        xs, ys = torch.meshgrid(ys, xs, indexing="ij")

        sigma = max(1.0, 0.25 * min(Hf, Wf))
        label = torch.exp(-((xs - Hf / 2)**2 + (ys - Wf / 2)**2) / (2 * sigma * sigma))
        label = label / label.sum()
        label = label.unsqueeze(0).unsqueeze(0)  # (1,1,Hf,Wf)

        # Compute filter iterates
        filter_iters = self.model.compute_filter_iterates(tpl_feat, label)
        self.filter_w = filter_iters[-1]  # use final filter

    # ---------------------------------------------------------
    # Tracking single new frame
    # ---------------------------------------------------------
    def track(self, image: Image.Image):
        """
        Input:
            image: PIL image of current frame
        Output:
            new box (xmin, ymin, w, h)
        """
        cx, cy, w, h = self.state.tolist()

        # Build search crop
        crop_w = max(w, h) * self.search_context
        crop_h = crop_w
        search_patch = crop_and_resize(image, (cx, cy), (crop_w, crop_h), self.search_size)
        search_tensor = to_tensor(search_patch).unsqueeze(0).to(self.device)

        # Classification features
        srch_feat = self.model.extract_classification_features(search_tensor)

        # Score map
        score_map = self.model.apply_filter(srch_feat, self.filter_w)  # (1,1,Hf,Wf)
        score_np = score_map.squeeze().detach().cpu().numpy()

        # Center prediction by argmax (PrDiMP also supports soft-argmax)
        y_idx, x_idx = np.unravel_index(np.argmax(score_np), score_np.shape)

        # Map back to image coords
        Hf, Wf = score_map.shape[-2:]
        cx_new = x_idx / Wf * crop_w + (cx - crop_w / 2)
        cy_new = y_idx / Hf * crop_h + (cy - crop_h / 2)

        # BBR refinement with IoU Head
        box_center = torch.tensor([[cx_new, cy_new, w, h]], device=self.device)
        samples, log_q = sample_gaussian_boxes(box_center, sigma_factor=0.2, K=64)

        scores = self.model.iou_head(
            srch_feat,                  # (1,C,Hf,Wf)
            samples,                    # (1,K,4)
            image_size=self.search_size # (H_img, W_img)
        ).unsqueeze(0)  # (1,K)

        log_p_label = label_pdf_box_gaussian(box_center, samples)

        # pick best sample (MAP)
        best_idx = torch.argmax(scores[0]).item()
        best_box = samples[0, best_idx].detach()

        cx_pred, cy_pred, w_pred, h_pred = best_box.tolist()

        # update tracker state
        self.state = torch.tensor([cx_pred, cy_pred, w_pred, h_pred], device=self.device)

        # convert to top-left format
        x1 = cx_pred - w_pred / 2
        y1 = cy_pred - h_pred / 2

        return [x1, y1, w_pred, h_pred]
