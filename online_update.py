# tracking/online_update.py
"""
OnlineUpdater for PrDiMP / DiMP
--------------------------------
Handles the online update loop described in DiMP (and used in PrDiMP):
 - Maintain S_train: a buffer of training samples (classification features + label maps)
 - Add safe samples when tracker confidence is high
 - Periodically run the optimizer on S_train using current w as initialization
 - Accept updated model subject to simple sanity checks (peak improvement or loss decrease)

References (implementation motivations):
 - DiMP (ICCV 2019) – online update strategy (Strain size, periodic update every 20 frames, recursions).
   See section 3.7 and Algorithm 1. :contentReference[oaicite:8]{index=8}
 - PrDiMP (CVPR 2020) – probabilistic reasoning for deciding to update and to model uncertainty. :contentReference[oaicite:9]{index=9}
"""

from typing import List, Tuple, Optional
import collections
import torch
import torch.nn.functional as F
import numpy as np

# We'll assume these are importable from your codebase
# - ModelInitializer is used only offline; for online we use the optimizer warm-start
# - SteepestDescentOptimizer exists at models.predictor.optimizer
# - helper crop_and_resize is available (copied from your tracker)
from src.optimizer import SteepestDescentOptimizer
from src.regression import sample_gaussian_boxes, label_pdf_box_gaussian
from torchvision.transforms.functional import to_tensor
from PIL import Image

# Reuse crop_and_resize logic (should match dataset loader / tracker)
def crop_and_resize(img: Image.Image, center: Tuple[float, float], size: Tuple[float, float],
                    out_size: Tuple[int, int]) -> Image.Image:
    cx, cy = center
    w, h = size
    left = cx - w / 2.0
    top  = cy - h / 2.0
    right = cx + w / 2.0
    bottom= cy + h / 2.0

    img_w, img_h = img.size
    crop_left = int(np.floor(left))
    crop_top = int(np.floor(top))
    crop_right = int(np.ceil(right))
    crop_bottom = int(np.ceil(bottom))

    canvas = Image.new("RGB", (int(round(w)), int(round(h))), (0,0,0))
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


class OnlineUpdater:
    def __init__(self,
                 model,                     # instance of PrDiMP50
                 device: str = "cuda",
                 template_size=(128,128),
                 search_size=(320,320),
                 template_context: float = 2.0,
                 search_context: float = 5.0,
                 init_samples: int = 15,    # DiMP uses ~15 initial samples
                 max_memory: int = 50,      # DiMP keeps max 50 samples
                 add_sample_thresh: float = 0.65,  # min peak score to accept a new sample
                 update_interval: int = 20, # run periodic updates every 20 frames
                 periodic_recursions: int = 2,
                 distractor_recursions: int = 1,
                 distractor_peak_ratio: float = 0.6,
                 reg_lambda: float = 1e-3,
                 n_opt_iter: int = 2):
        """
        Args:
            model: PrDiMP50 model instance (used for feature extraction & IoU head)
            device: "cuda" or "cpu"
            template_size/search_size, template_context/search_context: same as tracker
            init_samples: number of augmented template samples to build initial S_train
            max_memory: maximum number of stored samples
            add_sample_thresh: minimum normalized peak score to accept a sample into memory
            update_interval: periodic update frequency (frames)
            periodic_recursions: # of optimizer iterations for periodic update (small)
            distractor_recursions: # of iterations when distractor detected
            distractor_peak_ratio: if second peak >= ratio * main_peak → distractor
            reg_lambda: regularizer used when running optimizer during online updates
            n_opt_iter: default number of iterations for optimizer (per call)
        """
        self.model = model
        self.device = device
        self.template_size = template_size
        self.search_size = search_size
        self.template_context = template_context
        self.search_context = search_context

        self.init_samples = init_samples
        self.max_memory = max_memory
        self.add_sample_thresh = add_sample_thresh
        self.update_interval = update_interval
        self.periodic_recursions = periodic_recursions
        self.distractor_recursions = distractor_recursions
        self.distractor_peak_ratio = distractor_peak_ratio

        # memory buffer stores tuples: (cls_feat_tensor, label_map_tensor)
        self.S_train = collections.deque(maxlen=max_memory)

        self.frame_counter = 0

        # create a lightweight optimizer instance for online updates (smaller iterations)
        self.online_optimizer = SteepestDescentOptimizer(
            in_channels=self.model.cls_feat_dim,
            filter_size=self.model.filter_size,
            n_iter=n_opt_iter,
            reg_lambda=reg_lambda
        ).to(device)

    # ---------------------------
    # Utilities to create label map from box (center coords on patch)
    # ---------------------------
    @staticmethod
    def make_gaussian_label(center_xy, Hf, Wf, sigma=2.0, device="cpu"):
        """
        center_xy: (cx, cy) in patch pixel coords (float)
        returns (1,1,Hf,Wf) normalized gaussian
        """
        xs = torch.arange(Wf, device=device).float().view(1,1,1,Wf)
        ys = torch.arange(Hf, device=device).float().view(1,1,Hf,1)
        cx, cy = center_xy
        g = torch.exp(-((xs - cx)**2 + (ys - cy)**2) / (2.0 * sigma * sigma))
        g = g / (g.sum() + 1e-12)
        return g.unsqueeze(0)  # (1,1,Hf,Wf)

    # ---------------------------
    # Build initial memory (call during tracker.initialize)
    # ---------------------------
    def build_initial_memory(self, first_frame: Image.Image, init_box_tl: Tuple[float,float,float,float]):
        """
        Create initial S_train by augmenting the first template (random jittering)
        - first_frame: PIL Image (first frame)
        - init_box_tl: (xmin, ymin, w, h)
        """
        x, y, w, h = init_box_tl
        cx = x + w/2.0
        cy = y + h/2.0

        crop_w = max(w, h) * self.template_context
        crop_h = crop_w

        for i in range(self.init_samples):
            # jitter sampling: small random translation & scale
            if i == 0:
                # one unmodified sample (the canonical template)
                center = (cx, cy)
                crop_size = (crop_w, crop_h)
            else:
                # small jitter: translation up to 0.15*sqrt(area), scale factor log-normal
                area = max(1.0, w*h)
                max_trans = 0.15 * np.sqrt(area)
                dx = (np.random.rand()*2 - 1) * max_trans
                dy = (np.random.rand()*2 - 1) * max_trans
                scale = float(np.exp((np.random.rand()*2 - 1) * np.log(1.15)))
                center = (cx + dx, cy + dy)
                crop_size = (crop_w * scale, crop_h * scale)

            tpl_patch = crop_and_resize(first_frame, center, crop_size, self.template_size)
            tpl_tensor = to_tensor(tpl_patch).unsqueeze(0).to(self.device)

            with torch.no_grad():
                cls_feat = self.model.extract_classification_features(tpl_tensor)  # (1,C,Hf,Wf)
                _, C, Hf, Wf = cls_feat.shape
                # center in *template patch* coords for label construction:
                # since template is centered at `center`, its center pixel is (Wf/2, Hf/2)
                center_xy = (Wf / 2.0, Hf / 2.0)
                label = self.make_gaussian_label(center_xy, Hf, Wf, sigma=2.0, device=self.device)
                # store (cls_feat, label) as training sample
                self.S_train.append((cls_feat.squeeze(0).detach().cpu(), label.squeeze(0).detach().cpu()))

    # ---------------------------
    # Function to consider adding a new sample (called in tracking loop)
    # ---------------------------
    def consider_adding_sample(self, frame_img: Image.Image, box_tl: Tuple[float,float,float,float], peak_score: float):
        """
        Add sample to S_train if confident.
        - frame_img: PIL image
        - box_tl: (xmin,ymin,w,h) in frame coords (top-left)
        - peak_score: normalized peak score returned by classifier (0..1)
        """
        if peak_score < self.add_sample_thresh:
            return False

        x, y, w, h = box_tl
        cx = x + w/2.0
        cy = y + h/2.0
        crop_w = max(w, h) * self.template_context
        crop_h = crop_w

        tpl_patch = crop_and_resize(frame_img, (cx, cy), (crop_w, crop_h), self.template_size)
        tpl_tensor = to_tensor(tpl_patch).unsqueeze(0).to(self.device)

        with torch.no_grad():
            cls_feat = self.model.extract_classification_features(tpl_tensor)
            _, C, Hf, Wf = cls_feat.shape
            center_xy = (Wf/2.0, Hf/2.0)
            label = self.make_gaussian_label(center_xy, Hf, Wf, sigma=2.0, device=self.device)
            # append to S_train (as CPU tensors to keep GPU memory low)
            self.S_train.append((cls_feat.squeeze(0).detach().cpu(), label.squeeze(0).detach().cpu()))
        return True

    # ---------------------------
    # Detect distractor: check second peak ratio on score map (passed precomputed)
    # ---------------------------
    @staticmethod
    def has_distractor(score_map: torch.Tensor, ratio_thresh: float = 0.6):
        """
        score_map: (1,1,Hf,Wf) tensor
        returns True if second highest peak >= ratio_thresh * highest peak
        """
        s = score_map.detach().cpu().squeeze().numpy().ravel()
        if s.size < 2:
            return False
        idxs = np.argpartition(s, -2)[-2:]
        top = s[idxs].max()
        # get global max
        maxi = s.max()
        # avoid divide by zero
        if maxi <= 1e-8:
            return False
        return (top / maxi) >= ratio_thresh

    # ---------------------------
    # Run online update (called when periodic or distractor-triggered)
    # ---------------------------
        # -----------------------------------------------------------
    # KL-based acceptance (PrDiMP-style)
    # -----------------------------------------------------------
    def update_model(self, current_filter: torch.Tensor,
                     current_search_feat: torch.Tensor,
                     current_label_maps: Optional[torch.Tensor] = None,
                     recursions: Optional[int] = None):
        """
        Update filter using online optimizer, and accept update if
        KL_after < KL_before (probabilistic improvement).
        """

        if len(self.S_train) == 0:
            return current_filter, False

        recursions = recursions if recursions is not None else self.periodic_recursions

        # Prepare batch of (features, labels) from memory
        feats = torch.stack([s[0] for s in self.S_train], dim=0).to(self.device)
        labels = torch.stack([s[1] for s in self.S_train], dim=0).to(self.device)

        # TEMP optimizer (online)
        tmp_opt = SteepestDescentOptimizer(
            in_channels=self.model.cls_feat_dim,
            filter_size=self.model.filter_size,
            n_iter=recursions,
            reg_lambda=self.online_optimizer.reg_lambda
        ).to(self.device)

        # ----------------------------------------------------
        # Compute KL BEFORE UPDATE
        # ----------------------------------------------------
        with torch.no_grad():
            score_pre = self.model.apply_filter(current_search_feat, current_filter)  # (1,1,Hf,Wf)

            # Build label pdf for this search patch:
            _, _, Hf, Wf = score_pre.shape
            xs = torch.arange(Wf, device=self.device).float()
            ys = torch.arange(Hf, device=self.device).float()
            ys, xs = torch.meshgrid(ys, xs, indexing="ij")

            # GT center is patch center (since patch is centered on predicted target)
            cx = Wf / 2.0
            cy = Hf / 2.0
            sigma = 2.0

            g = torch.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * sigma * sigma))
            g = g / (g.sum() + 1e-8)
            label_pdf = g.unsqueeze(0).unsqueeze(0)

            # KL BEFORE
            kl_before = (
                torch.logsumexp(score_pre.view(1, -1), dim=1)
                - (label_pdf.view(1, -1) * score_pre.view(1, -1)).sum(1)
            )[0].item()

        # ----------------------------------------------------
        # Run optimizer (warm-start with current filter)
        # ----------------------------------------------------
        iterates = tmp_opt(current_filter.to(self.device), feats, labels)
        new_filter = iterates[-1]

        # ----------------------------------------------------
        # Compute KL AFTER UPDATE
        # ----------------------------------------------------
        with torch.no_grad():
            score_after = self.model.apply_filter(current_search_feat, new_filter)

            # KL AFTER
            kl_after = (
                torch.logsumexp(score_after.view(1, -1), dim=1)
                - (label_pdf.view(1, -1) * score_after.view(1, -1)).sum(1)
            )[0].item()

        # ----------------------------------------------------
        # Acceptance Criterion (PrDiMP-style)
        # ----------------------------------------------------
        # Accept if KL decreased (better probabilistic fit)
        # Else fall back to peak-based acceptance
        improved = kl_after < kl_before

        # fallback: peak degradation tolerance
        peak_before = score_pre.max().item()
        peak_after = score_after.max().item()
        peak_ok = peak_after >= 0.90 * peak_before

        accept = improved or peak_ok

        if accept:
            return new_filter.detach(), True
        else:
            return current_filter, False


    # ---------------------------
    # step called every tracked frame to run possible updates
    # ---------------------------
    def step(self, frame_img: Image.Image, frame_box_tl: Tuple[float,float,float,float],
             score_map: torch.Tensor, current_filter: torch.Tensor, current_search_feat: torch.Tensor):
        """
        Called from tracking loop each frame.

        - frame_img: PIL image of current frame
        - frame_box_tl: predicted box (xmin,ymin,w,h) in frame coords
        - score_map: (1,1,Hf,Wf) tensor output by classifier on search patch
        - current_filter: (1,C,k,k) current model filter
        - current_search_feat: (1,C,Hf,Wf) current search features (to evaluate before/after)

        Returns:
            updated_filter, accepted_flag (bool)
        """
        self.frame_counter += 1

        # compute normalized peak score (simple min-max across the map)
        s = score_map.detach().cpu().squeeze()
        peak = float(s.max())
        # To normalize, compute a rough normalization by subtracting min, dividing by (max-min)
        smin = float(s.min())
        denom = (peak - smin) if (peak - smin) > 1e-6 else 1.0
        normalized_peak = (peak - smin) / denom

        # Consider adding new training sample
        added = self.consider_adding_sample(frame_img, frame_box_tl, normalized_peak)

        # Periodic update
        accepted = False
        new_filter = current_filter
        if (self.frame_counter % self.update_interval) == 0:
            new_filter, accepted = self.update_model(current_filter, current_search_feat, recursions=self.periodic_recursions)

        # Distractor-triggered update
        elif self.has_distractor(score_map, ratio_thresh=self.distractor_peak_ratio):
            new_filter, accepted = self.update_model(current_filter, current_search_feat, recursions=self.distractor_recursions)

        return new_filter, accepted
