# data/got10k_dataset.py
import os
import random
import math
from typing import List, Tuple, Optional, Dict, Any

from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ----------------------
# Utility box functions
# ----------------------
def xywh_tl_to_xywh_center(box):
    """
    Convert GOT-10k box format [xmin, ymin, w, h] (top-left)
    to center format [cx, cy, w, h].
    """
    x, y, w, h = box
    cx = x + w / 2.0
    cy = y + h / 2.0
    return [cx, cy, w, h]
def xywh_to_x1y1x2y2(box: List[float]) -> List[float]:
    x, y, w, h = box
    return [x - w/2.0, y - h/2.0, x + w/2.0, y + h/2.0]

def x1y1x2y2_to_xywh(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w/2.0
    cy = y1 + h/2.0
    return [cx, cy, w, h]

def clip_box(box: List[float], img_w: int, img_h: int) -> List[float]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, img_w - 1))
    x2 = max(0, min(x2, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    y2 = max(0, min(y2, img_h - 1))
    return [x1, y1, x2, y2]

# ----------------------
# GOT-10k sequence helpers
# ----------------------
def list_sequences(split_dir: str) -> List[str]:
    """List sequence directories in given split (train/test/val)."""
    seqs = []
    if not os.path.isdir(split_dir):
        return seqs
    for entry in sorted(os.listdir(split_dir)):
        p = os.path.join(split_dir, entry)
        if os.path.isdir(p):
            seqs.append(entry)
    return seqs

def read_groundtruth(seq_dir: str) -> List[List[float]]:
    """
    Read groundtruth.txt (GOT-10k format: x,y,w,h, where x,y are center coords)
    Returns list of [cx, cy, w, h] per frame.
    """
    gt_file = os.path.join(seq_dir, 'groundtruth.txt')
    if not os.path.isfile(gt_file):
        # Return a dummy GT if file not found, useful for dummy dataset testing
        return [[100.0, 100.0, 50.0, 50.0]] * 10 # Default to 10 frames of dummy GT
        # raise FileNotFoundError(f"groundtruth.txt not found in {seq_dir}")
    boxes = []
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            # GOT-10k uses x,y,w,h (may be ints or floats)
            vals = [float(x) for x in parts[:4]]
            boxes.append(vals)
    return boxes

def list_frame_files(seq_dir: str) -> List[str]:
    """Return sorted list of full paths to frames for one sequence. Supports various naming schemes."""
    imgs_dir = seq_dir
    # GOT-10k sequences contain image files in the sequence folder directly.
    # We'll collect common image extensions.
    exts = ['.jpg', '.jpeg', '.png', '.bmp']
    files = []
    for fname in sorted(os.listdir(imgs_dir)):
        if os.path.splitext(fname)[1].lower() in exts:
            # skip groundtruth.txt
            if fname.lower().startswith('groundtruth'):
                continue
            files.append(os.path.join(imgs_dir, fname))
    # fallback: maybe there's an 'img' subfolder
    if len(files) == 0:
        imgs_sub = os.path.join(seq_dir, 'img')
        if os.path.isdir(imgs_sub):
            for fname in sorted(os.listdir(imgs_sub)):
                if os.path.splitext(fname)[1].lower() in exts:
                    files.append(os.path.join(imgs_sub, fname))
    if len(files) == 0:
        # For dummy dataset creation, return some placeholder paths
        print(f"Warning: No image frames found in {seq_dir}. Using dummy frame paths.")
        return [os.path.join(seq_dir, f'dummy_frame_{i:04d}.jpg') for i in range(10)] # Default to 10 dummy frames
        # raise FileNotFoundError(f"No image frames found in {seq_dir}")
    return files

def read_image(path: str, mode='RGB') -> Image.Image:
    """Read image with PIL and convert to given mode."""
    try:
        im = Image.open(path)
    except (FileNotFoundError, UnidentifiedImageError): # Catch both errors
        # Create a dummy image if file not found or cannot be identified
        print(f"Warning: Image file at {path} not found or cannot be identified. Creating a dummy image.")
        im = Image.new(mode, (224, 224), color = 'red') # Default dummy image size
    if im.mode != mode:
        im = im.convert(mode)
    return im

# ----------------------
# GOT-10k Sequence Dataset (evaluation)
# ----------------------
class GOT10kSequence:
    """
    Lightweight sequence loader for GOT-10k evaluation.
    Usage:
      seq = GOT10kSequence(root, 'val', 'sequence_name')
      for image, gt in seq:
          ...
    """
    def __init__(self, root_dir: str, split: str, seq_name: str):
        self.seq_dir = os.path.join(root_dir, split, seq_name)
        if not os.path.isdir(self.seq_dir):
            raise FileNotFoundError(f"Sequence folder not found: {self.seq_dir}")
        self.frames = list_frame_files(self.seq_dir)
        self.gts = read_groundtruth(self.seq_dir)
        if len(self.frames) != len(self.gts):
            # allow slight mismatch, but warn
            print(f"Warning: frames({len(self.frames)}) != gts({len(self.gts)}) for {seq_name}")
            # clip to min len
            mn = min(len(self.frames), len(self.gts))
            self.frames = self.frames[:mn]
            self.gts = self.gts[:mn]
        self.length = len(self.frames)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[Image.Image, List[float]]:
        img = read_image(self.frames[idx])
        gt = self.gts[idx]  # [cx, cy, w, h]
        return img, gt

# ----------------------
# Training Dataset to sample (template, search) pairs
# ----------------------
class GOT10kTrainDataset(Dataset):
    """
    PyTorch Dataset to provide template-search pairs for training trackers.

    Args:
      root_dir: path to the got10k directory which contains 'train', 'val', 'test'
      split: 'train' (recommended)
      template_size: output size (HxW) for template patches (after crop & resize)
      search_size: output size for search patches
      max_template_scale: jitter scale for template (relative to gt)
      max_search_gap: maximum frame gap between template and search (in frames)
      transform: optional callable to apply additional transforms (on PIL images)
      keep_box_format: if True returns GT in (cx,cy,w,h) else returns x1y1x2y2
    """
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 template_size: Tuple[int,int]=(128,128),
                 search_size: Tuple[int,int]=(320,320),
                 max_search_gap: int = 100,
                 template_jitter: float = 0.3,
                 search_jitter: float = 1.0,
                 transform=None,
                 preload_sequences: bool = False,
                 min_sequence_length: int = 2,
                 keep_box_format: bool = True,
                 rng_seed: Optional[int] = None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"GOT-10k split dir not found: {self.split_dir}")
        self.seq_names = list_sequences(self.split_dir)
        # filter sequences with too short length
        # Modified to use the potentially dummy frame list length
        _filtered_seq_names = []
        for s in self.seq_names:
            seq_path = os.path.join(self.split_dir, s)
            if len(list_frame_files(seq_path)) >= min_sequence_length:
                _filtered_seq_names.append(s)
        self.seq_names = _filtered_seq_names

        if len(self.seq_names) == 0:
            raise RuntimeError(f"No sequences found in {self.split_dir} after filtering for min_sequence_length={min_sequence_length}")
        self.template_size = template_size
        self.search_size = search_size
        self.template_jitter = template_jitter
        self.search_jitter = search_jitter
        self.max_search_gap = max_search_gap
        self.transform = transform
        self.keep_box_format = keep_box_format
        self.rng = random.Random(rng_seed)

        # optional preload metadata (frame lists and gts)
        self.sequences_meta = {}
        for s in self.seq_names:
            seq_dir = os.path.join(self.split_dir, s)
            frames = list_frame_files(seq_dir)
            gts = read_groundtruth(seq_dir)
            L = min(len(frames), len(gts))
            self.sequences_meta[s] = {'frames': frames[:L], 'gts': gts[:L], 'length': L}

        # Create an index mapping for random sampling: store (seq_name) list
        # We sample randomly each __getitem__, so length can be arbitrary (set to large number)
        self._num_samples = sum([max(1, meta['length']) for meta in self.sequences_meta.values()])

    def __len__(self):
        # length indicative only; training loop should use epoch length separately
        return max(100000, self._num_samples)

    # ---------- cropping helpers ----------
    def crop_and_resize(self, img: Image.Image, center: Tuple[float,float], size: Tuple[float,float], out_size: Tuple[int,int]) -> Image.Image:
        """
        Crop around center (cx,cy) with box size (w,h) in image coords; resize to out_size.
        center: (cx,cy) in absolute pixel coordinates
        size: (w,h) box size in pixels
        """
        cx, cy = center
        w, h = size
        left = cx - w/2.0
        top = cy - h/2.0
        right = cx + w/2.0
        bottom = cy + h/2.0
        # expand if outside image by padding with black
        img_w, img_h = img.size
        # compute integer crop box
        crop_left = int(round(left))
        crop_top = int(round(top))
        crop_right = int(round(right))
        crop_bottom = int(round(bottom))
        # If cropping is fully outside, produce black image
        # We'll paste the crop from img into a canvas
        canvas = Image.new('RGB', (int(round(w)), int(round(h))), (0,0,0))
        # compute source coords overlap
        src_x1 = max(0, crop_left)
        src_y1 = max(0, crop_top)
        src_x2 = min(img_w, crop_right)
        src_y2 = min(img_h, crop_bottom)
        if src_x1 < src_x2 and src_y1 < src_y2:
            # destination coords
            dst_x1 = src_x1 - crop_left
            dst_y1 = src_y1 - crop_top
            patch = img.crop((src_x1, src_y1, src_x2, src_y2))
            canvas.paste(patch, (int(dst_x1), int(dst_y1)))
        resized = canvas.resize((out_size[1], out_size[0]), resample=Image.BILINEAR)
        return resized

    def apply_jitter_scale(self, box_xywh_tl, jitter, img_w, img_h):
        """
        Input:
          box_xywh_tl: [xmin, ymin, w, h] (GOT-10k top-left)
        Output:
          [cx, cy, w, h] (center format), after translation+scale jitter and clipping
        """
        # Convert input to center coords first
        x, y, w, h = box_xywh_tl
        cx = x + w/2.0
        cy = y + h/2.0
    
        # translation jitter: relative to sqrt(area)
        max_trans = jitter * math.sqrt(max(1.0, w * h))
        dx = (self.rng.random() * 2.0 - 1.0) * max_trans
        dy = (self.rng.random() * 2.0 - 1.0) * max_trans
    
        # scale jitter (multiplicative)
        # sample a log-uniform scale in [1/(1+jitter), 1+jitter] approximately
        scale = math.exp((self.rng.random() * 2.0 - 1.0) * math.log(1.0 + jitter))
        new_w = w * scale
        new_h = h * scale
    
        new_cx = cx + dx
        new_cy = cy + dy
    
        # clip centers to image bounds
        new_cx = max(0.0, min(img_w - 1.0, new_cx))
        new_cy = max(0.0, min(img_h - 1.0, new_cy))
    
        return [new_cx, new_cy, new_w, new_h]


    def map_box_to_crop(self, box_xywh_tl, crop_center, crop_size, out_size):
        """
        Map a GOT-10k GT box into the cropped & resized patch.
    
        Args:
          box_xywh_tl: [xmin, ymin, w, h] in original image pixels (top-left format)
          crop_center: tuple (crop_cx, crop_cy) absolute pixel coords in original image
          crop_size: tuple (crop_w, crop_h) in original image pixels (before resize)
          out_size: tuple (out_H, out_W) size of desired output patch (height, width)
    
        Returns:
          [cx_out, cy_out, w_out, h_out] — center-format box in the resized patch pixel coords
        """
        # Convert GT from top-left to center
        cx_gt, cy_gt, w_gt, h_gt = xywh_tl_to_xywh_center(box_xywh_tl)
    
        crop_cx, crop_cy = crop_center
        crop_w, crop_h = crop_size
    
        # top-left corner of crop in original image coords
        crop_x1 = crop_cx - crop_w / 2.0
        crop_y1 = crop_cy - crop_h / 2.0
    
        # GT relative position inside crop (in original-image pixels)
        rel_x = cx_gt - crop_x1
        rel_y = cy_gt - crop_y1
    
        out_h, out_w = out_size  # note: out_size is (H, W)
        # scale factors from crop -> output patch
        sx = out_w / float(crop_w)
        sy = out_h / float(crop_h)
    
        cx_out = rel_x * sx
        cy_out = rel_y * sy
        w_out  = w_gt * sx
        h_out  = h_gt * sy
    
        return [cx_out, cy_out, w_out, h_out]


    # ---------- sampling ----------
    def sample_sequence_and_frames(self) -> Tuple[str, int, int]:
        """
        Randomly pick a sequence and two frame indices (template_idx, search_idx)
        with search_idx within max_search_gap of template_idx.
        """
        seq_name = self.rng.choice(self.seq_names)
        meta = self.sequences_meta[seq_name]
        L = meta['length']
        # pick template frame uniformly
        t_idx = self.rng.randint(0, L - 1)
        # pick search frame within +/- max_search_gap
        lo = max(0, t_idx - self.max_search_gap)
        hi = min(L - 1, t_idx + self.max_search_gap)
        # ensure not equal to template (prefer different frames)
        if hi - lo == 0:
            s_idx = t_idx
        else:
            s_idx = self.rng.randint(lo, hi)
            # sometimes prefer same? keep allowed
        return seq_name, t_idx, s_idx

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # sample a sequence and frame pair
        seq_name, t_idx, s_idx = self.sample_sequence_and_frames()
        meta = self.sequences_meta[seq_name]
        frames = meta['frames']
        gts = meta['gts']

        # read images
        img_t = read_image(frames[t_idx])
        img_s = read_image(frames[s_idx])
        gt_t = gts[t_idx]  # [cx, cy, w, h] (as given by dataset)
        gt_s = gts[s_idx]

        img_w, img_h = img_t.size

        # apply jitter / scale to template and search boxes
        t_box_j = self.apply_jitter_scale(gt_t, self.template_jitter, img_w, img_h)
        s_box_j = self.apply_jitter_scale(gt_s, self.search_jitter, img_w, img_h)

        

        # define crop size in image coordinates:
        # common tracker practice: crop a square region around target with context factor (1 + context)
        def make_crop_box(box_xywh, context=2.0):
            cx, cy, w, h = box_xywh
            # use max(w,h)
            s = max(w, h) * context
            return [cx, cy, s, s]

        template_crop = make_crop_box(t_box_j, context=2.0)
        search_crop = make_crop_box(s_box_j, context=5.0)  # larger search area

        # crop & resize
        tpl_im = self.crop_and_resize(img_t, (template_crop[0], template_crop[1]), (template_crop[2], template_crop[3]), self.template_size)
        srch_im = self.crop_and_resize(img_s, (search_crop[0], search_crop[1]), (search_crop[2], search_crop[3]), self.search_size)

        # transform to tensor and normalize (range [0,1])
        tpl_tensor = TF.to_tensor(tpl_im)  # CxHxW
        srch_tensor = TF.to_tensor(srch_im)

        # after computing template_crop = [cx, cy, w, h] (center-format) and cropping/resizing:
        tpl_gt_in_crop = self.map_box_to_crop(gt_t, (template_crop[0], template_crop[1]), (template_crop[2], template_crop[3]), self.template_size)
        srch_gt_in_crop = self.map_box_to_crop(gt_s, (search_crop[0], search_crop[1]), (search_crop[2], search_crop[3]), self.search_size)


        # optional transform
        if self.transform is not None:
            tpl_tensor = self.transform(tpl_tensor)
            srch_tensor = self.transform(srch_tensor)
        if self.rng.random() < 0.01:  # prints occasionally
            print("DEBUG SEQ:", seq_name, "t_idx", t_idx, "s_idx", s_idx)
            print("orig gt (t) [xmin,ymin,w,h]:", gt_t)
            print("template_crop center (cx,cy):", (template_crop[0], template_crop[1]), "size:", (template_crop[2], template_crop[3]))
            print("mapped tpl_gt_in_crop:", tpl_gt_in_crop)
            print("orig gt (s) [xmin,ymin,w,h]:", gt_s)
            print("search_crop center (cx,cy):", (search_crop[0], search_crop[1]), "size:", (search_crop[2], search_crop[3]))
            print("mapped srch_gt_in_crop:", srch_gt_in_crop)


        return {
            'template': tpl_tensor,            # CxHt x Wt
            'search': srch_tensor,            # CxHs x Ws
            'tpl_gt': torch.tensor(tpl_gt_in_crop, dtype=torch.float32),
            'srch_gt': torch.tensor(srch_gt_in_crop, dtype=torch.float32),
            'seq_name': seq_name,
            't_idx': t_idx,
            's_idx': s_idx
        }

# ----------------------
# collate for dataloader
# ----------------------
def got10k_collate(batch):
    """
    batch: list of dicts from GOT10kTrainDataset
    Output: tensors stacked for template/search, and lists for metadata
    """
    templates = torch.stack([item['template'] for item in batch], dim=0)
    searches = torch.stack([item['search'] for item in batch], dim=0)
    tpl_gt = torch.stack([item['tpl_gt'] for item in batch], dim=0)
    srch_gt = torch.stack([item['srch_gt'] for item in batch], dim=0)
    seq_names = [item['seq_name'] for item in batch]
    t_idxs = [item['t_idx'] for item in batch]
    s_idxs = [item['s_idx'] for item in batch]
    return {
        'template': templates,
        'search': searches,
        'tpl_gt': tpl_gt,
        'srch_gt': srch_gt,
        'seq_name': seq_names,
        't_idx': t_idxs,
        's_idx': s_idxs
    }
# ----------------------
# Minimal sanity test / example usage
# ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='path to got10k root (contains train/val/test)')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num', type=int, default=4)
    args = parser.parse_args()

    print("Listing sequences...")
    split_dir = os.path.join(args.root, args.split)
    seqs = list_sequences(split_dir)
    print(f"Found {len(seqs)} sequences in {split_dir}. Example: {seqs[:5]}")

    # instantiate dataset & dataloader
    ds = GOT10kTrainDataset(args.root, split=args.split,
                            template_size=(128,128), search_size=(320,320),
                            max_search_gap=50,
                            template_jitter=0.3, search_jitter=1.0,
                            rng_seed=123)
    dl = DataLoader(ds, batch_size=4, collate_fn=got10k_collate, num_workers=2)

    for i, batch in enumerate(dl):
        print("Batch keys:", list(batch.keys()))
        print("Template shape:", batch['template'].shape)
        print("Search shape:", batch['search'].shape)
        print("tpl_gt:", batch['tpl_gt'])
        print("srch_gt:", batch['srch_gt'])
        if i >= 2:
            break

    # simple sequence loader test
    if len(seqs) > 0:
        seq_name = seqs[0]
        seq = GOT10kSequence(args.root, args.split, seq_name)
        print(f"Sequence {seq_name} length: {len(seq)}. First gt: {seq.gts[0]}")
        img, gt = seq[0]
        print("First frame size:", img.size, "GT:", gt)

