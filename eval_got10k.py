# eval/eval_got10k.py
"""
Evaluation script for PrDiMP50 on GOT-10k (val/test)

Uses:
  - GOT10kSequence (from got10k_dataset.py)
  - PrDiMPTracker (tracking/prdimp_tracker.py)
  - PrDiMP50 model (models/prdimp/prdimp50.py)

Metrics:
  - Success (IoU)
  - Precision (center error < 20px)
  - Norm Precision (normalized center error)
"""

import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from models.prdimp.prdimp50 import PrDiMP50
from tracking.prdimp_tracker import PrDiMPTracker
from data.got10k_dataset import GOT10kSequence


# ---------------------------------------------------------
# IoU and Precision Metrics
# ---------------------------------------------------------
def iou(boxA, boxB):  
    # boxA, boxB: [x1,y1,w,h]
    xA1, yA1, wA, hA = boxA
    xA2, yA2 = xA1 + wA, yA1 + hA

    xB1, yB1, wB, hB = boxB
    xB2, yB2 = xB1 + wB, yB1 + hB

    inter_x1 = max(xA1, xB1)
    inter_y1 = max(yA1, yB1)
    inter_x2 = min(xA2, xB2)
    inter_y2 = min(yA2, yB2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = wA * hA
    areaB = wB * hB

    return inter_area / (areaA + areaB - inter_area + 1e-8)


def center_error(b1, b2):
    # center distance between 2 boxes
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    c1 = np.array([x1 + w1/2, y1 + h1/2])
    c2 = np.array([x2 + w2/2, y2 + h2/2])
    return np.linalg.norm(c1 - c2)


# ---------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------
def evaluate_got10k(model_path, data_root, split="val", device="cuda"):
    """
    Args:
        model_path: path to checkpoint (.pth)
        data_root: GOT-10k root (contains train/val/test)
        split: 'val' or 'test'
    """
    print(f"\n[INFO] Loading PrDiMP50 checkpoint: {model_path}")
    model = PrDiMP50().to(device)

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tracker = PrDiMPTracker(model, device=device)

    # List sequences
    split_dir = os.path.join(data_root, split)
    seq_names = sorted(os.listdir(split_dir))
    seq_names = [s for s in seq_names if os.path.isdir(os.path.join(split_dir, s))]

    results_dir = os.path.join("eval_results", split)
    os.makedirs(results_dir, exist_ok=True)

    all_success = []
    all_precision = []
    all_norm_precision = []

    print(f"[INFO] Evaluating {len(seq_names)} sequences from GOT-10k/{split} ...\n")

    for seq in tqdm(seq_names, desc="Evaluating sequences"):
        seq_obj = GOT10kSequence(data_root, split, seq)
        seq_len = len(seq_obj)

        gt = [seq_obj[i][1] for i in range(seq_len)]
        gt_xywh = []

        # Convert GT from (cx,cy,w,h) → (xmin,ymin,w,h)
        for g in gt:
            cx, cy, w, h = g
            x = cx - w/2
            y = cy - h/2
            gt_xywh.append([x, y, w, h])

        # First frame initialization
        first_img, first_gt_center = seq_obj[0]
        x1 = first_gt_center[0] - first_gt_center[2]/2
        y1 = first_gt_center[1] - first_gt_center[3]/2
        w = first_gt_center[2]
        h = first_gt_center[3]
        init_box = [x1, y1, w, h]

        tracker.initialize(first_img, init_box)

        pred_boxes = []
        pred_boxes.append(init_box)

        # Track all frames
        for i in range(1, seq_len):
            img, _ = seq_obj[i]
            pred = tracker.track(img)
            pred_boxes.append(pred)

        # ----------------------------------------------------
        # Save results in GOT-10K format
        # ----------------------------------------------------
        result_path = os.path.join(results_dir, f"{seq}.txt")
        with open(result_path, "w") as f:
            for box in pred_boxes:
                x, y, w, h = box
                f.write(f"{x:.2f},{y:.2f},{w:.2f},{h:.2f}\n")

        # ----------------------------------------------------
        # Compute metrics for sequence
        # ----------------------------------------------------
        succ_list = []
        prec_list = []
        norm_prec_list = []

        for (pb, gb) in zip(pred_boxes, gt_xywh):
            succ_list.append(iou(pb, gb))
            prec_list.append(1 if center_error(pb, gb) <= 20 else 0)

            # normalized precision: center error / sqrt(w*h)
            err = center_error(pb, gb)
            norm_prec_list.append(1 if err <= (0.1 * np.sqrt(gb[2]*gb[3])) else 0)

        all_success.append(np.mean(succ_list))
        all_precision.append(np.mean(prec_list))
        all_norm_precision.append(np.mean(norm_prec_list))

    # --------------------------------------------------------
    # Final aggregated scores
    # --------------------------------------------------------
    print("\n============== GOT-10K EVALUATION RESULTS ==============\n")
    print(f"Success (IoU):        {np.mean(all_success):.4f}")
    print(f"Precision (20px):     {np.mean(all_precision):.4f}")
    print(f"Norm Precision (0.1): {np.mean(all_norm_precision):.4f}")
    print("\n=========================================================\n")

    return {
        "success": np.mean(all_success),
        "precision": np.mean(all_precision),
        "norm_precision": np.mean(all_norm_precision)
    }


# ---------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to GOT-10k root folder")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained .pth checkpoint")

    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"])

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    evaluate_got10k(
        model_path=args.model,
        data_root=args.data_root,
        split=args.split,
        device=args.device
    )
