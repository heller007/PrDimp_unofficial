# train/train_prdimp.py
"""
Training script for PrDiMP50 on GOT-10k

Requires:
    - PrDiMP50 model (models/prdimp/prdimp50.py)
    - GOT10kTrainDataset (data/got10k_dataset.py)
    - Python 3.8+, PyTorch 1.7+ or newer
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from prdimp50 import PrDiMP50
from data.got10k_dataset import GOT10kTrainDataset, got10k_collate


# ------------------------------------------------------------
# Helper: save model checkpoint
# ------------------------------------------------------------
def save_checkpoint(model, optimizer, epoch, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(ckpt, save_path)
    print(f"[INFO] Saved checkpoint → {save_path}")


# ------------------------------------------------------------
# Training function
# ------------------------------------------------------------
def train_prdimp(
    data_root,
    save_dir="./checkpoints",
    batch_size=4,
    num_workers=4,
    lr=1e-4,
    weight_decay=1e-4,
    num_epochs=10,
    device="cuda",
):

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    print("\n[INFO] Loading GOT-10k training dataset...")
    dataset = GOT10kTrainDataset(
        root_dir=data_root,
        split="train",
        template_size=(128,128),
        search_size=(320,320),
        template_jitter=0.3,
        search_jitter=1.0,
        max_search_gap=50,
        rng_seed=123
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=got10k_collate,
        num_workers=num_workers,
        pin_memory=True
    )
    print(f"[INFO] Loaded {len(dataset)} samples.")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = PrDiMP50().to(device)
    print(f"[INFO] Model initialized on {device}")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    print(f"\n[INFO] Starting training for {num_epochs} epochs...\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(loader):

            # Move tensors to device
            template = batch["template"].to(device)
            search   = batch["search"].to(device)
            tpl_gt   = batch["tpl_gt"].to(device)
            srch_gt  = batch["srch_gt"].to(device)

            # Build template label maps (Gaussian)
            # Perfectly fine for training – PrDiMP uses Gaussian heatmaps
            B, C, Ht, Wt = template.shape
            Ntpl = template.shape[0]

            # Convert GT (cx,cy,w,h) → Gaussian label map for template features
            with torch.no_grad():
                tpl_feat = model.extract_classification_features(template)
                _, _, Hf, Wf = tpl_feat.shape

                xs = torch.arange(Wf, device=device).float().view(1,1,1,Wf)
                ys = torch.arange(Hf, device=device).float().view(1,1,Hf,1)

                tpl_label_maps = []
                sigma = 2.0

                for i in range(Ntpl):
                    cx, cy, _, _ = tpl_gt[i]  # in template-patch pixel coords
                    cx = cx / template.shape[3] * Wf
                    cy = cy / template.shape[2] * Hf

                    g = torch.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * sigma**2))
                    g = g / g.sum()
                    tpl_label_maps.append(g)

                tpl_label_maps = torch.stack(tpl_label_maps, dim=0)  # (N,1,Hf,Wf)

            template_label_maps = tpl_label_maps

            # Forward with AMP
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                while template_label_maps.dim() > 4:
                    template_label_maps = template_label_maps.squeeze(2)
                if template_label_maps.dim() == 3:
                    template_label_maps = template_label_maps.unsqueeze(1)

                losses = model.forward_train(
                    template_images=template,
                    search_images=search,
                    template_label_maps=template_label_maps,
                    search_gt_boxes=srch_gt
                )

                total_loss = losses["loss_total"]

            # Backprop
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging
            epoch_loss += total_loss.item()
            if step % 20 == 0:
                print(f"[Epoch {epoch:02d} Step {step:05d}] "
                      f"Total={total_loss.item():.4f}  "
                      f"TCR={losses['loss_tcr'].item():.4f}  "
                      f"BBR={losses['loss_bbr'].item():.4f}")

        # End of epoch
        avg_loss = epoch_loss / len(loader)
        print(f"\n[Epoch {epoch:02d}] Avg Loss = {avg_loss:.4f} | "
              f"Time = {time.time() - t0:.1f}s\n")

        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f"prdimp50_epoch{epoch:03d}.pth")
        save_checkpoint(model, optimizer, epoch, ckpt_path)

    print("\n[INFO] Training finished successfully!\n")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, type=str,
                        help="Path to GOT-10k root directory")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    args = parser.parse_args()

    train_prdimp(
        data_root=args.data_root,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
