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
from typing import Optional
import torch.backends.cudnn as cudnn

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
    log_file: Optional[str] = None,
    log_interval: int = 20,
):

    cudnn.benchmark = True
    os.makedirs(save_dir, exist_ok=True)
    log_path = log_file or os.path.join(save_dir, "training.log")
    open(log_path, "w").close()

    def _sanitize_loss(name: str, value: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(value).all():
            msg = f"[WARN] {name} became non-finite (value tensor contains inf/nan). Clamping to 0."
            print(msg)
            with open(log_path, "a") as lf:
                lf.write(msg + "\n")
            return torch.zeros_like(value)
        return value

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
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )
    print(f"[INFO] Loaded {len(dataset)} samples.")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = PrDiMP50().to(device)
    if torch.cuda.device_count() > 1 and device.startswith("cuda"):
        model = nn.DataParallel(model)
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel")
    model_core = model.module if isinstance(model, nn.DataParallel) else model

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda')

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    print(f"\n[INFO] Starting training for {num_epochs} epochs...\n")

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tcr = 0.0
        epoch_bbr = 0.0
        t0 = time.time()

        num_batches = len(loader)

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
                tpl_feat = model_core.extract_classification_features(template)
                _, _, Hf, Wf = tpl_feat.shape

                xs = torch.arange(Wf, device=device).view(1,1,1,Wf).float()
                ys = torch.arange(Hf, device=device).view(1,1,Hf,1).float()

                sigma = 2.0
                tpl_label_maps = torch.exp(-(((xs - (tpl_gt[:,0].view(-1,1,1,1) / template.shape[3] * Wf))**2 +
                                              (ys - (tpl_gt[:,1].view(-1,1,1,1) / template.shape[2] * Hf))**2)
                                             / (2 * sigma * sigma)))
                tpl_label_maps = tpl_label_maps / (tpl_label_maps.view(Ntpl, -1).sum(dim=1, keepdim=True).view(Ntpl,1,1,1) + 1e-12)
            template_label_maps = tpl_label_maps

            # Forward with AMP
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                while template_label_maps.dim() > 4:
                    template_label_maps = template_label_maps.squeeze(2)
                if template_label_maps.dim() == 3:
                    template_label_maps = template_label_maps.unsqueeze(1)

                losses = model(template, search, template_label_maps, srch_gt)

                losses["loss_total"] = _sanitize_loss("loss_total", losses["loss_total"])
                losses["loss_tcr"] = _sanitize_loss("loss_tcr", losses["loss_tcr"])
                losses["loss_bbr"] = _sanitize_loss("loss_bbr", losses["loss_bbr"])
                total_loss = losses["loss_total"]

            # Backprop
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging
            epoch_loss += total_loss.item()
            epoch_tcr += losses["loss_tcr"].item()
            epoch_bbr += losses["loss_bbr"].item()

            if ((step + 1) % log_interval == 0) or (step + 1 == num_batches):
                prog_total = epoch_loss / (step + 1)
                prog_tcr = epoch_tcr / (step + 1)
                prog_bbr = epoch_bbr / (step + 1)
                progress_line = (f"[Epoch {epoch:02d} | Step {step+1:05d}/{num_batches:05d}] "
                                 f"Total={prog_total:.4f}  TCR={prog_tcr:.4f}  BBR={prog_bbr:.4f}")
                print(progress_line)
                with open(log_path, "a") as lf:
                    lf.write(progress_line + "\n")
        avg_loss = epoch_loss / num_batches
        avg_tcr = epoch_tcr / num_batches
        avg_bbr = epoch_bbr / num_batches
        line = (f"[Epoch {epoch:02d}] Total={avg_loss:.4f}  "
                f"TCR={avg_tcr:.4f}  BBR={avg_bbr:.4f}  Time={time.time() - t0:.1f}s")
        print(line)
        with open(log_path, "a") as lf:
            lf.write(line + "\n")
        # End of epoch
        # avg_loss = epoch_loss / len(loader)
        # print(f"\n[Epoch {epoch:02d}] Avg Loss = {avg_loss:.4f} | "
        #       f"Time = {time.time() - t0:.1f}s\n")

        # Save checkpoint
        ckpt_state = model_core.state_dict()
        ckpt_path = os.path.join(save_dir, f"prdimp50_epoch{epoch:03d}.pth")
        save_checkpoint(model_core, optimizer, epoch, ckpt_path)

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
