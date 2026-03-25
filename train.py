import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.height_net import Sentinel2ResUNet
from datasets.raster_datasets import S2S1GEDITileFolderDataset
import config


def masked_sse_mae_count(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """
    pred/target/mask: [B,1,H,W]
    mask > 0 means valid GEDI supervision
    """
    valid = mask > 0.5
    n = valid.sum().item()
    if n == 0:
        return 0.0, 0.0, 0

    diff = pred[valid] - target[valid]
    sse = torch.sum(diff * diff).item()
    sae = torch.sum(torch.abs(diff)).item()
    return sse, sae, n


# -------------------------
# Logging setup
# -------------------------
log_path = getattr(config, "LOG_PATH", "logs/train_gedi.log")
Path(log_path).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)
logging.info("Starting GEDI sparse-supervision training run")


# -------------------------
# Device + threads
# -------------------------
torch.set_num_threads(getattr(config, "NUM_THREADS", 30))
device = torch.device(getattr(config, "DEVICE", "cuda"))
logging.info(f"Using device: {device}")


# -------------------------
# Datasets
# -------------------------
train_ds = S2S1GEDITileFolderDataset(
    config.TRAIN_ROOT,
    s2_subdir=getattr(config, "S2_SUBDIR", "S2"),
    s1_subdir=getattr(config, "S1_SUBDIR", "S1"),
    label_subdir=getattr(config, "GEDI_LABEL_SUBDIR", "GEDI_LABEL"),
    mask_subdir=getattr(config, "GEDI_MASK_SUBDIR", "GEDI_MASK"),
    expected_s2_channels=getattr(config, "NUM_BANDS", None),
)

val_ds = S2S1GEDITileFolderDataset(
    config.VAL_ROOT,
    s2_subdir=getattr(config, "S2_SUBDIR", "S2"),
    s1_subdir=getattr(config, "S1_SUBDIR", "S1"),
    label_subdir=getattr(config, "GEDI_LABEL_SUBDIR", "GEDI_LABEL"),
    mask_subdir=getattr(config, "GEDI_MASK_SUBDIR", "GEDI_MASK"),
    expected_s2_channels=getattr(config, "NUM_BANDS", None),
)

logging.info(f"Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")


# -------------------------
# DataLoaders
# -------------------------
num_workers = getattr(config, "NUM_WORKERS", 4)

train_loader = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    drop_last=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
)


# -------------------------
# Model / loss / optimizer
# -------------------------
model = Sentinel2ResUNet(
    in_channels=config.NUM_BANDS,
    s1_in_channels=config.S1_BANDS
).to(device)

# Good first GEDI choice:
criterion = torch.nn.L1Loss().to(device)
# alternatively:
# criterion = torch.nn.SmoothL1Loss(beta=1.0).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

use_amp = (device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
accum_steps = 4

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    threshold=1e-4,
    min_lr=1e-6,
)

best_val_mae = float("inf")
model_out = getattr(config, "MODEL_OUT", "model/output/model_best_gedi.pth")
Path(model_out).parent.mkdir(parents=True, exist_ok=True)


# -------------------------
# Training Loop
# -------------------------
for epoch in range(config.EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_sse = 0.0
    train_sae = 0.0
    train_n = 0
    train_batches = 0
    skipped_train_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader):
        s2 = batch["s2"].to(device, non_blocking=True)
        s1 = batch["s1"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        m = batch["mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(s2, s1)
            valid = (m > 0.5)

            if valid.sum().item() == 0:
                skipped_train_batches += 1
                continue

            raw_loss = criterion(pred[valid], y[valid])
            loss = raw_loss / accum_steps

        scaler.scale(loss).backward()

        train_loss_sum += raw_loss.item()
        batch_sse, batch_sae, batch_n = masked_sse_mae_count(
            pred.detach().float(), y.float(), m.float()
        )
        train_sse += batch_sse
        train_sae += batch_sae
        train_n += batch_n
        train_batches += 1

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    # flush leftover gradients
    if 'step' in locals() and (step + 1) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    avg_train_loss = train_loss_sum / max(train_batches, 1)
    avg_train_rmse = (train_sse / max(train_n, 1)) ** 0.5
    avg_train_mae = train_sae / max(train_n, 1)

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    val_loss_sum = 0.0
    val_batches = 0
    val_sse = 0.0
    val_sae = 0.0
    val_n = 0
    skipped_val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            s2 = batch["s2"].to(device, non_blocking=True)
            s1 = batch["s1"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            m = batch["mask"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(s2, s1)
                valid = (m > 0.5)

                if valid.sum().item() == 0:
                    skipped_val_batches += 1
                    continue

                vloss = criterion(pred[valid], y[valid])

            val_loss_sum += vloss.item()
            val_batches += 1

            batch_sse, batch_sae, batch_n = masked_sse_mae_count(
                pred.float(), y.float(), m.float()
            )
            val_sse += batch_sse
            val_sae += batch_sae
            val_n += batch_n

    avg_val_loss = val_loss_sum / max(val_batches, 1)
    avg_val_rmse = (val_sse / max(val_n, 1)) ** 0.5
    avg_val_mae = val_sae / max(val_n, 1)

    # For GEDI, MAE is usually the more stable quantity to schedule/save on
    scheduler.step(avg_val_mae)
    current_lr = optimizer.param_groups[0]["lr"]

    logging.info(
        f"Epoch {epoch + 1}/{config.EPOCHS} - "
        f"Train Loss: {avg_train_loss:.4f}, MAE: {avg_train_mae:.4f}, RMSE: {avg_train_rmse:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.4f}, RMSE: {avg_val_rmse:.4f} | "
        f"Train valid px: {train_n}, Val valid px: {val_n} | "
        f"Skipped train batches: {skipped_train_batches}, Skipped val batches: {skipped_val_batches} | "
        f"LR: {current_lr:.2e}"
    )

    if avg_val_mae < best_val_mae:
        best_val_mae = avg_val_mae
        torch.save(model.state_dict(), model_out)
        logging.info(f"Saved new best model to {model_out}")