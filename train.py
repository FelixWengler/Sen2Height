import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.height_net import Sentinel2ResUNet
from datasets.raster_datasets import S2DSMTileFolderDataset
from utils.metrics import rmse
from utils.WeightedL1 import BinWeightedL1
import config

# -------------------------
# Logging setup
# -------------------------
log_path = getattr(config, "LOG_PATH", "logs/train_tiled.log")
Path(log_path).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)
logging.info("Starting Sen2Height tiled training run")

# -------------------------
# Device + threads
# -------------------------
torch.set_num_threads(getattr(config, "NUM_THREADS", 30))
device = torch.device(getattr(config, "DEVICE", "cuda"))  # "cuda" if available
logging.info(f"Using device: {device}")

# -------------------------
# Datasets
# -------------------------
# Expected folder structure:
# TRAIN_ROOT/
#   S2/*.tif
#   BDOM/*.tif
# VAL_ROOT/
#   S2/*.tif
#   BDOM/*.tif
train_ds = S2DSMTileFolderDataset(config.TRAIN_ROOT)
val_ds = S2DSMTileFolderDataset(config.VAL_ROOT)

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
model = Sentinel2ResUNet(in_channels=config.NUM_BANDS).to(device)
criterion = BinWeightedL1()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    threshold=1e-4,
    min_lr=1e-6,
    verbose=True,
)

best_val_rmse = float("inf")
model_out = getattr(config, "MODEL_OUT", "models/output/model_best.pth")
Path(model_out).parent.mkdir(parents=True, exist_ok=True)

# -------------------------
# Training Loop
# -------------------------
for epoch in range(config.EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_rmse_sum = 0.0
    train_batches = 0

    for batch in train_loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient Clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss_sum += loss.item()
        train_rmse_sum += rmse(pred.detach(), y)
        train_batches += 1

    avg_train_loss = train_loss_sum / max(train_batches, 1)
    avg_train_rmse = train_rmse_sum / max(train_batches, 1)

    # Validation
    model.eval()
    val_loss_sum = 0.0
    val_rmse_sum = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            pred = model(x)
            loss = criterion(pred, y)

            val_loss_sum += loss.item()
            val_rmse_sum += rmse(pred, y)
            val_batches += 1

    avg_val_loss = val_loss_sum / max(val_batches, 1)
    avg_val_rmse = val_rmse_sum / max(val_batches, 1)

    # Update LR
    scheduler.step(avg_val_rmse)
    current_lr = optimizer.param_groups[0]["lr"]

    logging.info(
        f"Epoch {epoch + 1}/{config.EPOCHS} - "
        f"Train Loss: {avg_train_loss:.4f}, RMSE: {avg_train_rmse:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}, RMSE: {avg_val_rmse:.4f} | "
        f"LR: {current_lr:.2e}"
    )

    # Save best model
    if avg_val_rmse < best_val_rmse:
        best_val_rmse = avg_val_rmse
        torch.save(model.state_dict(), model_out)
        logging.info(f"Saved new best model to {model_out}")
