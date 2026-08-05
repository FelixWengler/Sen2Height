import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, Subset

from model.height_net import Sentinel2ResUNet
from datasets.raster_datasets import SplineS1DSMDataset
#from utils.metrics import rmse
from utils.WeightedL1 import BinWeightedL1
import config


class RandomSubsetSampler(Sampler):
    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.num_samples = min(
            int(num_samples),
            len(data_source),
        )

    def __iter__(self):
        indices = torch.randperm(
            len(self.data_source)
        )[:self.num_samples]

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


def sse_and_count(pred: torch.Tensor, target: torch.Tensor):
    # pred/target: [B,1,H,W]
    diff = pred - target
    sse = torch.sum(diff * diff).item()
    n = diff.numel()
    return sse, n

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

dataset_arguments = {
    "s1_nodata": getattr(
        config,
        "S1_NODATA",
        -32768.0,
    ),
    "s1_scale_factor": getattr(
        config,
        "S1_SCALE_FACTOR",
        100.0,
    ),
    "add_s1_ratio": getattr(
        config,
        "ADD_S1_RATIO",
        True,
    ),
    "excluded_years": getattr(
        config,
        "EXCLUDED_YEARS",
        (),
    ),
}

train_ds = SplineS1DSMDataset(
    root_dir=config.TRAIN_ROOT,
    **dataset_arguments,
)

val_ds = SplineS1DSMDataset(
    root_dir=config.VAL_ROOT,
    **dataset_arguments,
)

logging.info(
    f"Train chips: {len(train_ds)} | "
    f"Validation chips: {len(val_ds)}"
)

sample = train_ds[0]

logging.info(
    "First sample shapes: "
    f"spline={tuple(sample['s2'].shape)}, "
    f"S1={tuple(sample['s1'].shape)}, "
    f"label={tuple(sample['label'].shape)}"
)

if sample["s2"].shape[0] != config.NUM_BANDS:
    raise RuntimeError(
        f"NUM_BANDS={config.NUM_BANDS}, but spline "
        f"contains {sample['s2'].shape[0]} channels."
    )

if sample["s1"].shape[0] != config.S1_BANDS:
    raise RuntimeError(
        f"S1_BANDS={config.S1_BANDS}, but S1 input "
        f"contains {sample['s1'].shape[0]} channels."
    )

logging.info(
    "First sample ranges: "
    f"spline=["
    f"{sample['s2'].min().item():.4f}, "
    f"{sample['s2'].max().item():.4f}], "
    f"S1=["
    f"{sample['s1'].min().item():.4f}, "
    f"{sample['s1'].max().item():.4f}], "
    f"label=["
    f"{sample['label'].min().item():.4f}, "
    f"{sample['label'].max().item():.4f}], "
    f"valid_fraction="
    f"{sample['label_valid_mask'].float().mean().item():.4f}"
)
# -------------------------
# DataLoaders
# -------------------------

train_sampler = RandomSubsetSampler(
    train_ds,
    num_samples=config.TRAIN_CHIPS_PER_EPOCH,
)


validation_chips = min(
    getattr(
        config,
        "VALIDATION_CHIPS",
        len(val_ds),
    ),
    len(val_ds),
)

validation_seed = getattr(
    config,
    "VALIDATION_SEED",
    42,
)

generator = torch.Generator().manual_seed(
    validation_seed
)

validation_indices = torch.randperm(
    len(val_ds),
    generator=generator,
)[:validation_chips].tolist()

val_subset = Subset(
    val_ds,
    validation_indices,
)

num_workers = getattr(
    config,
    "NUM_WORKERS",
    4,
)

persistent_workers = num_workers > 0

train_loader = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    sampler=train_sampler,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    drop_last=True,
)

val_loader = DataLoader(
    val_subset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=device.type == "cuda",
    persistent_workers=persistent_workers,
    drop_last=False,
)


# -------------------------
# Model / loss / optimizer
# -------------------------
model = Sentinel2ResUNet(in_channels=config.NUM_BANDS, s1_in_channels=config.S1_BANDS).to(device)
criterion = BinWeightedL1().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
use_amp = (device.type=="cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
accum_steps = max(
    1,
    int(getattr(config, "ACCUM_STEPS", 1)),
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode = "min",
    factor = 0.5,
    patience = 5,
    threshold = 1e-4,
    min_lr = 1e-6, 
)

best_val_rmse = float("inf")
model_out = getattr(config, "MODEL_OUT", "model/output/model_best.pth")
Path(model_out).parent.mkdir(parents=True, exist_ok=True)

# -------------------------
# Training Loop
# -------------------------
for epoch in range(config.EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_sse = 0.0
    train_n = 0
    train_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader):
        s2 = batch["s2"].to(device, non_blocking=True)
        s1 = batch["s1"].to(device, non_blocking=True)
        y  = batch["label"].to(device, non_blocking=True)

        valid_mask = batch["label_valid_mask"].to(
            device,
            non_blocking=True,
        )

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(s2, s1)
            raw_loss = criterion(pred, y)
            loss = raw_loss / accum_steps

        scaler.scale(loss).backward()

        train_loss_sum += raw_loss.item()
        diff = pred.detach().float() - y.float()
        valid_diff = diff[valid_mask]
        train_sse += torch.sum(valid_diff * valid_diff).item()
        train_n += valid_diff.numel()
        train_batches += 1

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    # flush leftover gradients
    if (step + 1) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    avg_train_loss = train_loss_sum / max(train_batches, 1)
    avg_train_rmse = (train_sse / max(train_n, 1)) ** 0.5

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    val_loss_sum = 0.0
    val_batches = 0
    val_sse = 0.0
    val_n = 0

    with torch.no_grad():
        for batch in val_loader:
            s2 = batch["s2"].to(device, non_blocking=True)
            s1 = batch["s1"].to(device, non_blocking=True)
            y = batch["label"].to(
                device,
                non_blocking=True,
            )

            valid_mask = batch[
                "label_valid_mask"
            ].to(
                device,
                non_blocking=True,
            )

            with torch.cuda.amp.autocast(
                enabled=use_amp
            ):
                pred = model(s2, s1)
                vloss = criterion(pred, y)

            val_loss_sum += vloss.item()
            val_batches += 1

            diff = pred.float() - y.float()
            valid_diff = diff[valid_mask]

            val_sse += torch.sum(
                valid_diff * valid_diff
            ).item()

            val_n += valid_diff.numel()

    avg_val_loss = val_loss_sum / max(val_batches, 1)
    avg_val_rmse = (val_sse / max(val_n, 1)) ** 0.5

    scheduler.step(avg_val_rmse)
    current_lr = optimizer.param_groups[0]["lr"]

    logging.info(
        f"Epoch {epoch + 1}/{config.EPOCHS} - "
        f"Train Loss: {avg_train_loss:.4f}, RMSE: {avg_train_rmse:.4f} | "
        f"Val Loss: {avg_val_loss:.4f}, RMSE: {avg_val_rmse:.4f} | "
        f"LR: {current_lr:.2e}"
    )

    if avg_val_rmse < best_val_rmse:
        best_val_rmse = avg_val_rmse
        torch.save(model.state_dict(), model_out)
        logging.info(f"Saved new best model to {model_out}")