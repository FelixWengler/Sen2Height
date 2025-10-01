import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import logging
import albumentations as A
from torch.utils.data import DataLoader, IterableDataset, random_split
from torchgeo.datasets import stack_samples
from torchgeo.samplers import RandomGeoSampler
from models.height_net import Sentinel2ResUNet
from datasets.raster_datasets import SentinelImage, DSMImage, SentinelDSMCombo
from utils.metrics import rmse
import config
from torchgeo.datasets.utils import BoundingBox

# Logging setup
log_path = "logs/train_56_50.log"
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(message)s")
logging.info("Starting Sen2Height server training run")

# Device setup
torch.set_num_threads(30)  # Number of CPUs
device = torch.device("cpu")  # cuda for gpu
logging.info(f"Using device: {device}")

# Initialize base dataset
sentinel_ds = SentinelImage(config.SENTINEL_DIR)
dsm_ds = DSMImage(config.DSM_DIR)
full_dataset = SentinelDSMCombo(sentinel_ds, dsm_ds)

# Split into train and validation (80/20)
train_len = int(0.8 * 1000)
val_len = 1000 - train_len

train_sampler = RandomGeoSampler(full_dataset, size=config.PATCH_SIZE, length=train_len)
val_sampler = RandomGeoSampler(full_dataset, size=config.PATCH_SIZE, length=val_len)


# Generate random patches
class GeoPatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, sampler, augment=False):  # set to T for augmentation
        self.dataset = dataset
        self.sampler = sampler
        self.augment = augment

        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ], additional_targets={'mask': 'mask'})

    def __iter__(self):
        for query in self.sampler:
            sample = self.dataset[query]

            image = sample["image"]
            label = sample["label"]

            # Data Augmentation (if enabled)
            if self.augment:
                image_np = image.numpy().transpose(1, 2, 0).astype(np.float32)
                label_np = label.numpy().squeeze(0).astype(np.float32)

                augmented = self.transform(image=image_np, mask=label_np)

                # Convert back to torch
                sample["image"] = torch.from_numpy(augmented["image"].transpose(2, 0, 1))
                sample["label"] = torch.from_numpy(augmented["mask"]).unsqueeze(0)

            yield sample


train_dataset = GeoPatchDataset(full_dataset, sampler=train_sampler)
val_dataset = GeoPatchDataset(full_dataset, sampler=val_sampler)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=28, persistent_workers=True, collate_fn=stack_samples)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, num_workers=28, persistent_workers=True, collate_fn=stack_samples)

# Model setup
model = Sentinel2ResUNet(in_channels=config.NUM_BANDS).to(device)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

best_val_rmse = float("inf")

# Training Loop
for epoch in range(config.EPOCHS):
    model.train()
    total_train_loss = 0.0
    train_batches = 0

    for i, batch in enumerate(train_loader):
        x = batch["image"]
        y = batch["label"]

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_batches += 1

    avg_train_loss = total_train_loss / train_batches if train_batches > 0 else float("inf")
    train_rmse = rmse(pred, y)

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_batches = 0
    val_rmse_total = 0.0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            pred = model(x)
            loss = criterion(pred, y)

            val_loss += loss.item()
            val_rmse_total += rmse(pred, y)
            val_batches += 1

    avg_val_loss = val_loss / val_batches if val_batches > 0 else float("inf")
    avg_val_rmse = val_rmse_total / val_batches if val_batches > 0 else float("inf")

    logging.info(f"Epoch {epoch + 1}/{config.EPOCHS} - "
                 f"Train Loss: {avg_train_loss:.4f}, RMSE: {train_rmse:.4f} | "
                 f"Val Loss: {avg_val_loss:.4f}, RMSE: {avg_val_rmse:.4f}")

    # Save model if validation improves
    if avg_val_rmse < best_val_rmse:
        best_val_rmse = avg_val_rmse
        torch.save(model.state_dict(), "models/output/model_230708_56_50.pth")  # set model output
        logging.info("Saved new best model")