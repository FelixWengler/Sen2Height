import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, IterableDataset, random_split
from torchgeo.datasets import stack_samples
from torchgeo.samplers import RandomGeoSampler
from models.height_net import Sentinel2UNet
from datasets.raster_datasets import SentinelImage, DSMImage, SentinelDSMCombo
from utils.metrics import rmse
import config

# Initialize base dataset
sentinel_ds = SentinelImage(config.SENTINEL_DIR)
dsm_ds = DSMImage(config.DSM_DIR)
full_dataset = SentinelDSMCombo(sentinel_ds, dsm_ds)

# Split into train and validation (e.g., 80/20 split)
train_len = int(0.8 * 1000)
val_len = 1000 - train_len

train_sampler = RandomGeoSampler(full_dataset, size=config.PATCH_SIZE, length=train_len)
val_sampler = RandomGeoSampler(full_dataset, size=config.PATCH_SIZE, length=val_len)

class GeoPatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, sampler, augment=False):
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

            image = sample["image"]  # torch.Tensor [C, H, W]
            label = sample["label"]  # torch.Tensor [1, H, W]

            if self.augment:
                # Albumentations expects numpy HWC, float32
                image_np = image.numpy().transpose(1, 2, 0).astype(np.float32)
                label_np = label.numpy().squeeze(0).astype(np.float32)

                augmented = self.transform(image=image_np, mask=label_np)

                # Convert back to torch
                sample["image"] = torch.from_numpy(augmented["image"].transpose(2, 0, 1))
                sample["label"] = torch.from_numpy(augmented["mask"]).unsqueeze(0)

            yield sample


train_dataset = GeoPatchDataset(full_dataset, sampler=train_sampler)
val_dataset = GeoPatchDataset(full_dataset, sampler=val_sampler)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=0, collate_fn=stack_samples)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, num_workers=0, collate_fn=stack_samples)

# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Sentinel2UNet(in_channels=config.NUM_BANDS).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

best_val_rmse = float("inf")

for epoch in range(config.EPOCHS):
    model.train()
    total_train_loss = 0.0
    train_batches = 0

    for batch in train_loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

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

    print(f"Epoch {epoch+1}/{config.EPOCHS} - "
          f"Train Loss: {avg_train_loss:.4f}, RMSE: {train_rmse:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, RMSE: {avg_val_rmse:.4f}")

    # Save best model
    if avg_val_rmse < best_val_rmse:
        best_val_rmse = avg_val_rmse
        torch.save(model.state_dict(), "models/output/model_29052025.pth")
        print("✅ Saved new best model")