import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from models.height_net import Sentinel2UNet
import config
from torch.utils.data import DataLoader, IterableDataset
from torchgeo.samplers import RandomGeoSampler
from torchgeo.datasets import stack_samples
from datasets.raster_datasets import SentinelImage, DSMImage, SentinelDSMCombo


# Get absolute path to project root (parent of utils/)
project_root = os.path.dirname(os.path.dirname(__file__))
sentinel_path = os.path.join(project_root, config.SENTINEL_DIR)
dsm_path = os.path.join(project_root, config.DSM_DIR)

# Construct correct absolute model path
model_path = os.path.join(project_root, "models", "output", "model_16052025.pth")
assert os.path.exists(model_path), f"❌ Model file not found: {model_path}"

# Load model
model = Sentinel2UNet(in_channels=config.NUM_BANDS)
model.load_state_dict(torch.load(model_path))
model.eval()

print("✅ Model loaded successfully.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Reconstruct datasets
sentinel_ds = SentinelImage(sentinel_path)
dsm_ds = DSMImage(dsm_path)
full_dataset = SentinelDSMCombo(sentinel_ds, dsm_ds)

# Validation sampler and loader (same logic as training)
val_sampler = RandomGeoSampler(full_dataset, size=config.PATCH_SIZE, length=200)

class GeoPatchDataset(IterableDataset):
    def __init__(self, dataset, sampler):
        self.dataset = dataset
        self.sampler = sampler

    def __iter__(self):
        for query in self.sampler:
            yield self.dataset[query]

val_dataset = GeoPatchDataset(full_dataset, sampler=val_sampler)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=0,
    collate_fn=stack_samples
)

actuals = []
predictions = []

with torch.no_grad():
    for batch in val_loader:
        x = batch["image"].to(device)
        y_true = batch["label"].to(device)
        y_pred = model(x)

        # Flatten and store
        actuals.append(y_true.cpu().flatten())
        predictions.append(y_pred.cpu().flatten())

# Combine all pixels
actuals = torch.cat(actuals).numpy()
predictions = torch.cat(predictions).numpy()

# Convert to numpy
actuals_np = np.array(actuals)
preds_np = np.array(predictions)

# Optionally filter out background
mask = actuals_np > 0
actuals_np = actuals_np[mask]
preds_np = preds_np[mask]

# Metrics
mae = mean_absolute_error(actuals_np, preds_np)
rmse_val = np.sqrt(mean_squared_error(actuals_np, preds_np))
r2 = r2_score(actuals_np, preds_np)
bias = np.mean(preds_np - actuals_np)

with open("figures/apu_16052025.txt", "w") as f:
    f.write("APU Statistics\n")
    f.write(f"MAE   : {mae:.2f} cm\n")
    f.write(f"RMSE  : {rmse_val:.2f} cm\n")
    f.write(f"R²    : {r2:.4f}\n")
    f.write(f"Bias  : {bias:.2f} cm\n")
print("✅ Stats saved to figures/apu_16052025.txt")

plt.figure(figsize=(6, 6))
plt.scatter(actuals, predictions, s=1, alpha=0.3, label="pixels")
plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', label="perfect prediction")

plt.xlabel("Actual Height (cm)")
plt.ylabel("Predicted Height (cm)")
plt.title("Actual vs. Predicted Heights (APU Plot)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.xlim(0, 8000)
plt.ylim(0, 8000)
plt.savefig("figures/apu_plot16052025.png", dpi=300)
plt.show()