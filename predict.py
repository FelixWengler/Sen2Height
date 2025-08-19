import torch
import rasterio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import os
from models.height_net import Sentinel2ResUNet
import config


##helpers##

def make_hann2d(size: int, eps: float = 1e-6) -> torch.Tensor:
    """2-D Hann window normalized to max=1 (avoid zero weights at exact border)."""
    w1d = torch.hann_window(size, periodic=False)  # [0..size-1]
    w2d = torch.outer(w1d, w1d)
    w2d = w2d / (w2d.max() + eps)
    # clip a tiny floor so borders still contribute a bit, reducing ‘stitch’ risk
    return torch.clamp(w2d, min=0.05).unsqueeze(0)  # shape (1,H,W)

def build_valid_mask(image_np: np.ndarray, nodata_value, nodata_eps=0):
    """
    Returns a boolean mask (H,W) where True = valid data.
    If nodata_value is None, a common S2 heuristic is 'all bands == 0' -> NoData.
    """
    _, H, W = image_np.shape
    if nodata_value is not None:
        nodata = np.all(image_np == nodata_value, axis=0)
    else:
        # Heuristic: all bands exactly zero means NoData (typical for S2 L2A GeoTIFFs)
        nodata = np.all(image_np == 0, axis=0)
        # If your data are scaled floats, consider nodata_eps tolerance:
        # nodata = np.all(np.abs(image_np) <= nodata_eps, axis=0)

    valid = ~nodata
    return valid  # (H,W) bool


# Force CPU-only for server
device = torch.device("cpu")
torch.set_num_threads(config.PREDICTION_WORKERS)

# --- Dataset for patch extraction ---
class SlidingWindowDataset(Dataset):
    def __init__(self, image_tensor, window_size, stride):
        self.image_tensor = image_tensor
        self.window_size = window_size
        self.stride = stride
        self.patches = []

        _, height, width = image_tensor.shape
        for i in range(0, height - window_size + 1, stride):
            for j in range(0, width - window_size + 1, stride):
                self.patches.append((i, j))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        i, j = self.patches[idx]
        patch = self.image_tensor[:, i:i+self.window_size, j:j+self.window_size]
        return patch, i, j

# --- Helpers ---
def load_image(path):
    with rasterio.open(path) as src:
        data = src.read()  # (bands, H, W)
        profile = src.profile
    return data, profile

def save_prediction(output_array, profile, out_path):
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(output_array.astype(rasterio.float32), 1)

# --- Main prediction ---
if __name__ == "__main__":
    print("Loading input image...")
    image_path = os.path.join(config.PREDICTION_INPUT)
    model_path = os.path.join(config.PREDICTION_MODEL)
    output_path = os.path.join(config.PREDICTION_OUTPUT)

    image_np, profile = load_image(image_path)
    image_tensor = torch.tensor(image_np, dtype=torch.float32)
    image_tensor = torch.clamp(image_tensor / 10000.0, 0, 1)

    model = Sentinel2ResUNet(in_channels=config.NUM_BANDS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()


    window_size = config.PREDICTION_PATCH_SIZE
    stride = window_size // 4   # <- more overlap than 50% helps reduce blocking

    dataset = SlidingWindowDataset(image_tensor, window_size, stride)
    loader = DataLoader(dataset, batch_size=config.PREDICTION_BATCH_SIZE,
                        num_workers=config.PREDICTION_WORKERS, pin_memory=False)

    _, height, width = image_tensor.shape
    output_sum = torch.zeros((1, height, width), dtype=torch.float32)
    weight_sum = torch.zeros((1, height, width), dtype=torch.float32)

    # Build blend window (1, Hpatch, Wpatch)
    blend_window = make_hann2d(window_size)

    # Build valid-mask from original (H,W) -> tensor (1,H,W)
    nodata_value = profile.get("nodata", None)
    valid_mask_np = build_valid_mask(image_np, nodata_value)
    valid_mask = torch.from_numpy(valid_mask_np.astype(np.float32)).unsqueeze(0)

    print(f"Starting prediction using {config.PREDICTION_WORKERS} workers...")
    with torch.no_grad():
        for patches, is_, js_ in tqdm(loader, desc="Predicting", total=len(loader)):
            preds = model(patches.to(device))      # (B,1,ps,ps) or (B,ps,ps)
            preds = preds.cpu()

            # Ensure channel-first (B,1,H,W)
            if preds.ndim == 3:
                preds = preds.unsqueeze(1)

            for k in range(preds.shape[0]):
                i, j = is_[k].item(), js_[k].item()

                # local valid mask slice & weights
                vm = valid_mask[:, i:i+window_size, j:j+window_size]    # (1,ps,ps)
                w = blend_window * vm                                   # down-weight/zero NoData

                # weighted accumulation
                output_sum[:, i:i+window_size, j:j+window_size] += preds[k] * w
                weight_sum[:, i:i+window_size, j:j+window_size] += w

    # Avoid divide by zero
    safe_weight = weight_sum.clone()
    safe_weight[safe_weight == 0] = 1.0

    output_avg = output_sum / safe_weight
    output_np = output_avg.squeeze(0).numpy()
    save_prediction(output_np, profile, output_path)
    print(f"✅ Prediction saved to {output_path}")
