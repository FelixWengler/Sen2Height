import torch
import rasterio
from rasterio.windows import Window
from models.height_net import Sentinel2UNet
from tqdm import tqdm
import config
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(path):
    with rasterio.open(path) as src:
        data = src.read()  # shape: (bands, height, width)
        profile = src.profile
    return data, profile

def sliding_window_prediction(model, image_tensor, window_size, stride):
    _, height, width = image_tensor.shape
    output_sum = torch.zeros((1, height, width), dtype=torch.float32)
    output_count = torch.zeros((1, height, width), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, height - window_size + 1, stride), desc="🧠 Predicting rows"):
            for j in range(0, width - window_size + 1, stride):
                patch = image_tensor[:, i:i+window_size, j:j+window_size].unsqueeze(0).to(device)
                pred = model(patch).cpu().squeeze(0)  # shape [1, H, W]
                output_sum[:, i:i+window_size, j:j+window_size] += pred
                output_count[:, i:i+window_size, j:j+window_size] += 1

    # Avoid division by zero (if any area is untouched)
    output_count[output_count == 0] = 1
    output_avg = output_sum / output_count
    return output_avg.squeeze(0).numpy()

def save_prediction(output_array, profile, out_path):
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(output_array.astype(rasterio.float32), 1)

if __name__ == "__main__":
    image_path = os.path.join(config.SENTINEL_DIR)  # Input image to predict
    model_path = "models/output/model_341_16052025.pth"  # Pretrained model weights
    output_path = config.PREDICTION_OUTPUT

    image_np, profile = load_image(image_path)
    image_tensor = torch.tensor(image_np, dtype=torch.float32)
    image_tensor = torch.clamp((image_tensor - 1000) / 10000.0, 0, 1)

    model = Sentinel2UNet(in_channels=config.NUM_BANDS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    prediction = sliding_window_prediction(
        model,
        image_tensor,
        window_size=config.PREDICTION_PATCH_SIZE,
        stride=config.PREDICTION_PATCH_SIZE // 2  # 50% overlap
    )

    save_prediction(prediction, profile, output_path)

    print(f"Prediction saved to {output_path}")