import torch
import rasterio
from rasterio.windows import Window
from models.height_net import Sentinel2HeightNet
import config
import numpy as np
import os

def load_image(path):
    with rasterio.open(path) as src:
        data = src.read()  # shape: (bands, height, width)
        profile = src.profile
    return data, profile

def sliding_window_prediction(model, image_tensor, window_size):
    _, height, width = image_tensor.shape
    output = torch.zeros((1, height, width), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        for i in range(0, height - window_size + 1, window_size):
            for j in range(0, width - window_size + 1, window_size):
                patch = image_tensor[:, i:i+window_size, j:j+window_size].unsqueeze(0).to(config.DEVICE)
                pred = model(patch).cpu().squeeze(0)
                output[:, i:i+window_size, j:j+window_size] = pred

    return output.squeeze(0).numpy()

def save_prediction(output_array, profile, out_path):
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(output_array.astype(rasterio.float32), 1)

if __name__ == "__main__":
    image_path = os.path.join(config.SENTINEL_DIR, "example_image.tif")  # Input image to predict
    model_path = "checkpoints/model.pth"  # Pretrained model weights
    output_path = config.PREDICTION_OUTPUT

    image_np, profile = load_image(image_path)
    image_tensor = torch.tensor(image_np, dtype=torch.float32)

    model = Sentinel2HeightNet(in_channels=config.NUM_BANDS)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)

    prediction = sliding_window_prediction(model, image_tensor, config.PREDICTION_PATCH_SIZE)
    save_prediction(prediction, profile, output_path)

    print(f"Prediction saved to {output_path}")