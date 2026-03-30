import os
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

from model.height_net import Sentinel2ResUNet
import config

PATCH = config.PREDICTION_PATCH_SIZE
STRIDE = PATCH // 2
BATCH = getattr(config, "PREDICTION_BATCH_SIZE", 4)
TILE = 1024
HALO = PATCH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_hann2d(size: int, eps: float = 1e-6) -> torch.Tensor:
    w1d = torch.hann_window(size, periodic=False)
    w2d = torch.outer(w1d, w1d)
    w2d = w2d / (w2d.max() + eps)
    return torch.clamp(w2d, min=0.05).unsqueeze(0)  # (1,H,W)


blend_window = make_hann2d(PATCH)


def build_valid_mask(image_np: np.ndarray, nodata_value, nodata_eps=0):
    """
    image_np: (C,H,W)
    returns: (H,W) bool where True means valid
    """
    if nodata_value is not None:
        if nodata_eps > 0:
            nodata = np.all(
                np.abs(image_np.astype(np.float32) - float(nodata_value)) <= nodata_eps,
                axis=0
            )
        else:
            nodata = np.all(image_np == nodata_value, axis=0)
    else:
        nodata = np.all(image_np == 0, axis=0)

    return ~nodata


def normalize_s2(tile_np: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(tile_np).float()
    t = torch.clamp(t / 10000.0, 0.0, 1.0)
    return t


def normalize_s1_and_make_3ch(s1_np: np.ndarray, nodata_value=None) -> torch.Tensor:
    """
    Input: raw S1 patch aligned to S2 grid, shape (2,H,W)
    Output: normalized S1, shape (3,H,W):
      [VH_log, VV_log, (VH_log - VV_log)]
    """
    s1 = torch.from_numpy(s1_np).float()

    if nodata_value is not None:
        s1 = torch.where(s1 == float(nodata_value), torch.zeros_like(s1), s1)

    s1 = torch.clamp(s1, min=0.0)
    s1 = torch.log1p(s1)

    vh = s1[0:1]
    vv = s1[1:2]
    ratio = vh - vv
    s1 = torch.cat([s1, ratio], dim=0)  # (3,H,W)

    return s1


def predict_tile(model, s2_np, s1_np, s2_nodata, s1_nodata, nodata_eps=0, out_nodata_value=None):
    """
    s2_np: (C2,H,W) raw 10m
    s1_np: (2,H,W)  raw 10m aligned to S2 grid
    returns: (H,W) float32 with nodata filled where invalid
    """
    if out_nodata_value is None:
        out_nodata_value = float(s2_nodata) if s2_nodata is not None else -9999.0

    valid_mask_np = build_valid_mask(s2_np, s2_nodata, nodata_eps=nodata_eps)

    if not valid_mask_np.any():
        _, H, W = s2_np.shape
        return np.ones((H, W), dtype=np.float32) * out_nodata_value

    s2 = normalize_s2(s2_np)
    s1_raw = s1_np  # keep as numpy; patchwise preprocessing mirrors training

    _, H, W = s2.shape
    out_sum = torch.zeros((H, W), dtype=torch.float32)
    w_sum = torch.zeros((H, W), dtype=torch.float32)
    valid = torch.from_numpy(valid_mask_np.astype(np.float32))

    coords = []
    patches_s2 = []
    patches_s1 = []

    use_amp = (device.type == "cuda")

    for i in range(0, H - PATCH + 1, STRIDE):
        for j in range(0, W - PATCH + 1, STRIDE):
            vm = valid[i:i+PATCH, j:j+PATCH]
            if vm.sum().item() == 0:
                continue

            s2_patch = s2[:, i:i+PATCH, j:j+PATCH]                  # (C,256,256)
            s1_patch_np = s1_raw[:, i:i+PATCH, j:j+PATCH]          # (2,256,256)
            s1_patch = normalize_s1_and_make_3ch(
                s1_patch_np,
                nodata_value=s1_nodata
            )                                                      # (3,256,256)

            coords.append((i, j))
            patches_s2.append(s2_patch)
            patches_s1.append(s1_patch)

            if len(patches_s2) == BATCH:
                batch_s2 = torch.stack(patches_s2, dim=0).to(device, non_blocking=True)
                batch_s1 = torch.stack(patches_s1, dim=0).to(device, non_blocking=True)

                with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(batch_s2, batch_s1)

                pred = pred.float().detach().cpu()
                if pred.ndim == 3:
                    pred = pred.unsqueeze(1)

                for k, (ii, jj) in enumerate(coords):
                    vm2 = valid[ii:ii+PATCH, jj:jj+PATCH]
                    w = blend_window[0] * vm2
                    out_sum[ii:ii+PATCH, jj:jj+PATCH] += pred[k, 0] * w
                    w_sum[ii:ii+PATCH, jj:jj+PATCH] += w

                coords.clear()
                patches_s2.clear()
                patches_s1.clear()

    if patches_s2:
        batch_s2 = torch.stack(patches_s2, dim=0).to(device, non_blocking=True)
        batch_s1 = torch.stack(patches_s1, dim=0).to(device, non_blocking=True)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(batch_s2, batch_s1)

        pred = pred.float().detach().cpu()
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        for k, (ii, jj) in enumerate(coords):
            vm2 = valid[ii:ii+PATCH, jj:jj+PATCH]
            w = blend_window[0] * vm2
            out_sum[ii:ii+PATCH, jj:jj+PATCH] += pred[k, 0] * w
            w_sum[ii:ii+PATCH, jj:jj+PATCH] += w

    out = torch.empty((H, W), dtype=torch.float32)
    m = (w_sum > 0)
    out[m] = out_sum[m] / w_sum[m]
    out[~m] = float(out_nodata_value)

    return out.numpy().astype(np.float32)


def main():
    s2_path = config.PREDICTION_INPUT
    s1_path = config.PREDICTION_INPUT_S1_ALIGNED
    model_path = config.PREDICTION_MODEL
    output_path = config.PREDICTION_OUTPUT

    model = Sentinel2ResUNet(
        in_channels=config.NUM_BANDS,
        s1_in_channels=config.S1_BANDS
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()

    with rasterio.open(s2_path) as s2_src, rasterio.open(s1_path) as s1_src:
        assert s2_src.crs == s1_src.crs, "CRS mismatch between S2 and S1"
        assert s2_src.transform == s1_src.transform, "Transform mismatch: S1 must be aligned to S2 grid"
        assert s2_src.width == s1_src.width and s2_src.height == s1_src.height, "Shape mismatch"

        profile = s2_src.profile.copy()
        s2_nodata = s2_src.nodata
        s1_nodata = s1_src.nodata
        H, W = s2_src.height, s2_src.width

        out_nodata_value = float(s2_nodata) if s2_nodata is not None else -9999.0
        out_profile = profile.copy()
        out_profile.update(
            count=1,
            dtype=rasterio.float32,
            nodata=out_nodata_value
        )

        with rasterio.open(output_path, "w", **out_profile) as dst:
            for top in tqdm(range(0, H, TILE), desc="Tiles"):
                tile_h = min(TILE, H - top)
                for left in range(0, W, TILE):
                    tile_w = min(TILE, W - left)

                    r0 = max(0, top - HALO)
                    c0 = max(0, left - HALO)
                    r1 = min(H, top + tile_h + HALO)
                    c1 = min(W, left + tile_w + HALO)

                    read_h0 = r1 - r0
                    read_w0 = c1 - c0

                    pad_h = (PATCH - (read_h0 - PATCH) % STRIDE - 1) % STRIDE
                    pad_w = (PATCH - (read_w0 - PATCH) % STRIDE - 1) % STRIDE

                    read_h = min(read_h0 + pad_h, H - r0)
                    read_w = min(read_w0 + pad_w, W - c0)

                    window = Window(c0, r0, read_w, read_h)

                    s2_np = s2_src.read(window=window)
                    s1_np = s1_src.read(window=window)

                    pred = predict_tile(
                        model,
                        s2_np,
                        s1_np,
                        s2_nodata=s2_nodata,
                        s1_nodata=s1_nodata,
                        nodata_eps=0,
                        out_nodata_value=out_nodata_value,
                    )

                    inner_top = top - r0
                    inner_left = left - c0
                    pred_center = pred[
                        inner_top: inner_top + tile_h,
                        inner_left: inner_left + tile_w
                    ]

                    pred_center = np.clip(pred_center, 0.0, None).astype(np.float32)
                    dst.write(pred_center, 1, window=Window(left, top, tile_w, tile_h))


if __name__ == "__main__":
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    main()