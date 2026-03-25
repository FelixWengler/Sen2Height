from pathlib import Path
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np


class S2S1GEDITileFolderDataset(Dataset):
    """
    Dataset that reads tiled Sentinel-2 + Sentinel-1 + GEDI label/mask chips.

    Expected structure:
      root/
        S2/
          x0085_y0059_2019.tif        (C=?, H=256, W=256)
        S1/
          x0085_y0059_2019.tif        (C=2, H=128, W=128) or (C=2, H=256, W=256) depending on preprocessing
        GEDI_LABEL/
          x0085_y0059_2019.tif        (C=1, H=256, W=256)  RH95 at valid GEDI pixels, nodata elsewhere
        GEDI_MASK/
          x0085_y0059_2019.tif        (C=1, H=256, W=256)  1 where GEDI exists, 0 elsewhere
    """

    def __init__(
        self,
        root_dir,
        s2_subdir="S2",
        s1_subdir="S1",
        label_subdir="gedi_label",
        mask_subdir="gedi_mask",
        s2_divisor=10000.0,
        s2_clamp01=True,
        s1_nodata=-32768.0,
        s1_use_log1p=True,
        expected_s2_channels=None,
        transforms=None,
    ):
        self.root = Path(root_dir)
        self.s2_dir = self.root / s2_subdir
        self.s1_dir = self.root / s1_subdir
        self.label_dir = self.root / label_subdir
        self.mask_dir = self.root / mask_subdir

        if not self.s2_dir.exists():
            raise FileNotFoundError(f"Missing S2 directory: {self.s2_dir}")
        if not self.s1_dir.exists():
            raise FileNotFoundError(f"Missing S1 directory: {self.s1_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Missing GEDI label directory: {self.label_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing GEDI mask directory: {self.mask_dir}")

        self.s2_divisor = float(s2_divisor)
        self.s2_clamp01 = bool(s2_clamp01)
        self.s1_nodata = float(s1_nodata) if s1_nodata is not None else None
        self.s1_use_log1p = bool(s1_use_log1p)
        self.expected_s2_channels = expected_s2_channels
        self.transforms = transforms

        # Require all four files with the same name
        self.files = sorted([
            f for f in self.s2_dir.glob("*.tif")
            if (self.s1_dir / f.name).exists()
            and (self.label_dir / f.name).exists()
            and (self.mask_dir / f.name).exists()
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No paired S2/S1/GEDI_LABEL/GEDI_MASK tiles found in {root_dir}")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read()
        return arr.astype(np.float32)

    def __getitem__(self, idx):
        s2_path = self.files[idx]
        s1_path = self.s1_dir / s2_path.name
        label_path = self.label_dir / s2_path.name
        mask_path = self.mask_dir / s2_path.name

        # ---- S2 ----
        s2 = torch.from_numpy(self._read(s2_path)).float()   # (C,H,W)

        if self.expected_s2_channels is not None and s2.shape[0] != self.expected_s2_channels:
            raise ValueError(
                f"Expected {self.expected_s2_channels} S2 channels, got {s2.shape[0]} for {s2_path}"
            )

        s2 = s2 / self.s2_divisor
        if self.s2_clamp01:
            s2 = torch.clamp(s2, 0.0, 1.0)

        # ---- S1 ----
        s1 = torch.from_numpy(self._read(s1_path)).float()   # usually (2,128,128)

        if self.s1_nodata is not None:
            s1 = torch.where(s1 == self.s1_nodata, torch.zeros_like(s1), s1)

        if self.s1_use_log1p:
            s1 = torch.log1p(torch.clamp(s1, min=0.0))

        # Add log-ratio channel: VH - VV
        vh = s1[0:1]
        vv = s1[1:2]
        ratio = vh - vv
        s1 = torch.cat([s1, ratio], dim=0)  # now (3,H,W)

        # ---- GEDI label ----
        label = torch.from_numpy(self._read(label_path)).float()
        if label.ndim == 2:
            label = label[None, ...]
        elif label.shape[0] != 1:
            raise ValueError(f"Expected 1 GEDI label band, got {label.shape[0]} for {label_path}")

        # ---- GEDI mask ----
        mask = torch.from_numpy(self._read(mask_path)).float()
        if mask.ndim == 2:
            mask = mask[None, ...]
        elif mask.shape[0] != 1:
            raise ValueError(f"Expected 1 GEDI mask band, got {mask.shape[0]} for {mask_path}")

        # force mask to binary float tensor
        mask = (mask > 0).float()

        # Optional safety: zero out invalid label values where mask == 0
        label = torch.where(mask > 0, label, torch.zeros_like(label))

        sample = {
            "s2": s2,
            "s1": s1,
            "label": label,
            "mask": mask,
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample