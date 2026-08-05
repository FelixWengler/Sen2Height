from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SplineS1DSMDataset(Dataset):
    """
    Read paired spline, Sentinel-1, and DSM NPZ chips.

    Expected structure:

        root/
            spline/
                X0053_Y0049/
                    chip_001_r0000_c0000_2023.npz
            S1/
                X0053_Y0049/
                    chip_001_r0000_c0000_2023.npz
            dsm/
                X0053_Y0049/
                    chip_001_r0000_c0000_2023.npz

    Matching samples must have the same relative path beneath
    the spline, S1, and dsm directories.
    """

    def __init__(
        self,
        root_dir,
        spline_subdir="spline",
        s1_subdir="S1",
        dsm_subdir="dsm",
        spline_scale_factor=10000.0,
        s1_nodata=-32768.0,
        s1_scale_factor=100.0,
        add_s1_ratio=True,
        transforms=None,
        excluded_years=None,
    ):
        self.root = Path(root_dir)

        self.spline_dir = self.root / spline_subdir
        self.s1_dir = self.root / s1_subdir
        self.dsm_dir = self.root / dsm_subdir

        for directory, name in (
            (self.spline_dir, "spline"),
            (self.s1_dir, "S1"),
            (self.dsm_dir, "DSM"),
        ):
            if not directory.exists():
                raise FileNotFoundError(
                    f"Missing {name} directory: {directory}"
                )

        self.s1_nodata = s1_nodata
        self.s1_scale_factor = s1_scale_factor
        self.spline_scale_factor = spline_scale_factor
        self.add_s1_ratio = bool(add_s1_ratio)
        self.transforms = transforms

        if excluded_years is None:
            excluded_years = ()

        self.excluded_years = set(excluded_years)

        self.files = []
        excluded_count = 0
        missing_pair_count = 0

        for spline_path in sorted(
            self.spline_dir.rglob("*.npz")
        ):
            relative_path = spline_path.relative_to(
                self.spline_dir
            )

            try:
                year = int(
                    spline_path.stem.rsplit("_", 1)[-1]
                )
            except ValueError as exc:
                raise ValueError(
                    f"Could not extract year from filename: "
                    f"{spline_path.name}"
                ) from exc

            if year in self.excluded_years:
                excluded_count += 1
                continue

            s1_path = self.s1_dir / relative_path
            dsm_path = self.dsm_dir / relative_path

            if not s1_path.exists() or not dsm_path.exists():
                missing_pair_count += 1
                continue

            self.files.append(relative_path)

        if not self.files:
            raise RuntimeError(
                "No paired spline/S1/DSM NPZ chips found under "
                f"{self.root}"
            )

        print(
            f"Found {len(self.files)} paired "
            "spline/S1/DSM chips."
        )

        if self.excluded_years:
            print(
                f"Excluded {excluded_count} chips from years: "
                f"{sorted(self.excluded_years)}"
            )

        if missing_pair_count > 0:
            print(
                f"Skipped {missing_pair_count} spline chips because "
                "the matching S1 or DSM file was missing."
            )

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _load_npz_array(
        path: Path,
        key: str,
        dtype=np.float32,
    ) -> np.ndarray:
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise KeyError(
                    f"Key '{key}' not found in {path}. "
                    f"Available keys: {archive.files}"
                )

            array = archive[key].astype(
                dtype,
                copy=False,
            )

        return array

    def __getitem__(self, idx):
        relative_path = self.files[idx]

        spline_path = self.spline_dir / relative_path
        s1_path = self.s1_dir / relative_path
        dsm_path = self.dsm_dir / relative_path

        # -------------------------------------------------------------
        # Spline coefficients
        # -------------------------------------------------------------

        spline_array = self._load_npz_array(
            path=spline_path,
            key="data",
        )

        spline = torch.from_numpy(spline_array)

        if spline.ndim != 3:
            raise RuntimeError(
                f"Expected spline shape (C, H, W), but found "
                f"{tuple(spline.shape)} in {spline_path}"
            )

        spline_valid = torch.isfinite(spline)

        spline = torch.where(
            spline_valid,
            spline,
            torch.zeros_like(spline),
        )

        if self.spline_scale_factor is not None:
            spline = spline / float(
                self.spline_scale_factor
            )

        # -------------------------------------------------------------
        # Sentinel-1
        # -------------------------------------------------------------

        s1_array = self._load_npz_array(
            path=s1_path,
            key="data",
        )

        s1 = torch.from_numpy(s1_array)

        if s1.ndim != 3 or s1.shape[0] < 2:
            raise RuntimeError(
                f"Expected at least two S1 bands in {s1_path}, "
                f"but found shape {tuple(s1.shape)}."
            )

        s1_valid = torch.isfinite(s1)

        if self.s1_nodata is not None:
            s1_valid &= (
                s1 != float(self.s1_nodata)
            )

        s1 = torch.where(
            s1_valid,
            s1,
            torch.zeros_like(s1),
        )

        if self.s1_scale_factor is not None:
            s1 = s1 / float(
                self.s1_scale_factor
            )

        if self.add_s1_ratio:
            vh = s1[0:1]
            vv = s1[1:2]

            ratio_valid = (
                s1_valid[0:1]
                & s1_valid[1:2]
            )

            vh_minus_vv = vh - vv

            vh_minus_vv = torch.where(
                ratio_valid,
                vh_minus_vv,
                torch.zeros_like(vh_minus_vv),
            )

            s1 = torch.cat(
                [
                    s1,
                    vh_minus_vv,
                ],
                dim=0,
            )

        # -------------------------------------------------------------
        # DSM label and validity mask
        # -------------------------------------------------------------

        with np.load(
            dsm_path,
            allow_pickle=False,
        ) as archive:
            required_keys = {
                "label",
                "valid_mask",
            }

            missing_keys = (
                required_keys - set(archive.files)
            )

            if missing_keys:
                raise KeyError(
                    f"Missing keys {sorted(missing_keys)} "
                    f"in {dsm_path}. "
                    f"Available keys: {archive.files}"
                )

            label_array = archive[
                "label"
            ].astype(
                np.float32,
                copy=False,
            )

            valid_mask_array = archive[
                "valid_mask"
            ].astype(
                bool,
                copy=False,
            )

        label = torch.from_numpy(label_array)
        valid_mask = torch.from_numpy(
            valid_mask_array
        ).bool()

        if label.ndim == 2:
            label = label.unsqueeze(0)

        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(0)

        if label.shape != valid_mask.shape:
            raise RuntimeError(
                f"DSM label and mask shapes differ in {dsm_path}: "
                f"{tuple(label.shape)} versus "
                f"{tuple(valid_mask.shape)}"
            )

        valid_mask &= torch.isfinite(label)

        # Clamp negative canopy heights to 0 m
        label = torch.where(
            valid_mask,
            label.clamp_min(0.0),
            torch.zeros_like(label),
        )

        sample = {
            "s2": spline,
            "spline": spline,
            "s1": s1,
            "label": label,
            "label_valid_mask": valid_mask,
            "filename": relative_path.name,
            "relative_path": str(relative_path),
            "tile_id": relative_path.parent.name,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample