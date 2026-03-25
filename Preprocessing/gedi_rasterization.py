from pathlib import Path
from collections import defaultdict

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import box
from shapely.geometry import Point
from tqdm import tqdm


# ======================
# CONFIG
# ======================

YEAR = 2024
GEDI_FILE = "/data/ahsoka/eocp/wengler/height_database/GEDI/RLP/2024/processed/2024_gedi_rh95_rlp_merged.parquet"
TILE_DIR = "/data/ahsoka/eocp/wengler/height_database/GEDI/chips/S2"
OUT_LABEL_DIR = "/data/ahsoka/eocp/wengler/height_database/GEDI/chips/GEDI/label"
OUT_MASK_DIR = "/data/ahsoka/eocp/wengler/height_database/GEDI/chips/GEDI/mask"

GEDI_VALUE_COLUMN = "rh95"


AGGREGATION = "mean"    # mean / max / first
LABEL_NODATA = -9999.0
YEAR_COLUMN = None

# ======================
# FUNCTIONS
# ======================

def aggregate_values(values):
    if len(values) == 0:
        return None
    if AGGREGATION == "mean":
        return float(np.mean(values))
    if AGGREGATION == "max":
        return float(np.max(values))
    if AGGREGATION == "first":
        return float(values[0])
    raise ValueError("Unknown aggregation")


def rasterize_tile(tile_path, gedi):

    with rasterio.open(tile_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        height = src.height
        width = src.width
        tile_poly = box(*src.bounds)

    label = np.full((height, width), LABEL_NODATA, dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)

    tile_points = gedi[gedi.geometry.intersects(tile_poly)]

    if len(tile_points) == 0:
        return label, mask, profile

    pixel_values = defaultdict(list)

    for _, row in tile_points.iterrows():
        x, y = row.geometry.x, row.geometry.y
        r, c = rowcol(transform, x, y)

        if 0 <= r < height and 0 <= c < width:
            v = row[GEDI_VALUE_COLUMN]
            if v is None or np.isnan(v):
                continue
            pixel_values[(r, c)].append(float(v))

    for (r, c), vals in pixel_values.items():
        agg = aggregate_values(vals)
        if agg is not None:
            label[r, c] = agg
            mask[r, c] = 1

    return label, mask, profile


# ======================
# MAIN
# ======================

def main():
  
    df = pd.read_parquet(GEDI_FILE)

    print(df.columns)   # IMPORTANT debug

    gedi = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon_high"], df["lat_high"]),
        crs="EPSG:4326"
    )

    if YEAR is not None and YEAR_COLUMN is not None:
        gedi = gedi[gedi[YEAR_COLUMN] == YEAR]

    all_tile_paths = sorted(Path(TILE_DIR).glob("*.tif"))

    if YEAR is not None:
        tile_paths = [p for p in all_tile_paths if p.stem.endswith(f"_{YEAR}")]
    else:
        tile_paths = all_tile_paths

    if len(tile_paths) == 0:
        raise RuntimeError(f"No tile files found for year {YEAR} in {TILE_DIR}")

    with rasterio.open(tile_paths[0]) as src:
        tile_crs = src.crs

    if gedi.crs != tile_crs:
        gedi = gedi.to_crs(tile_crs)

    total = 0

    for tile_path in tqdm(tile_paths):

        label, mask, profile = rasterize_tile(tile_path, gedi)

        profile_label = profile.copy()
        profile_label.update(count=1, dtype=rasterio.float32, nodata=LABEL_NODATA)

        profile_mask = profile.copy()
        profile_mask.update(count=1, dtype=rasterio.uint8, nodata=0)

        out_label = Path(OUT_LABEL_DIR) / tile_path.name
        out_mask = Path(OUT_MASK_DIR) / tile_path.name

        out_label.parent.mkdir(parents=True, exist_ok=True)
        out_mask.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(out_label, "w", **profile_label) as dst:
            dst.write(label, 1)

        with rasterio.open(out_mask, "w", **profile_mask) as dst:
            dst.write(mask, 1)

        total += mask.sum()

    print("Total GEDI pixels rasterized:", int(total))


if __name__ == "__main__":
    main()