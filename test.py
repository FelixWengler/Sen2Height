import rasterio
import numpy as np


input_path = "D:/MSc. Arbeit/Sen2Height/data/dsm/test1.tif"
output_path = "D:/MSc. Arbeit/Sen2Height/data/dsm/cleaned_dsm.tif"

NODATA_VALUE = -9999.0  # ✅ Use something safely outside valid range

with rasterio.open(input_path) as src:
    profile = src.profile
    dsm = src.read(1)

    # Identify extreme invalid values
    dsm[(dsm == src.nodata) | (dsm < -1e10)] = NODATA_VALUE

    # Update profile to preserve nodata explicitly
    profile.update(
        dtype=rasterio.float32,
        nodata=NODATA_VALUE
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dsm.astype(np.float32), 1)

print(f"✅ Cleaned DSM saved with proper NoData to: {output_path}")

#transform for data
#more training data CHECK!
#try torchgeo trainer and model
#add validation