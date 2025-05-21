import config
import rasterio

print("✅ Checking DSM...")
with rasterio.open(config.DSM_DIR) as src:
    print("DSM bounds:", src.bounds)
    print("DSM CRS:", src.crs)

print("\n✅ Checking Sentinel...")
with rasterio.open(config.SENTINEL_DIR) as src:
    print("Sentinel bounds:", src.bounds)
    print("Sentinel CRS:", src.crs)
