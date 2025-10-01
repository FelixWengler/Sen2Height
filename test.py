# pip install rasterio numpy matplotlib scipy scikit-learn
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

pred_path  = r"D:/MSc. Arbeit/Sen2Height/predictions/full_s2/X0056_Y0050/X0056_Y0050/20240730_SEN2B_HEIGHT.tif"
lidar_path = r"D:/MSc. Arbeit/BDoms/Lidar/Austausch_MSc_UT_FelixWengler/Naturschutzareal_240925_lidar_10m_val_align_nodata.tif"

# ---------- read LiDAR on its native grid ----------
with rasterio.open(lidar_path) as ref_src:
    ref = ref_src.read(1).astype(np.float64)
    # Use mask band (0 = nodata, 255 = valid)
    ref_maskband = ref_src.read_masks(1)  # uint8
    ref_mask = ref_maskband > 0

    ref_scale  = (ref_src.scales[0] if ref_src.scales else 1.0)
    ref_offset = (ref_src.offsets[0] if ref_src.offsets else 0.0)
    ref = ref * ref_scale + ref_offset

    crs, transform = ref_src.crs, ref_src.transform
    width, height  = ref_src.width, ref_src.height

# ---------- read prediction resampled to LiDAR grid (no nodata bleed) ----------
with rasterio.open(pred_path) as src:
    src_nd = src.nodata  # may be 0, -9999, etc.
    with WarpedVRT(
        src,
        crs=crs, transform=transform, width=width, height=height,
        resampling=Resampling.bilinear,
        src_nodata=src_nd,    # tell VRT what source nodata is
        nodata=np.nan         # ensure destination nodata is NaN (won't influence stats)
    ) as vrt:
        pred = vrt.read(1).astype(np.float64)
        pred_scale  = (vrt.scales[0] if vrt.scales else 1.0)
        pred_offset = (vrt.offsets[0] if vrt.offsets else 0.0)
        pred = pred * pred_scale + pred_offset

# Build masks (don’t exclude zero; zero can be a real CHM value)
pred_mask = np.isfinite(pred)
valid = ref_mask & pred_mask & np.isfinite(ref)

# Extract pairs
x = ref[valid]   # LiDAR
y = pred[valid]  # Prediction

# ---------- sanity prints ----------
def summarize(name, v):
    q = np.percentile(v, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(f"{name}: N={v.size:,}  min={q[0]:.3f}  p1={q[1]:.3f}  p5={q[2]:.3f}  "
          f"median={q[4]:.3f}  p95={q[6]:.3f}  p99={q[7]:.3f}  max={q[8]:.3f}")
    print(f"  <=0.1m: {100*np.mean(v<=0.1):.1f}%  <=0.5m: {100*np.mean(v<=0.5):.1f}%")

print("---- Value ranges among VALID pairs ----")
summarize("LiDAR (x)", x)
summarize("Pred  (y)", y)

# quick linear fit for the legend
if x.size > 1:
    slope, intercept, r, p, se = stats.linregress(x, y)
else:
    slope = intercept = r = np.nan

# ---------- hexbin plot (log scale) ----------
fig, ax = plt.subplots(figsize=(7,6))
hb = ax.hexbin(x, y, gridsize=55, bins='log')  # log-scaled counts
cbar = fig.colorbar(hb, ax=ax)
cbar.set_label("count (log scale)")

mn = float(min(x.min(), y.min()))
mx = float(max(x.max(), y.max()))
ax.plot([mn, mx], [mn, mx], ls="--", lw=1.5, color="white", label="1:1")
if np.isfinite(slope):
    ax.plot([mn, mx], [slope*mn + intercept, slope*mx + intercept],
            lw=1.5, color="gold", label=f"Fit: y={slope:.3f}x+{intercept:.3f}")

ax.set_xlabel("LiDAR height (m) — reference")
ax.set_ylabel("Prediction height (m)")
ax.set_title("Prediction vs LiDAR (hexbin)")
ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
ax.legend(loc="upper left")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()
