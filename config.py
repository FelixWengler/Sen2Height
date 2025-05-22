# sentinel2_dsm_regression/config.py

SENTINEL_DIR = "data/sentinel/10m_layerstack_clip.tif"
DSM_DIR = "data/dsm/bdom_mosaic_clipped_mult.tif"
PATCH_SIZE = 64
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-3
NUM_BANDS = 4  # Number of input bands from Sentinel-2

DEVICE = "cpu"  # or "cpu"

# Prediction settings
PREDICTION_OUTPUT = "predictions/output16052025.tif"
PREDICTION_PATCH_SIZE = 32
