# sentinel2_dsm_regression/config.py

#--Training Settings--#

SENTINEL_DIR = "data/sentinel/train/10m_layerstack_clip2.tif"
DSM_DIR = "data/dsm/train_dsm/bdom_fin2.tif"
PATCH_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-3
NUM_BANDS = 4  # Number of input bands from Sentinel-2

DEVICE = "cpu"  # or "cpu"

#--Prediction settings--#

PREDICTION_INPUT = "data/sentinel/class/10m_layerstack_comp.tif"
PREDICTION_OUTPUT = "predictions/full_s2/output_30052025.tif"
PREDICTION_PATCH_SIZE = 64
