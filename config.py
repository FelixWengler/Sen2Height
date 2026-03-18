# -----------------------
# Training Settings
# -----------------------

# Folder-based train/val inputs (expected subfolders: S2/ S1/ and BDOM/)
TRAIN_ROOT = "/data/ahsoka/eocp/wengler/height_database/train"
VAL_ROOT   = "/data/ahsoka/eocp/wengler/height_database/val"

BATCH_SIZE = 4
EPOCHS = 70
LEARNING_RATE = 1e-4
NUM_BANDS = 10  # number of Sentinel-2 bands in your chips
S1_BANDS = 3

DEVICE = "cuda"  # or "cpu"

# Optional performance knobs
NUM_WORKERS = 4     # start with 0 or small number; increase if stable
NUM_THREADS = 28       # CPU threads for torch
AUGMENT = True         # if your dataset supports augment=True/False

# Output paths
MODEL_OUT = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/output/TEST_DELETE.pth"
LOG_PATH = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/log/TEST_DELETE.log"


# -----------------------
# Prediction Settings
# -----------------------
PREDICTION_INPUT = "/data/ahsoka/eocp/wengler/height_database/composite/median/0608/25832/2025_0608_median_25832.tif"
PREDICTION_INPUT_S1_ALIGNED = "/data/ahsoka/eocp/wengler/height_database/S1/res/S1_2025_VH_VV_stack_clip_res.tif"
PREDICTION_OUTPUT = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/predictions/2025_21022026_firsttry.tif"
PREDICTION_PATCH_SIZE = 256
PREDICTION_MODEL = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/output/S1_firsttest_20022026_2.82m.pth"
PREDICTION_WORKERS = 30
PREDICTION_BATCH_SIZE = 8
