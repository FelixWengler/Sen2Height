# -----------------------
# Training Settings
# -----------------------

# Folder-based train/val inputs (expected subfolders: S2/ S1/ and BDOM/)
TRAIN_ROOT = "/data/ahsoka/eocp/wengler/height_database/train"
VAL_ROOT   = "/data/ahsoka/eocp/wengler/height_database/val"

BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_BANDS = 10  # number of Sentinel-2 bands in your chips
S1_BANDS = 3

DEVICE = "cuda"  # or "cpu"

# Optional performance knobs
NUM_WORKERS = 4     # start with 0 or small number; increase if stable
NUM_THREADS = 28       # CPU threads for torch
AUGMENT = True         # if your dataset supports augment=True/False

# Output paths
MODEL_OUT = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/output/S1S2/S1S2_10mS1_28032026_2.pth"
LOG_PATH = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/log/S1S2_10mS1_28032026_2log"


# -----------------------
# Prediction Settings
# -----------------------
PREDICTION_INPUT = "/data/ahsoka/eocp/wengler/height_database/composite/median/0608/25832/2023_0608_median_25832.tif"
PREDICTION_INPUT_S1_ALIGNED = "/data/ahsoka/eocp/wengler/height_database/composite/S1/median_25832/res/2023_res.tif"
PREDICTION_OUTPUT = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/predictions/S1_10m/2023_S1_10m_28032026"
PREDICTION_PATCH_SIZE = 256
PREDICTION_MODEL = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/output/S1S2/S1S2_10mS1_28032026_293.pth"
PREDICTION_WORKERS = 30
PREDICTION_BATCH_SIZE = 8
