# -----------------------
# Training Settings
# -----------------------

TRAIN_ROOT = "/data/ahsoka/eocp/wengler/height_database/spline/train"
VAL_ROOT   = "/data/ahsoka/eocp/wengler/height_database/spline/val"

BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_BANDS = 220
S1_BANDS = 3

DEVICE = "cuda"

NUM_WORKERS = 4
NUM_THREADS = 28
AUGMENT = True

MODEL_OUT = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/output/Spline/TEST.pth"
LOG_PATH = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/log/TEST.log"
TB_LOG_DIR = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/runs/spline3_S110mreal_10042026"

# -----------------------
# Prediction Settings
# -----------------------
PREDICTION_SPLINE_ROOT = "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/ThermSpline_DC"
PREDICTION_S1_ROOT = "/data/ahsoka/eocp/wengler/height_database/composite/S1/DE"
PREDICTION_OUTPUT_ROOT = "/data/ahsoka/eocp/wengler/height_database/ger_height_out/RLP_model"

PREDICTION_TILE_LIST_FILE = "/data/ahsoka/eocp/wengler/height_database/ger_height_out/Tile_allow_ger.txt"
PREDICTION_YEARS = [2023]

PREDICTION_PATCH_SIZE = 256
PREDICTION_BATCH_SIZE = 8
PREDICTION_MODEL = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/output/Spline/spline3_S110mreal_10042026_224.pth"
PREDICTION_TILE_SIZE = 1024