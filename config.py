# -----------------------
# Training Settings
# -----------------------

TRAIN_ROOT = "/data/ahsoka/eocp/wengler/height_database/spline/train"
VAL_ROOT   = "/data/ahsoka/eocp/wengler/height_database/spline/val"

BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_BANDS = 220
S1_BANDS = 3

DEVICE = "cuda"

NUM_WORKERS = 4
NUM_THREADS = 28
AUGMENT = True

MODEL_OUT = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/output/Spline/splinetest2_s110m_02042026_2.pth"
LOG_PATH = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/log/splinetest2_s110m_02042026_2.log"

# -----------------------
# Prediction Settings
# -----------------------
PREDICTION_SPLINE_ROOT = "/data/ahsoka/eocp/forestpulse/01_data/02_processed_data/ThermSpline_DC"
PREDICTION_S1_ROOT = "/data/ahsoka/eocp/wengler/height_database/composite/S1/median_3035"
PREDICTION_OUTPUT = "/data/ahsoka/eocp/wengler/height_database/spline/predictions/splinetest2_s110m_02042026_2/2025_rlp_height.tif"
PREDICTION_OUTPUT_ROOT = "/data/ahsoka/eocp/wengler/height_database/spline/predictions/splinetest2_s110m_02042026_2"

PREDICTION_TILE_LIST_FILE = "/data/ahsoka/eocp/wengler/height_database/composite/rlp_tileallow/Tile_allow_spline.txt"
PREDICTION_YEARS = [2025]

PREDICTION_PATCH_SIZE = 256
PREDICTION_BATCH_SIZE = 8
PREDICTION_MODEL = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/output/Spline/splinetest2_s110m_02042026_2.pth"
PREDICTION_TILE_SIZE = 1024