# -----------------------
# Training Settings
# -----------------------

# Expected subfolders: S2/ S1/ GEDI_LABEL/ GEDI_MASK/
TRAIN_ROOT = "/data/ahsoka/eocp/wengler/height_database/GEDI/chips/train"
VAL_ROOT   = "/data/ahsoka/eocp/wengler/height_database/GEDI/chips/val"

S2_SUBDIR = "S2"
S1_SUBDIR = "S1"
GEDI_LABEL_SUBDIR = "gedi_label"
GEDI_MASK_SUBDIR = "gedi_mask"

BATCH_SIZE = 4
EPOCHS = 70
LEARNING_RATE = 1e-4

NUM_BANDS = 10      # use 220 if using spline coefficient stacks
S1_BANDS = 3

DEVICE = "cuda"

NUM_WORKERS = 4
NUM_THREADS = 28
AUGMENT = True

MODEL_OUT = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/output/GEDI/gedifirsttest_24032026.pth"
LOG_PATH = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/log/GEDI/gedifirsttest_24032026.log"


# -----------------------
# Prediction Settings
# -----------------------
PREDICTION_INPUT = "/data/ahsoka/eocp/wengler/height_database/composite/median/0608/25832/2021_0608_median_25832.tif"
PREDICTION_INPUT_S1_ALIGNED = "/data/ahsoka/eocp/wengler/height_database/S1/res/S1_2021_VH_VV_stack_clip_res.tif"
PREDICTION_OUTPUT = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/predictions/GEDI/2021_gedifirsttry_24032026.tif"
PREDICTION_PATCH_SIZE = 256
PREDICTION_MODEL = "/data/ahsoka/eocp/wengler/Sen2height_dualenc/model/output/GEDI/gedifirsttest_24032026.pth"
PREDICTION_WORKERS = 30
PREDICTION_BATCH_SIZE = 8