# sentinel2_dsm_regression/config.py

#--Training Settings--#

SENTINEL_DIR = "data/sentinel/20230708_SEN2A_clip_25832.tif"
DSM_DIR = "data/dsm/dgm10_repaired.tif"
PATCH_SIZE = 64
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_BANDS = 10  # Number of input bands from Sentinel-2


DEVICE = "cpu"  # or "cuda"

#--Prediction settings--#

PREDICTION_INPUT = "/data/ahsoka/dc/deu/ard/X0056_Y0050/20240625_LEVEL2_SEN2A_BOA.tif"
PREDICTION_OUTPUT = "/data/ahsoka/student/s1feweng/Sen2Height_main/predictions/force_predict/X0056_Y0050_smooth/20240625_SEN2A_HEIGHT.tif"
PREDICTION_PATCH_SIZE = 32
PREDICTION_MODEL = "models/output/model_230708_56_50_2m_final.pth"
PREDICTION_WORKERS = 20  # Number of CPU cores to use
PREDICTION_BATCH_SIZE = 8  # Number of patches predicted at once

