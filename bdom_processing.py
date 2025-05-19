import pdal
import json
import os
from glob import glob

input_folder = "D:/MSc. Arbeit/BDoms/laz"
output_folder = "D:/MSc. Arbeit/BDoms/tifs"

laz_files = glob(os.path.join(input_folder, "*.laz"))

for laz_file in laz_files:
    tif_file = os.path.splitext(os.path.basename(laz_file))[0] + ".tif"
    tif_path = os.path.join(output_folder, tif_file)

    pipeline_json = [
        {
            "type": "readers.las",
            "filename": laz_file
        },
        {
            "type": "writers.gdal",
            "filename": tif_path,
            "resolution": 1.0,
            "data_type": "float",
            "output_type": "max"
        }
    ]

    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()
    print(f"Processed {laz_file} -> {tif_path}")