import pdal
import json
import os
from glob import glob
from osgeo import gdal

input_folder = "D:/MSc. Arbeit/BDoms/laz/Hermeskeil"
output_folder = "D:/MSc. Arbeit/BDoms/tifs/unprocessed"
mosaic_path = "D:/MSc. Arbeit/BDoms/tifs/bdom_mosaic.tif"
resampled_path = "D:/MSc. Arbeit/BDoms/tifs/bdom_mosaic_10m.tif"

laz_files = glob(os.path.join(input_folder, "*.laz"))
tif_paths=[]

#read laz and store max z-value per cell

for laz_file in laz_files:
    tif_file = os.path.splitext(os.path.basename(laz_file))[0] + ".tif"
    tif_path = os.path.join(output_folder, tif_file)
    tif_paths.append(tif_path)

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


#gdal resample and mosaic

vrt_path = os.path.join(output_folder, "mosaic.vrt")
gdal.BuildVRT(vrt_path, tif_paths)

gdal.Translate(mosaic_path, vrt_path)

gdal.Warp(resampled_path, mosaic_path, xRes=10, yRes=10, resampleAlg='bilinear')

print(f"Final resampled DSM saved to: {resampled_path}")