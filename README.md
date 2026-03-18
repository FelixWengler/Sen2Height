# Sen2Height
A Residual U-net model for height predictions using digital surface 
models (DSMs), Sentinel1 and Sentinel2 data.


## Repository Structure
```
├── Bdom_processing/
│ └── bdom_processing.py    #preprocessing of DSM tiles
├── datasets/
│ └── raster_datasets.py    #dataloader (Sentinel + DSM)
├── models/
│ └── height_net.py         #Residual U-net model
├── utils/
│ └── metrics.py            #RMSE and related metrics
├── config.py               #configuration (paths, hyperparameters)
├── predict.py              #single image prediction
├── train.py                #training (set log, workers and output)
```
