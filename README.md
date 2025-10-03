# Sen2Height
This repo contains code for training and applying a Residual U-net model for height predictions using digital surface 
models (DSMs) and Sentinel-2 imagery using PyTorch and TorchGeo.


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