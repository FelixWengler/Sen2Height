import torch
from torchgeo.datasets.utils import BoundingBox
from torchgeo.datasets import RasterDataset, GeoDataset, IntersectionDataset
import rasterio
import numpy as np


# S2 data wrapper
class SentinelImage(RasterDataset):
    filename_glob = '*.tif'  # adjust based on Sentinel naming (e.g., 'T32*.tif')
    is_image = True

    def __getitem__(self, query):
        sample = super().__getitem__(query)

        # Normalize Sentinel reflectance to 0–1
        image = sample["image"]  # [C, H, W]
        image = torch.clamp(image / 10000.0, 0, 1)  # remove offset of 1000 if working with regular sentinel data
        sample["image"] = image

        return sample

#class DSMImage(RasterDataset):
    #filename_glob = '*.tif'
    #is_image = True

# DSM data wrapper
class DSMImage(RasterDataset):
    filename_glob = '*.tif'
    is_image = True

    def __init__(self, root):
        super().__init__(root)
        import rasterio
        with rasterio.open(self.files[0]) as src:
            self.full_dsm = src.read(1)
            self.transform = src.transform
            self.height = src.height
            self.width = src.width

    # Extract DSM patch corresponding to boundingbox
    def __getitem__(self, query: BoundingBox):

        col_start, row_start = ~self.transform * (query.minx, query.maxy)
        col_end, row_end = ~self.transform * (query.maxx, query.miny)

        row_start = max(0, min(self.height, int(np.floor(row_start))))
        row_end = max(0, min(self.height, int(np.ceil(row_end))))
        col_start = max(0, min(self.width, int(np.floor(col_start))))
        col_end = max(0, min(self.width, int(np.ceil(col_end))))

        # Avoid empty patch
        if row_end <= row_start or col_end <= col_start:
            patch = np.zeros((64, 64), dtype=np.float32)
        else:
            patch = self.full_dsm[row_start:row_end, col_start:col_end]

        patch_tensor = torch.from_numpy(patch).unsqueeze(0)
        return {"image": patch_tensor}


# Dataset pairs S2 with DSM data
class SentinelDSMCombo(GeoDataset):
    def __init__(self, sentinel_dataset, dsm_dataset, transforms=None):
        super().__init__()
        self.sentinel_dataset = sentinel_dataset
        self.dsm_dataset = dsm_dataset
        self.transforms = transforms

        # Create intersection
        self.intersection = IntersectionDataset(sentinel_dataset, dsm_dataset)
        self._index = self.intersection.index

    def __len__(self):
        return len(self.intersection)

    def __getitem__(self, query):
        if not isinstance(query, BoundingBox):
            import traceback
            traceback.print_stack()

        # Add temporal info if not present (some samplers only return 4D)
        if query.mint is None or query.maxt is None:
            query = BoundingBox(
                minx=query.minx,
                maxx=query.maxx,
                miny=query.miny,
                maxy=query.maxy,
                mint=0,
                maxt=1,
            )

        # Load S2 and DSM data for the given bounding box and build sample
        sample_sentinel = self.sentinel_dataset[query]
        sample_dsm = self.dsm_dataset[query]

        image = sample_sentinel["image"]
        label = sample_dsm["image"]

        sample = {
            "image": image,
            "label": label,
            "bounds": query,
            "crs": sample_sentinel.get("crs", None),
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    # GeoDataset interface
    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def bounds(self):
        return self._index.bounds

    @property
    def res(self):
        return self.sentinel_dataset.res