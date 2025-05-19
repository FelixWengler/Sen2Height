import torch
from torchgeo.datasets.utils import BoundingBox
from torchgeo.datasets import RasterDataset, GeoDataset, IntersectionDataset

class SentinelImage(RasterDataset):
    filename_glob = '*.tif'  # adjust based on Sentinel naming (e.g., 'T32*.tif')
    is_image = True

    def __getitem__(self, query):
        sample = super().__getitem__(query)

        # Normalize Sentinel reflectance to 0–1
        image = sample["image"]  # [C, H, W]
        image = torch.clamp((image-1000) / 10000.0, 0, 1)
        sample["image"] = image

        return sample

class DSMImage(RasterDataset):
    filename_glob = '*.tif'
    is_image = True


class SentinelDSMCombo(GeoDataset):
    def __init__(self, sentinel_dataset, dsm_dataset, transforms=None):
        super().__init__()  # ensures proper GeoDataset setup
        self.sentinel_dataset = sentinel_dataset
        self.dsm_dataset = dsm_dataset
        self.transforms = transforms

        # Create an intersection to define a valid spatial index
        self.intersection = IntersectionDataset(sentinel_dataset, dsm_dataset)

        # Set the internal spatial index
        self._index = self.intersection.index

    def __len__(self):
        return len(self.intersection)

    def __getitem__(self, query):
        if not isinstance(query, BoundingBox):
            import traceback
            traceback.print_stack()

        # Add dummy temporal info if not present (some samplers only return 4D)
        if query.mint is None or query.maxt is None:
            query = BoundingBox(
                minx=query.minx,
                maxx=query.maxx,
                miny=query.miny,
                maxy=query.maxy,
                mint=0,
                maxt=1,
            )

        # Load Sentinel-2 and DSM data for the given bounding box
        sample_sentinel = self.sentinel_dataset[query]
        sample_dsm = self.dsm_dataset[query]

        # Image is Sentinel, Label is DSM
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
# Example usage:
# sentinel = SentinelImage(root='/path/to/sentinel')
# dsm = DSMImage(root='/path/to/dsm')
# dataset = SentinelDSMCombo(sentinel, dsm)
# sample = dataset[dataset.index[0]]
# print(sample["image"].shape)  # Should be [Sentinel bands + 1, H, W]
