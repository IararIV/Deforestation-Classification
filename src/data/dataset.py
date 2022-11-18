from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ShipDataset(Dataset):
    """This dataset provides maritime scenes of optical aerial images from visible spectrum."""

    def __init__(self, root_data_dir: Union[str, Path], metadata_df: pd.DataFrame, transforms=None):
        self.root_data_dir = Path(root_data_dir)
        self.metadata_df = metadata_df
        self.transforms = transforms

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        image_metadata = self.metadata_df.iloc[idx]
        image_file = self.root_data_dir / "images" / image_metadata["image"]
        image = torch.tensor(np.array(Image.open(image_file).convert("RGB")), dtype=torch.float)  # Transform to RGB since we have grayscale images
        image = image.permute((2, 0, 1))

        target = torch.tensor(image_metadata["category"] - 1, dtype=torch.long)  # Classes [1, 5] ==> [0, 4]
        sample = {"image": image, "target": target}

        if self.transforms:
            sample["image"] = self.transforms(sample["image"])

        return sample
