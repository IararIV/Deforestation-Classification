from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset



class DeforestDataset(Dataset):
    """This dataset provides deforestation scenes of aerial images."""

    def __init__(self, root_data_dir: Union[str, Path], metadata_df: pd.DataFrame, transforms=None):
        self.root_data_dir = Path(root_data_dir)
        self.metadata_df = metadata_df
        self.transforms = transforms

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        image_metadata = self.metadata_df.iloc[idx]
        image_file = self.root_data_dir / image_metadata["images_path"]
        image = torch.tensor(np.array(Image.open(image_file)), dtype=torch.float)
        image = (image - image.min()) / (image.max() - image.min())
        image = image.permute((2, 0, 1))

        target = torch.tensor(image_metadata["label"], dtype=torch.long)  # Classes [0, 1, 2]
        metadata = torch.tensor(image_metadata[["year","latitude","longitude"]].tolist(), dtype=torch.float)
        metadata[0] -= 2001
        metadata[1] -= -0.4
        metadata[2] -= 104
        sample = {"image": image, "target": target, "metadata": metadata}

        if self.transforms:
            sample["image"] = self.transforms(sample["image"])

        return sample
