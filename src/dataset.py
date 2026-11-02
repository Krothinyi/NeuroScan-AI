# Build the PyTorch Dataset and DataLoader
# We'll now convert our MRI image files into tensors 
# that ResNet18 can process.


# ==========================================
# CUSTOM BRAIN MRI DATASET
# ==========================================

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

CLASS_TO_IDX = {
    name: idx for idx, name in enumerate(CLASSES)
}


class BrainMRIDataset(Dataset):

    def __init__(self, manifest_path, transform=None):

        self.df = pd.read_csv(manifest_path)
        self.transform = transform

        # Validate labels before loading images
        unknown = set(self.df["label"]) - set(CLASSES)

        if unknown:
            raise ValueError(
                f"Unknown classes: {unknown}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = Path(row["filepath"])
        label = CLASS_TO_IDX[row["label"]]

        # Read the MRI image
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # Apply resizing, normalization, etc.
        if self.transform is not None:
            image = self.transform(image)

        return image, label
