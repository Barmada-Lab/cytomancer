from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd
from PIL import Image as PILImage
from torchvision.datasets import VisionDataset
from torchvision.tv_tensors import Image


class MaskSegDataset(VisionDataset):
    """
    Dataset for mask-based segmentation.

    Args:
        path (Path): Path to the dataset. Should contain a `index.csv` file with
            the following columns: `image`, `labels`, `split`. `image` and `labels`
            are paths to the image and mask files, and `split` is one of `train`,
            `test`, or `val`.
        image_set (Literal["train", "test", "val"]): Image set to use.
        transforms (Callable | None): Transforms to apply to the image and target.
    """

    def __init__(
        self,
        path: Path,
        image_set: Literal["train", "test", "val"] = "train",
        transforms: Callable | None = None,
    ):
        super().__init__(
            path,
            transforms=transforms,
        )

        self.image_set = image_set

        df = pd.read_csv(
            self.root / "index.csv",
            converters={
                "image": lambda x: self.root / Path(x),
                "labels": lambda x: self.root / Path(x),
            },
        )
        self.index = df[df["split"] == image_set]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        row = self.index.iloc[idx]
        image = Image(PILImage.open(row["image"]))
        target = Image(PILImage.open(row["labels"]))

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
