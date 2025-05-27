from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from skimage.exposure import rescale_intensity
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
            Must conclude with conversion to a CHW tensor.
    """

    def __init__(
        self,
        path: Path,
        image_set: Literal["train", "test", "val"] = "train",
        transforms: Callable | None = None,
        use_cache: bool = False,
    ):
        super().__init__(
            path,
            transforms=transforms,
        )

        self.image_set = image_set
        self.use_cache = use_cache
        self.cache: dict[tuple[Path, Path], tuple[np.ndarray, np.ndarray]] = {}

        df = pd.read_csv(
            self.root / "index.csv",
            converters={
                "image": lambda x: self.root / Path(x),
                "labels": lambda x: self.root / Path(x),
            },
        )
        self.index = df[df["split"] == image_set]
        self.mask_values = [0, 1]  # foreground / background

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        row = self.index.iloc[idx]

        key = (row["image"], row["labels"])
        if key in self.cache:
            image, target = self.cache[key]
        else:
            image = rescale_intensity(
                np.array(PILImage.open(row["image"])), out_range=np.float32
            )
            target = rescale_intensity(
                np.array(PILImage.open(row["labels"])), out_range=np.float32
            )
            if self.use_cache:
                self.cache[key] = (image, target)

        if self.transforms is not None:
            return self.transforms(image=image, mask=target)

        else:
            return Image(image), Image(target)
