import logging
from pathlib import Path

from tqdm import tqdm
from PIL import Image
from distributed import as_completed, get_client
from skimage import exposure, restoration  # type: ignore
import dask
import fiftyone as fo
import numpy as np

logger = logging.getLogger(__name__)


def zhuzh(dataset_name: str, apply_wavelet_denoising: bool = True, adapteq_clip_limit: float | None = 0.01):
    """
    Zhushes the dataset by applying adaptive histogram equalization and wavelet denoising
    to the view images (doesn't touch raw images)
    """

    client = get_client()

    if not fo.dataset_exists(dataset_name):
        raise ValueError("Dataset does not exist")

    dataset = fo.load_dataset(dataset_name)

    @dask.delayed
    def zhush_image(sample_filepath: Path):
        arr = np.array(Image.open(sample_filepath))
        if adapteq_clip_limit is not None:
            arr = exposure.equalize_adapthist(arr, clip_limit=adapteq_clip_limit)
        if apply_wavelet_denoising:
            arr = restoration.denoise_wavelet(arr)
        rescaled = exposure.rescale_intensity(arr, out_range="uint8")
        image = Image.fromarray(rescaled)
        image.save(sample_filepath, format="PNG")

    paths = [zhush_image(sample.filepath) for sample in dataset]
    for _ in tqdm(as_completed(client.compute(paths)), total=len(paths)):
        continue
