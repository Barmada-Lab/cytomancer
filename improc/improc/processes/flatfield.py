import numpy as np

from multiprocessing import Pool
from numpy.typing import NDArray
from typing import Hashable
from skimage.morphology import disk, white_tophat

from improc.experiment import Image
from improc.experiment.types import Dataset, Experiment, Exposure, MemoryImage, Mosaic, Vertex
from improc.processes.types import OneToOneTask, Task
from improc.utils import agg

from tqdm import tqdm

def apply_shading_correction(images: NDArray[np.float64]) -> NDArray[np.float64]:
    assert(len(images.shape) == 3)
    from pybasic import shading_correction
    basic = shading_correction.BaSiC(images)
    basic.prepare()
    basic.run()
    transformed = np.apply_over_axes(basic.normalize, images, axes=0) # type: ignore
    return transformed

class BaSiC(Task):

    def __init__(self, overwrite=False) -> None:
        super().__init__("basic_corrected")
        self.overwrite = overwrite

    def group_pred(self, image: Image) -> Hashable:
        vertex = image.get_tag(Vertex)
        mosaic_pos = image.get_tag(Mosaic)
        channel = image.get_tag(Exposure)
        return (vertex, mosaic_pos, channel)

    def correct(self, ims: list[Image]):
        arr = np.array([im.data for im in ims]).astype(np.float64)
        return (ims, apply_shading_correction(arr))

    def process(self, dataset: Dataset, experiment: Experiment) -> Dataset:
        output = experiment.new_dataset(self.output_label, overwrite=self.overwrite)
        groups = list(agg.groupby(dataset.images, self.group_pred).values())

        with Pool() as p:
            for group, corrected in tqdm(p.imap(self.correct, groups), total=len(groups), desc=self.__class__.__name__):
                for orig, corrected_slice in zip(group, corrected):
                    tags = orig.tags
                    axes = orig.axes
                    output.write_image(corrected_slice, tags, axes)
            return output

class RollingBall(OneToOneTask):

    def __init__(self, radius: int=13) -> None:
        super().__init__("rollingball")
        self.radius = radius

    def transform(self, image: Image) -> Image:
        se = disk(self.radius)
        transformed: np.ndarray = white_tophat(image.data, footprint=se) # type: ignore
        return MemoryImage(transformed, image.axes, image.tags)
