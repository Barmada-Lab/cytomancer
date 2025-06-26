import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from skimage.exposure import rescale_intensity

from .unet import UNet


def predict(
    model: nn.Module, image: np.ndarray, threshold: float = 0.5, device: str = "cpu"
) -> np.ndarray:
    model.eval()
    rescaled = rescale_intensity(image, out_range=np.float32)
    transforms = A.Compose(
        [
            A.CLAHE(p=1, clip_limit=(10, 10)),
            A.Normalize(normalization="standard", mean=0, std=1),
            A.ToTensorV2(),
        ]
    )
    imgt = (
        transforms(image=rescaled)["image"]
        .unsqueeze(0)
        .to(device, dtype=torch.float32, memory_format=torch.channels_last)
    )
    with torch.inference_mode():
        output = model(imgt)
        output = torch.sigmoid(output)
        output = output > threshold
    return output.squeeze(0).cpu().numpy()


def load_unet(model_path, n_channels=1, n_classes=1, device="cpu"):
    net = UNet(n_channels=n_channels, n_classes=n_classes)
    state_dict = torch.load(model_path, map_location=device)
    state_dict.pop("mask_values")
    net.load_state_dict(state_dict)
    net = net.to(device)
    return net
