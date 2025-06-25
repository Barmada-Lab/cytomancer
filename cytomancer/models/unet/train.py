# This file is part of Cytomancer.
#
# Cytomancer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Cytomancer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MyProject. If not, see <http://www.gnu.org/licenses/>.
#
# Portions of this code are based on the original project Pytorch-UNet,
# developed by Milesi Alexandre. Pytorch-UNet is available at:
# <https://github.com/milesial/Pytorch-UNet>
#
# Some modifications made by Jacob Waksmacki, 2025
import logging
from pathlib import Path

import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from cytomancer.io.datasets import MaskSegDataset

from .dice import dice_coeff, dice_loss
from .unet import UNet


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch in tqdm(
            dataloader,
            total=num_val_batches,
            desc="Validation round",
            unit="batch",
            leave=False,
        ):
            image, mask_true = batch["image"], batch["mask"]

            # move images and labels to correct device and type
            image = image.to(
                device=device, dtype=torch.float32, memory_format=torch.channels_last
            )
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if not (mask_true.min() >= 0 and mask_true.max() <= 1):
                raise ValueError("True mask indices should be in [0, 1]")
            mask_pred = (F.sigmoid(mask_pred) > 0.5).long().squeeze(1)
            # compute the Dice score
            dice_score += dice_coeff(mask_pred, mask_true)

    net.train()
    return dice_score / max(num_val_batches, 1)


def train_model(
    model,
    dataset_path,
    checkpoint_path,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    save_checkpoint: bool = True,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
    dataloader_workers: int = 4,
):
    train_transforms = A.Compose(
        [
            A.RandomBrightnessContrast(
                p=0.5, brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3)
            ),
            A.RandomGamma(p=0.5, gamma_limit=(70, 130)),
            A.CLAHE(p=1, clip_limit=(10, 10)),
            A.Normalize(normalization="standard", mean=0, std=1),
            A.RandomCrop(height=256, width=256, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ToTensorV2(),
        ]
    )

    train_set = MaskSegDataset(
        dataset_path, image_set="train", transforms=train_transforms, use_cache=True
    )

    val_transforms = A.Compose(
        [
            A.CLAHE(p=1, clip_limit=(10, 10)),
            A.Normalize(normalization="standard", mean=0, std=1),
            A.Crop(x_max=512, y_max=512),
            A.ToTensorV2(p=1),
        ]
    )
    val_set = MaskSegDataset(
        dataset_path, image_set="val", transforms=val_transforms, use_cache=True
    )

    # 3. Create data loaders
    train_loader: DataLoader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=batch_size,
        num_workers=dataloader_workers,
        pin_memory=True,
    )
    val_loader: DataLoader = DataLoader(
        val_set,
        shuffle=False,
        drop_last=True,
        batch_size=batch_size,
        num_workers=dataloader_workers,
        pin_memory=True,
    )
    # (Initialize logging)
    experiment = wandb.init(project="U-Net", resume="allow", anonymous="allow")
    experiment.config.update(
        {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "save_checkpoint": save_checkpoint,
            "amp": amp,
        }
    )

    n_train = len(train_set)
    n_val = len(val_set)

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    """
    )

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, eta_min=1e-9
    )
    grad_scaler = torch.amp.GradScaler(enabled=amp)
    criterion = nn.BCEWithLogitsLoss()
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for i, batch in enumerate(train_loader):
                images, true_masks = batch["image"], batch["mask"]

                if not images.shape[1] == model.n_channels:
                    raise ValueError(
                        f"Network has been defined with {model.n_channels} input channels, "
                        f"but loaded images have {images.shape[1]} channels. Please check that "
                        "the images are loaded correctly."
                    )

                images = images.to(
                    device=device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(
                    device.type if device.type != "mps" else "cpu", enabled=amp
                ):
                    masks_pred = model(images)
                    loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    loss += dice_loss(
                        F.sigmoid(masks_pred.squeeze(1)), true_masks.float()
                    )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step(epoch + i / len(train_loader))

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log(
                    {"train loss": loss.item(), "step": global_step, "epoch": epoch}
                )
                pbar.set_postfix(**{"loss (batch)": loss.item()})

                # Evaluation round
                division_step = n_train // (5 * batch_size)
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace("/", ".")
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms["Weights/" + tag] = wandb.Histogram(
                                    value.data.cpu()
                                )
                            if not (
                                torch.isinf(value.grad) | torch.isnan(value.grad)
                            ).any():
                                histograms["Gradients/" + tag] = wandb.Histogram(
                                    value.grad.data.cpu()
                                )

                        val_score = evaluate(model, val_loader, device, amp)

                        logging.info(f"Validation Dice score: {val_score}")
                        experiment.log(
                            {
                                "learning rate": optimizer.param_groups[0]["lr"],
                                "validation Dice": val_score,
                                "images": wandb.Image(images[0].cpu()),
                                "masks": {
                                    "true": wandb.Image(true_masks[0].float().cpu()),
                                    "pred": wandb.Image(
                                        masks_pred.argmax(dim=1)[0].float().cpu()
                                    ),
                                },
                                "step": global_step,
                                "epoch": epoch,
                                **histograms,
                            }
                        )

        if save_checkpoint:
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict["mask_values"] = train_set.mask_values
            torch.save(
                state_dict, str(checkpoint_path / f"checkpoint_epoch{epoch}.pth")
            )
            logging.info(f"Checkpoint {epoch} saved!")


def run(
    dataset_path: Path,
    checkpoint_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    momentum: float,
    amp: bool,
    bilinear: bool,
    n_classes: int,
    load: str | None,
):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    model = UNet(n_channels=1, n_classes=n_classes, bilinear=bilinear)

    logging.info(
        f"Network:\n"
        f"\t{model.n_channels} input channels\n"
        f"\t{model.n_classes} output channels (classes)\n"
        f"\t{'Bilinear' if model.bilinear else 'Transposed conv'} upscaling"
    )

    if load:
        state_dict = torch.load(load, map_location=device)
        del state_dict["mask_values"]
        model.load_state_dict(state_dict)
        logging.info(f"Model loaded from {load}")

    model.to(device=device)
    try:
        train_model(
            model=model,
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            amp=amp,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error(
            "Detected OutOfMemoryError! "
            "Enabling checkpointing to reduce memory usage, but this slows down training. "
            "Consider enabling AMP (--amp) for fast and memory efficient training"
        )
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            dataset_path=dataset_path,
            checkpoint_path=checkpoint_path,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            amp=amp,
        )
