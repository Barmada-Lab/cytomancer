import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage import (
    exposure,  # type: ignore
    filters,  # type: ignore
)
from skimage.exposure import equalize_adapthist
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.restoration import denoise_wavelet
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


SIGMAS = [0.7, 1, 3.5]

FOREGROUND = 1
BACKGROUND = 2


def preprocess_image(raw_image):
    rescaled_float = exposure.rescale_intensity(raw_image, out_range=float)
    eqd = equalize_adapthist(rescaled_float, clip_limit=0.05)
    denoised = denoise_wavelet(eqd)
    gaussian_stack = [
        (f"gaussian-{s}", filters.gaussian(denoised, sigma=s)) for s in SIGMAS
    ]
    log_stack = [
        (f"LoG-{s}", filters.laplace(g[1]))
        for s, g in zip(SIGMAS[1:], gaussian_stack[1:], strict=False)
    ]
    gg_mag_stack = [
        (f"sobel-{s}", filters.sobel(g[1]))
        for s, g in zip(SIGMAS[1:], gaussian_stack[1:], strict=False)
    ]
    hessian_stack = [
        hessian_matrix(
            raw_image, sigma=s, mode="nearest", use_gaussian_derivatives=True
        )
        for s in SIGMAS[1:]
    ]
    hessian_eig_stack = [
        (f"hess_eig-{s}", hessian_matrix_eigvals(hess)[0])
        for s, hess in zip(SIGMAS, hessian_stack, strict=False)
    ]

    feature_stack = gaussian_stack + log_stack + gg_mag_stack + hessian_eig_stack

    feature_labels = [f[0] for f in feature_stack]
    feature_data = np.stack([f[1] for f in feature_stack], axis=-1)
    return feature_labels, feature_data


def extract_ilastish_features(raw_image, mask_image):
    """
    Extracts pixel-level features and labels from raw and mask images
    featureset inspired by ilastik pixel classification
    """

    feature_labels, feature_data = preprocess_image(raw_image)
    nonzero_indices = np.nonzero(mask_image)
    labels = mask_image[nonzero_indices]
    labelled_features = feature_data[nonzero_indices]

    rows = []
    for label, features in zip(labels, labelled_features, strict=False):
        row = {f: features[i] for i, f in enumerate(feature_labels)}
        row["label"] = label
        rows.append(row)

    return pd.DataFrame.from_records(rows)


def load_dataset(raw_paths, mask_paths):
    df = pd.DataFrame()
    for raw_path, mask_path in zip(raw_paths, mask_paths, strict=False):
        raw_img = tifffile.imread(raw_path)
        mask_path = tifffile.imread(mask_path)
        frame_df = extract_ilastish_features(raw_img, mask_path)
        df = pd.concat([df, frame_df], ignore_index=True)
    return df


def train_seg_model(sample_dir: Path):
    """
    Trains a random forest classifier on a directory of raw and mask images
    using ilastik-inspired feature extraction.

    Masks should be named as the raw images with "Labels" appended, e.g.
    "raw_image.tif" and "raw_image_Labels.tif". Masks and raw images are
    implicitly paired by lexicographic sorting.

    Masks should have integer labels corresponding to classes.
    """
    raw_paths = sorted(sample_dir.glob("*[!Labels].tif"))
    mask_paths = sorted(sample_dir.glob("*Labels.tif"))

    df = load_dataset(raw_paths, mask_paths)

    y = df["label"]
    X = df.drop(columns=["label"])

    model = make_pipeline(StandardScaler(), RandomForestClassifier())

    results = cross_validate(model, X, y)
    test_scores = results["test_score"]
    logger.info("Cross-validation test scores: ", test_scores)
    return model.fit(X, y)


def predict(raw_image, model):
    df = extract_ilastish_features(raw_image, np.ones(raw_image.shape)).drop(
        columns=["label"]
    )
    predictions = model.predict(df).reshape(raw_image.shape)
    return predictions
