"""
Implement an image patcher selecting only the patches that are not empty
using torch tensors.
"""
from skimage.morphology import isotropic_closing, disk

import sys
import os
import time

import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional, Callable
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Prostate_image():
    name_image: str
    path_image: Path
    centers: tuple = None
    size: tuple = None
    annotations: list[Path] = None
    annotators: list[str] = None
    saving_paths: list[Path] = None

    def __post_init__(self):
        # Get the center of the image
        if self.annotations is None:
            annotators = ["Maps1", "Maps3", "Maps4", "Maps5"]
            paths_to_test = [self.path_image.parent.parent.joinpath(annotator+"/"+self.name_image+"_classimg_nonconvex.png") for annotator in annotators]
            self.annotations = [path for path in paths_to_test if path.exists()]
            self.annotations.append(self.path_image)
            self.annotators = [path.parent.name for path in self.annotations]
            save_path = self.path_image.parent.parent.joinpath("data_npy")
            self.saving_paths = [save_path.joinpath(annotator+"/"+self.name_image) for annotator in self.annotators]
            for path in self.saving_paths:
                path.parent.mkdir(parents=True, exist_ok=True)


    def __repr__(self):
        return (f'''This is a {self.__class__.__name__} called {self.name_image}.''')

def make_mask(image: torch.Tensor, with_gauss: bool = True):
    im_gray = T.functional.rgb_to_grayscale(image).to(device)
    if with_gauss:
        im_gauss = T.GaussianBlur(61, 50)(im_gray)
        im_np = im_gauss.squeeze().cpu().numpy()
        th = threshold_otsu(im_gauss.cpu().numpy())
    else:
        im_np = im_gray.squeeze().cpu().numpy()
        th = threshold_otsu(im_gray.cpu().numpy())
    im_np_th = (im_np < th).astype(np.uint8)
    im_filled = ndi.binary_fill_holes(im_np_th)
    return torch.from_numpy(im_filled).unsqueeze(0).to(device)

def save_patch(center: Tuple[int], p_image: Prostate_image, patch_size: int, i: int):
    with open(path_remove, "r") as f:
        bad_paths = f.read().splitlines()
    t1 = time.time()
    for annotation, path in zip(p_image.annotations, p_image.saving_paths):
        # check path
        path_tmp = Path(path)
        path_tmp = path_tmp.with_name(path_tmp.stem)
        if str(path_tmp) in bad_paths:
            continue
        # Load the image
        image = read_image(str(annotation)).to(device)
        # Get the patch
        patch = image[:, center[1] - patch_size//2:center[1] + patch_size//2, center[0] - patch_size//2:center[0] + patch_size//2]
        # Save the patch
        saving_path = str(path) + f"_patch_{i}.npy"
        np.save(saving_path, patch.cpu().numpy())



def extract_centers(paths: list[Path], patch_size: int =256, stride: int =128, perc_filled: float = 0.5) -> list[torch.Tensor]:
    # Make the patches
    p_images = []
    t0 = time.time()
    for path in tqdm(paths):
        p_image = Prostate_image(path.stem,path)
        image = read_image(str(path)).to(device)
        t1 = time.time()
        mask = make_mask(image)
        t2 = time.time()
        p_image.size = image.shape[1:]
        # Get the dimensions of the image
        height, width = mask.shape[1:]
        # Get the number of patches
        n_patches = ((height - patch_size) // stride + 1) * ((width - patch_size) // stride + 1)
        # Make the patches
        count_patches = 0
        centers = []
        t3 = time.time()
        for i in range(n_patches):
            # Get the coordinates of the patch
            x = i % ((height - patch_size) // stride + 1)
            y = i // ((height - patch_size) // stride + 1)
            # Get the patch
            patch = mask[:, y * stride:y * stride + patch_size, x * stride:x * stride + patch_size]
            # apply threshold to the patch
            if torch.count_nonzero(patch) > perc_filled * patch_size ** 2:

                count_patches += 1
                center = (x * stride + patch_size // 2, y * stride + patch_size // 2)
                centers.append(center)
                # Save the patch
                save_patch(center, p_image, patch_size, i)

        t4 = time.time()
        p_image.centers = centers
        p_images.append(p_image)
    t5 = time.time()
    return p_images

def clean_paths(paths: list[Path], path_remove: Path) -> list[Path]:
    with open(path_remove, "r") as f:
        bad_paths = f.readlines()
    for path in paths:
        path_tmp = Path(path)
        path_tmp = path_tmp.with_name(path_tmp.stem)
        if path_tmp in bad_paths:
            paths.remove(path)
    return paths

def main(dir: Path, path_remove: Path):
    # Get the paths of jpg images from the dir
    paths = [dir.joinpath(path) for path in os.listdir(dir) if path.endswith('.jpg')]

    # Open the images using torch
    # Set variables
    patch_size = 750
    stride = patch_size // 2

    # Make the patches
    p_images = extract_centers(paths, patch_size, stride)

    # Save


if __name__ == "__main__":
    dir = Path(sys.argv[1])
    path_remove = Path(sys.argv[2])
    main(dir, path_remove)