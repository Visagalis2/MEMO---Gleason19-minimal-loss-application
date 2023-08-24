import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import ast
import numba
import PIL.Image as Image
import sys
from pathlib import Path
import torch
from monai.transforms import CenterSpatialCrop
import torchvision.transforms as T
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt


@numba.jit(nopython=True, parallel=True)
def make_conf_mat(anno1, anno2, mask):
    dices = []
    out = []

    mask = mask.flatten()
    anno1 = anno1.flatten()
    anno2 = anno2.flatten()
    anno1 = anno1[mask == 1]
    anno2 = anno2[mask == 1]

    anno1[anno1 == 1] = 0
    anno2[anno2 == 1] = 0
    anno1[anno1 == 2] = 0
    anno2[anno2 == 2] = 0
    anno1[anno1 == 6] = 0
    anno2[anno2 == 6] = 0
    anno1[anno1 == 3] = 1
    anno2[anno2 == 3] = 1
    anno1[anno1 == 4] = 2
    anno2[anno2 == 4] = 2
    anno1[anno1 == 5] = 2
    anno2[anno2 == 5] = 2

    for label in [0,1,2]:
        y1 = np.sum(anno1 == label)
        y2 = np.sum(anno2 == label)
        inter = np.sum((anno1 == label) & (anno2 == label))
        if y1 + y2 != 0:
            dices.append(2 * inter / (y1 + y2))
        else:
            dices.append(0)
    n_classes = 3
    dices = np.array(dices)
    assert anno1.shape == anno2.shape
    conf_mat = np.zeros((n_classes, n_classes))
    for i in range(anno1.shape[0]):
        conf_mat[int(anno1[i]), int(anno2[i])] += 1
    conf_mat_flat = conf_mat.flatten().astype(np.float64)
    out.append(conf_mat_flat)
    out.append(dices)
    return out


def get_path(slide, main_path, maps, num_anno, ext1, ext2):
    for i in range(num_anno):
        for j in range(i + 1, num_anno):
            pth1 = main_path + "\\" + maps[i] + "\\" + slide + ext1
            pth2 = main_path + "\\" + maps[j] + "\\" + slide + ext2
            yield pth1, pth2, i, j, maps[0] ,maps[1]

def handle_scores(score, labels, m_tot: pd.DataFrame, name, none_value=-1):
    for i in range(len(labels)):
        score_ci = score[i]
        score_ci = score_ci[score_ci != none_value]
        m_tot.loc[m_tot['slide_core_patch'] == slide, f'{name}-{labels[i]}'] = score_ci


def handle_conf_mat(score, labels, m_tot: pd.DataFrame,):
    for i in range(len(labels)):
        for j in range(len(labels)):
            m_tot.loc[m_tot['slide_core_patch'] == slide, f'{labels[i]}-{labels[j]}'] = score[i, j]


def make_mask(image: torch.tensor, with_gauss: bool = True, make_crop: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image.to(device)
    im_gray = T.functional.rgb_to_grayscale(image).to(device)
    im_gray = im_gray.squeeze(0)
    if make_crop:
        im_gray = CenterSpatialCrop((4608,4608))(im_gray)
    if with_gauss:
        im_gauss = T.GaussianBlur(61, 50)(im_gray)
        im_np = im_gauss.squeeze().cpu().numpy()
        th = threshold_otsu(im_gauss.cpu().numpy())
    else:
        im_np = im_gray.squeeze().cpu().numpy()
        th = threshold_otsu(im_gray.cpu().numpy())
    im_np_th = (im_np < th).astype(np.uint8)
    im_filled = ndi.binary_fill_holes(im_np_th)
    return im_filled

def perf(slide, main_path, maps, m_tot, ext1=".npy", ext2=".png",make_crop=True):
    only_classes_labels = [0, 3, 4]

    conf_mat = np.zeros((3, 3))
    dice_classes = np.ones((len(only_classes_labels), len(maps), len(maps))) * -1


    if len(maps) >= 2:
        for pth1, pth2, i, j, m1, m2 in get_path(slide, main_path, maps, 2, ext1, ext2):
            if Path(pth1).exists() and Path(pth2).exists():
                if pth1.endswith(".npy"):
                    anno1 = np.load(pth1)
                else:
                    anno1 = np.array(Image.open(pth1))
                    if m1 in ["Maps1", "Maps3", "Maps4", "Maps5"]:
                        annotations = torch.from_numpy(anno1).unsqueeze(0)
                        annotations = CenterSpatialCrop((4608, 4608))(annotations)
                        anno1 = annotations.squeeze(0).cpu().numpy()
                if pth2.endswith(".npy"):
                    anno2 = np.load(pth2)
                else:
                    anno2 = np.array(Image.open(pth2))
                    if m2 in ["Maps1", "Maps3", "Maps4", "Maps5"]:
                        annotations = torch.from_numpy(anno2).unsqueeze(0)
                        annotations = CenterSpatialCrop((4608, 4608))(annotations)
                        anno2 = annotations.squeeze(0).cpu().numpy()

                path_img = main_path + "\\Train Imgs\\" + slide + ".jpg"
                img = np.array(Image.open(path_img)).copy()
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                img2 = img.clone().detach()
                mask = make_mask(img2, with_gauss=True, make_crop=make_crop)
                tmp_conf_mat, dices = make_conf_mat(anno1, anno2, mask)
                dices = np.array(dices)
                dice_classes[:, i, j] = dices
                conf_mat += np.array(tmp_conf_mat).reshape(3, 3)
            else:
                print(f"Missing {pth1} or {pth2}")
                print(f"Existe {Path(pth1).exists()} or {Path(pth2).exists()}")
                import sys
                sys.exit()
        handle_conf_mat(conf_mat, only_classes_labels, m_tot)
        handle_scores(dice_classes, only_classes_labels, m_tot, "dice", none_value=-1)


if __name__ == "__main__":

    m_tot_patch = pd.read_csv(sys.argv[1])
    m_tot_patch['annotators'] = m_tot_patch['annotators'].apply(ast.literal_eval)
    slide_patch = m_tot_patch['slide_core_patch'].unique()
    ext1 = sys.argv[2]
    ext2 = sys.argv[3]
    make_crop = bool(sys.argv[5])
    for slide in tqdm(slide_patch):
        mask = m_tot_patch['slide_core_patch'] == slide
        maps = np.array(m_tot_patch.loc[mask, 'annotators'].to_numpy()[0])
        paths = m_tot_patch.loc[mask, 'path']
        main_path = Path(paths.iloc[0]).parent.parent
        perf(slide, str(main_path), maps, m_tot_patch, ext1=ext1, ext2=ext2, make_crop=make_crop)
    save_path = Path(sys.argv[4]).parent
    os.makedirs(save_path, exist_ok=True)
    m_tot_patch.to_csv(sys.argv[4])