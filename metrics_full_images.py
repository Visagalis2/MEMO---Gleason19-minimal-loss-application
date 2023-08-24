import numpy as np
import pandas as pd
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
def make_all_metrics(anno1, anno2, mask):
    def kappa(y_a1, y_a2, conf_mat, n_classes):
        if np.sum(y_a1) == 0 and np.sum(y_a2) == 0:
            return 0.0
        elif np.sum(y_a1) == len(y_a1)**2 and np.sum(y_a2) == len(y_a2)**2:
            # True only for square patches with 1 class
            return 1.0
        else:
            sum0 = np.sum(conf_mat, axis=0)
            sum1 = np.sum(conf_mat, axis=1)
            expected = np.outer(sum0, sum1) / np.sum(sum0)
            w_mat = np.ones((n_classes, n_classes), dtype=numba.int64)
            for i in range(n_classes):
                w_mat[i, i] = 0
            k = np.sum(w_mat * conf_mat) / np.sum(w_mat * expected)
            return 1 - k
    def rk(y_a1, y_a2, conf_mat):
        # Calcul du score
        C = conf_mat
        N = np.sum(C)
        C_t = np.transpose(C)

        sum_Ck_Cl = np.sum(np.dot(C, C))
        sum_Ck_C_T_l = np.sum(np.dot(C, C_t))
        sum_C_T_k_Cl = np.sum(np.dot(C_t, C))

        cov_ypyp = N ** 2 - sum_Ck_C_T_l
        cov_ytyp = N ** 2 - sum_C_T_k_Cl

        numerator = N * np.trace(C) - sum_Ck_Cl
        if cov_ypyp * cov_ytyp == 0:
            return 0.0
        else:
            denominator = np.sqrt(cov_ypyp * cov_ytyp)
            return numerator / denominator

    labels = [0, 3, 4]
    all_metrics = []
    kappa_score = []
    iou_score = []
    dice_score = []
    rk_score = []
    perc_agreement = []
    anno_shape = max(anno1.shape)

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
    anno1[anno1 == 5] = 4
    anno2[anno2 == 5] = 4

    for label in labels:
        y1 = anno1 == label
        y2 = anno2 == label
        if np.sum(y1) == 0 and np.sum(y2) == 0 or np.sum(y1) == len(y1) and np.sum(y2) == len(y2):
            if np.sum(y1) == 0 and np.sum(y2) == 0:
                kappa_score.append(0.0)
                iou_score.append(0.0)
                dice_score.append(0.0)
                rk_score.append(0.0)
                perc_agreement.append(0.0)
            else:
                kappa_score.append(1.0)
                iou_score.append(1.0)
                dice_score.append(1.0)
                rk_score.append(1.0)
                perc_agreement.append(1.0)
        else:
            # iou
            inter = y1 * y2
            score = np.sum(inter)/(np.sum(y1) + np.sum(y2) - np.sum(inter))
            iou_score.append(score)
            dice = 2 * np.sum(inter)/(np.sum(y1) + np.sum(y2))
            dice_score.append(dice)
            # perc agreement
            inter = y1 == y2
            score = np.sum(inter)/(anno_shape*anno_shape)
            perc_agreement.append(score)

            # Matrice de confusion
            n_classes = 2
            assert y1.shape == y2.shape
            conf_mat = np.zeros((n_classes, n_classes))
            for i in range(y1.shape[0]):
                conf_mat[int(y1[i]), int(y2[i])] += 1
            # kappa
            kappa_score.append(kappa(y1, y2, conf_mat, 2))
            # rk
            rk_score.append(rk(y1, y2, conf_mat))
    # perc agreement
    inter = anno1 == anno2
    score = np.sum(inter)/(anno_shape*anno_shape)
    perc_agreement.append(score)

    # kappa
    assert anno1.shape == anno2.shape
    conf_mat = np.zeros((n_classes, n_classes))
    for i in range(anno1.shape[0]):
        conf_mat[int(anno1[i]), int(anno2[i])] += 1
    kappa_score.append(kappa(anno1, anno2, conf_mat, n_classes))
    # rk
    rk_score.append(rk(anno1, anno2, conf_mat))

    all_metrics.append(kappa_score)
    all_metrics.append(iou_score)
    all_metrics.append(rk_score)
    all_metrics.append(perc_agreement)
    all_metrics.append(dice_score)
    return all_metrics

@numba.jit(nopython=True)
def get_path(slide, main_path, maps, iou, ext):
    for i in range(iou.shape[1]):
        for j in range(i + 1, iou.shape[1]):
            pth1 = main_path + "\\" + maps[i] + "\\" + slide + ext
            pth2 = main_path + "\\" + maps[j] + "\\" + slide + ext
            yield pth1, pth2, i, j

def handle_scores(score, labels, m_tot: pd.DataFrame, name, none_value=-1):
    for i in range(len(labels)):
        score_ci = score[i]
        score_ci = score_ci[score_ci != none_value]
        if score_ci.shape[0] == 0:
            mean_score_ci = None
            std_score_ci = None
        else:
            mean_score_ci = score_ci.mean()
            std_score_ci = score_ci.std()

        m_tot.loc[m_tot['slide_core_patch'] == slide, f'mean {name}-{labels[i]}'] = mean_score_ci
        m_tot.loc[m_tot['slide_core_patch'] == slide, f'std dev {name}-{labels[i]}'] = std_score_ci

def make_mask(image: torch.tensor, with_gauss: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image.to(device)
    im_gray = T.functional.rgb_to_grayscale(image).to(device)
    im_gray = im_gray.squeeze(0)
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

def perf(slide, main_path, maps, m_tot, ext=".npy"):
    only_classes_labels = [0, 3, 4]
    classes_and_global_labels = [0, 3, 4, "glob"]

    kappa_classes = np.ones((len(classes_and_global_labels), len(maps), len(maps))) * -1
    iou_classes = np.ones((len(only_classes_labels), len(maps), len(maps))) * -1
    dice_classes = np.ones((len(only_classes_labels), len(maps), len(maps))) * -1
    rks_classes = np.ones((len(classes_and_global_labels), len(maps), len(maps))) * -2
    perc_agreement_classes = np.ones((len(classes_and_global_labels), len(maps), len(maps))) * -1

    path_img = main_path + "\\Train Imgs\\" + slide + ".jpg"
    img = np.array(Image.open(path_img)).copy()
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img2 = img.clone().detach()
    mask = make_mask(img2, with_gauss=True)

    for pth1, pth2, i, j in get_path(slide, main_path, maps, iou_classes, ext):
        if ext == ".npy":
            anno1 = np.load(pth1)
            anno2 = np.load(pth2)
        else:
            anno1 = np.array(Image.open(pth1))
            anno2 = np.array(Image.open(pth2))

        kappa_classes[:,i,j], iou_classes[:,i,j], rks_classes[:,i,j], perc_agreement_classes[:,i,j], dice_classes[:,i,j] = make_all_metrics(anno1, anno2, mask)

    handle_scores(kappa_classes, classes_and_global_labels, m_tot, "kappa", none_value=-1)
    handle_scores(iou_classes, only_classes_labels, m_tot, "iou", none_value=-1)
    handle_scores(dice_classes, only_classes_labels, m_tot, "dice", none_value=-1)
    handle_scores(rks_classes, classes_and_global_labels, m_tot, "rks", none_value=-2)
    handle_scores(perc_agreement_classes, classes_and_global_labels, m_tot, "score", none_value=-1)


if __name__ == "__main__":

    m_tot_patch = pd.read_csv(sys.argv[1])
    m_tot_patch['annotators'] = m_tot_patch['annotators'].apply(ast.literal_eval)
    slide_patch = m_tot_patch['slide_core_patch'].unique()
    ext = sys.argv[2]
    for slide in tqdm(slide_patch):
        mask = m_tot_patch['slide_core_patch'] == slide
        maps = np.array(m_tot_patch.loc[mask, 'annotators'].to_numpy()[0])
        paths = m_tot_patch.loc[mask, 'path']
        main_path = Path(paths.iloc[0]).parent.parent
        perf(slide, str(main_path), maps, m_tot_patch, ext=ext)
    m_tot_patch.to_csv(sys.argv[3])