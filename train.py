import pandas as pd
import os
import sys
import ast
import numpy as np
import torch
import torchvision
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler, DataLoader
from monai.transforms import (
    LoadImageD,
    Compose,
    MapTransform,
    NormalizeIntensityD,
    ScaleIntensityD,
    RandAxisFlipd,
    RandSpatialCropD,
    RandRotate90D,
    RandScaleIntensityD,
    ResizeD,
)
from monai.data import Dataset

from pathlib import Path
from tqdm import tqdm

from Transforms.UpdateClasses import ChangeClassD, MakeMajVoteD
from Data.ProstateLModule import ProstateLModule

def plot_informations_dataset(imgs_train, imgs_val, version, save_dir,scale="core"):
    version = version.split(".")[0]
    # plot histogram of maj_class
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x="maj_class", data=imgs_train)
    plt.title("Train")
    plt.subplot(1, 2, 2)
    sns.countplot(x="maj_class", data=imgs_val)
    plt.title("Validation")
    plt.tight_layout()
    plt.savefig(save_dir.joinpath(f"{scale}_{version}_train_val_split.png"))
    # plt.show()
    plt.close()

    # plot histogram of std dev score-glob
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(x="std dev score-glob", data=imgs_train, kde=True, alpha=.4, edgecolor='none')
    plt.title("Train")
    plt.subplot(1, 2, 2)
    sns.histplot(x="std dev score-glob", data=imgs_val, kde=True, alpha=.4, edgecolor='none')
    plt.title("Validation")
    plt.tight_layout()
    plt.savefig(save_dir.joinpath(f"{scale}_{version}_std_dev_score_glob.png"))
    # plt.show()
    plt.close()

def split_train_val(imgs, frac, random_state):
    imgs_train = imgs.sample(frac=frac, random_state=random_state)
    imgs_val = imgs.drop(imgs_train.index)
    return imgs_val

def make_dataset(full_image_dataframe, patches_dataframe):
    slide_core = full_image_dataframe["slide_core"].unique()
    dataset = patches_dataframe[patches_dataframe["slide_core"].isin(slide_core)].copy()
    # set maj_class patches the same as full image regarding the slide_core

    dataset0 = dataset[dataset["maj_class"] <= 2].copy()
    dataset3 = dataset[dataset["maj_class"] == 3].copy()
    dataset4 = dataset[dataset["maj_class"] >= 4].copy()

    dataset0["real_maj_class"] = 0
    dataset4["real_maj_class"] = 4
    dataset3["real_maj_class"] = 3
    dataset = pd.concat([dataset0, dataset3, dataset4])
    return dataset

def make_dict(df_start):
    dict = {}
    list_annotators = ["Maps1", "Maps3", "Maps4", "Maps5"]
    max_px_value = df_start["num_px"].max()
    df = df_start[df_start["num_px"] == max_px_value].copy()  # keep only 750x750 (lose 56 patches on the total)
    df["Maps1"] = False
    df["Maps3"] = False
    df["Maps4"] = False
    df["Maps5"] = False

    path_false_img = Path(r"D:\MEMO_DATA\Data\data_npy\false_img.npy")
    path_false_img = Path(r"d:\Users\MFE\Desktop\MFE_Gleason\DATA\data_npy\false_img.npy")

    for row in df.iterrows():
        for elem in row[1]["annotators"]:
            df.loc[row[0], elem] = True
    for row in df.iterrows():
        if row[1]['Maps1']:
            path = Path(row[1]["path"])
            pth_m1 = path.parent.parent.joinpath('Maps1').joinpath(path.name)
            df.loc[row[0], 'Maps1'] = str(pth_m1)
        else:
            df.loc[row[0], 'Maps1'] = str(path_false_img)
        if row[1]['Maps3']:
            path = Path(row[1]["path"])
            pth_m3 = path.parent.parent.joinpath('Maps3').joinpath(path.name)
            df.loc[row[0], 'Maps3'] = str(pth_m3)
        else:
            df.loc[row[0], 'Maps3'] = str(path_false_img)
        if row[1]['Maps4']:
            path = Path(row[1]["path"])
            pth_m4 = path.parent.parent.joinpath('Maps4').joinpath(path.name)
            df.loc[row[0], 'Maps4'] = str(pth_m4)
        else:
            df.loc[row[0], 'Maps4'] = str(path_false_img)
        if row[1]['Maps5']:
            path = Path(row[1]["path"])
            pth_m5 = path.parent.parent.joinpath('Maps5').joinpath(path.name)
            df.loc[row[0], 'Maps5'] = str(pth_m5)
        else:
            df.loc[row[0], 'Maps5'] = str(path_false_img)

    dict = [
        {"im_pth": path, 'maj_class': maj_class, "Maps1": Maps1, "Maps3": Maps3, "Maps4": Maps4, "Maps5": Maps5,
         'mean_score': mean_score, "make_majority_vote": maj_vote}
        for path, Maps1, Maps3, Maps4, Maps5, mean_score, maj_class, maj_vote in
        zip(df["path"], df["Maps1"], df["Maps3"], df["Maps4"], df["Maps5"], df["mean score-glob"], df["maj_class"], df["make_majority_vote"])
    ]
    return dict, df

def train(main_dir_project, train_path_patches, train_path_core, validation_path_patches, validation_path_core, version, save_dir, seed, batch_size=32, max_epoch=25, num_per_group=1, norm_num_groups=16, tune=True):
    # Saving path
    add_to_path = version.split(".")[0]
    save_dir = save_dir.joinpath(add_to_path)
    save_images = save_dir.joinpath("images")  # ex: D:\MEMO\MEMO-2\Saved\data_classic_train\images
    os.makedirs(save_images, exist_ok=True)
    save_models = save_dir.joinpath("models")  # ex: D:\MEMO\MEMO-2\Saved\data_classic_train\models
    os.makedirs(save_models, exist_ok=True)
    save_logs = save_dir.joinpath("logs")  # ex: D:\MEMO\MEMO-2\Saved\data_classic_train\logs
    os.makedirs(save_logs, exist_ok=True)

    imgs_patches = pd.read_csv(train_path_patches)
    imgs_core = pd.read_csv(train_path_core)
    imgs_patches["annotators"] = imgs_patches["annotators"].apply(lambda x: ast.literal_eval(x))

    val_patches = pd.read_csv(validation_path_patches)
    val_core = pd.read_csv(validation_path_core)
    val_patches["annotators"] = val_patches["annotators"].apply(lambda x: ast.literal_eval(x))

    imgs_class_0_core = imgs_core[imgs_core["maj_class"] <= 2].copy()
    imgs_class_3_core = imgs_core[imgs_core["maj_class"] == 3].copy()
    imgs_class_4_core = imgs_core[imgs_core["maj_class"] >= 4].copy()

    imgs_class_0_core["maj_class"] = 0
    imgs_class_3_core["maj_class"] = 3
    imgs_class_4_core["maj_class"] = 4

    val_class_0_core = val_core[val_core["maj_class"] <= 2].copy()
    val_class_3_core = val_core[val_core["maj_class"] == 3].copy()
    val_class_4_core = val_core[val_core["maj_class"] >= 4].copy()

    val_class_0_core["maj_class"] = 0
    val_class_3_core["maj_class"] = 3
    val_class_4_core["maj_class"] = 4

    # split train and validation (70/30)
    # Is done on the same dataset for each experiment
    imgs_class_0_val_core = split_train_val(val_class_0_core, 0.8, seed)
    imgs_class_3_val_core = split_train_val(val_class_3_core, 0.8, seed)
    imgs_class_4_val_core = split_train_val(val_class_4_core, 0.8, seed)

    # Remove the core used in the validation from the training dataset (rmk: ~ is the not operator)
    imgs_class_0_train_core = imgs_class_0_core[~imgs_class_0_core["path"].isin(imgs_class_0_val_core["path"])]
    imgs_class_3_train_core = imgs_class_3_core[~imgs_class_3_core["path"].isin(imgs_class_3_val_core["path"])]
    imgs_class_4_train_core = imgs_class_4_core[~imgs_class_4_core["path"].isin(imgs_class_4_val_core["path"])]

    # concat train and validation
    imgs_train = pd.concat([imgs_class_0_train_core, imgs_class_3_train_core, imgs_class_4_train_core])
    imgs_val = pd.concat([imgs_class_0_val_core, imgs_class_3_val_core, imgs_class_4_val_core])

    print(f"train class 0: {len(imgs_class_0_train_core)}, val class 0: {len(imgs_class_0_val_core)}")
    print(f"train class 3: {len(imgs_class_3_train_core)}, val class 3: {len(imgs_class_3_val_core)}")
    print(f"train class 4: {len(imgs_class_4_train_core)}, val class 4: {len(imgs_class_4_val_core)}")

    plot_informations_dataset(imgs_train, imgs_val, version, save_images, "core")

    train_patches = make_dataset(imgs_train, imgs_patches)
    val_patches = make_dataset(imgs_val, imgs_patches)

    plot_informations_dataset(train_patches, val_patches, version, save_images,"patches")

    patch_size = (512,512)
    # Transforms
    ## made by calculate_mean_std.py on the train set
    mean = np.array([202.0163435, 123.11469512, 173.71647922])
    std = np.array([33.333138, 50.4682048, 33.92489938])

    tr_transform = Compose([
        LoadImageD(keys=['im_pth','Maps1', 'Maps3', 'Maps4', 'Maps5']),
        NormalizeIntensityD(keys=["im_pth"], subtrahend=mean, divisor=std, channel_wise=True),
        RandSpatialCropD(keys=['im_pth','Maps1', 'Maps3', 'Maps4', 'Maps5'], roi_size=patch_size, max_roi_size=patch_size
                         ,random_center=True, random_size=False),
        RandAxisFlipd(keys=['im_pth','Maps1', 'Maps3', 'Maps4', 'Maps5'], prob=0.25),
        RandRotate90D(keys=['im_pth','Maps1', 'Maps3', 'Maps4', 'Maps5'], prob=0.25,),
        RandScaleIntensityD(keys=['im_pth'], factors=0.1, prob=0.3),
        ScaleIntensityD(keys=['im_pth'], minv=0.0, maxv=1.0),
        MakeMajVoteD(keys=['Maps1', 'Maps3', 'Maps4', 'Maps5']),
        ChangeClassD(keys=['Maps1', 'Maps3', 'Maps4', 'Maps5']),
    ]).set_random_state(seed=seed)

    val_transform = Compose([
        LoadImageD(keys=['im_pth','Maps1', 'Maps3', 'Maps4', 'Maps5']),
        NormalizeIntensityD(keys=["im_pth"], subtrahend=mean, divisor=std, channel_wise=True),
        ScaleIntensityD(keys=['im_pth'], minv=0.0, maxv=1.0),
        MakeMajVoteD(keys=['Maps1', 'Maps3', 'Maps4', 'Maps5']),
        ChangeClassD(keys=['Maps1', 'Maps3', 'Maps4', 'Maps5']),
    ])

    # Datasets
    train_dict, train_patches = make_dict(train_patches)
    val_dict, val_patches = make_dict(val_patches)  # path, Maps1, Maps3, Maps4, Maps5, mean_score, maj_class

    train_dataset = Dataset(data=train_dict, transform=tr_transform)
    val_dataset = Dataset(data=val_dict, transform=val_transform)

    class_freq = train_patches["maj_class"].value_counts()
    samples_weight = np.array([1 / class_freq[i] for i in train_patches["maj_class"].values])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=5, sampler=sampler,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=5, drop_last=True)

    # dict_class = {"Benign": 0, "3": 0, "High": 0}
    # for batch in train_loader:
    #     classes = batch["maj_class"]
    #     for elem in classes:
    #         if elem <= 2:
    #             dict_class["Benign"] += 1
    #         elif elem == 3:
    #             dict_class["3"] += 1
    #         else:
    #             dict_class["High"] += 1

    # Callbacks
    callbacks = [RichProgressBar(),
                 ModelCheckpoint(dirpath=str(save_models), monitor='val_loss', filename='sample-gleason-{epoch:02d}-{val_loss:.2f}',
                                                    every_n_epochs=1, save_top_k=-1, mode='min')]
    logger = TensorBoardLogger(str(save_logs))
    # Model
    print(torch.cuda.is_available())
    # LR found with the tuner for the majority of the training with both b16 and b32
    dm = ProstateLModule(lr=0.007686, batch_size=batch_size, num_per_group=num_per_group, norm_num_groups=norm_num_groups, save_dir=str(save_dir), tune=tune)
    dm.setup(stage='fit')

    trainer = L.Trainer(devices='auto', max_epochs=max_epoch, callbacks=callbacks, log_every_n_steps=2, logger=logger,
                        deterministic=True)
    if tune:
        tuner = L.pytorch.tuner.tuning.Tuner(trainer)
        lr_finder = tuner.lr_find(dm, train_loader, val_loader, min_lr=1e-6, max_lr=1e-2, num_training=35)
        fig, ax = plt.subplots()
        lr_finder.plot(ax=ax, suggest=True)
        ax.set_xscale('log')
        plt.savefig(save_dir.joinpath("lr_finder.png"))
        plt.close()
        new_lr = lr_finder.suggestion()
        print(f"New learning rate: {new_lr}")
    else:
        dm.lr = 0.007686  # put lr found in previous step
    print(f"Learning rate: {dm.lr}")
    print(dm.hparams)
    trainer.fit(dm, train_loader, val_loader)

if __name__ == '__main__':
    # take the arguments
    seed = int(sys.argv[1])
    version = sys.argv[2]
    batch_size = int(sys.argv[3])
    exp = sys.argv[4]
    max_epoch = int(sys.argv[5])
    num_per_group = int(sys.argv[6])
    norm_num_groups = int(sys.argv[7])
    main_dir_project = Path(sys.argv[8])

    train_path_patches = Path(sys.argv[9])
    train_path_core = Path(sys.argv[10])
    validation_path_patches = Path(sys.argv[11])
    validation_path_core = Path(sys.argv[12])
    tune = bool(int(sys.argv[13]))

    L.pytorch.seed_everything(seed, workers=True)

    print(f"PyTorch Lightning version: {L.__version__}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")

    save_dir = main_dir_project.joinpath("Saving_files", str(seed), exp, str(num_per_group))
    os.makedirs(save_dir, exist_ok=True)

    train(main_dir_project, train_path_patches, train_path_core, validation_path_patches, validation_path_core, version, save_dir, seed, batch_size, max_epoch, num_per_group, norm_num_groups, tune)
