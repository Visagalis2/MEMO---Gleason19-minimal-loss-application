from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
import torch
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torch.nn import Softmax
from monai.transforms import Compose, LoadImage, CenterSpatialCrop, NormalizeIntensity, ScaleIntensity


def binarize(img):
    img = img.numpy()
    img = np.argmax(img, axis=0)
    return img

if __name__ == "__main__":
    versions = ["validation_selection"]
    models = [model_pth for model_pth in os.listdir(r"D:\MEMO\MEMO-END2\final\validation_selection\models")]

    print(models)
    image_paths = [r"D:\MEMO_DATA\Data\Train Imgs\slide006_core125.jpg"]
    mean = np.array([202.0163435, 123.11469512, 173.71647922])
    std = np.array([33.333138, 50.4682048, 33.92489938])
    test_transform = Compose([
                         LoadImage(ensure_channel_first=True, image_only=True),
                         CenterSpatialCrop(roi_size=(4608, 4608)),
                         NormalizeIntensity(subtrahend=mean, divisor=std, channel_wise=True),
                         ScaleIntensity(minv=0.0, maxv=1.0),])
    preds = []
    for i, m in tqdm(enumerate(models), total=len(models)):
        models_path = fr"D:\MEMO\MEMO-END\final\final\{versions[0]}\models\{m}"
        model = UNet(spatial_dims=2,
                     in_channels=3,
                     out_channels=3,
                     channels=(32, 64, 128, 256),
                     strides=(2, 2, 2),
                     norm=("group", {"num_groups": 16}),  # following https://sci-hub.se/10.1109/LRA.2019.2896518
                     num_res_units=0,
                     bias=False,
                     adn_ordering="NA") 
        state = torch.load(models_path, map_location=torch.device('cpu'))['state_dict']
        for key in list(state.keys()):
            if "model" in key:
                state[key.replace("model.m", "m")] = state.pop(key)
        model.load_state_dict(state)
        model.eval()
        image_paths = image_paths[:65]
        for img_pth in image_paths:
            img = test_transform([img_pth])
            img = img[0].unsqueeze(0)
            with torch.no_grad():
                logits = sliding_window_inference(img, (512, 512), img.shape[0], model, mode='gaussian')
                pred = torch.nn.Softmax(dim=1)(logits.as_tensor())
                pred = pred.squeeze(0)
                pred = binarize(pred)
                pred[pred==1]=3
                pred[pred==2]=4
                preds.append(pred)
                print(len(preds))
                name = img_pth.split("\\")[-1].replace(".jpg", ".png")

    plt.figure(figsize=(10, 12))
    plt.suptitle("Predictions on slide006 core125")
    for i, pre in enumerate(preds):
        plt.subplot(4, 3, i + 1)
        plt.title(f"Epoch {i + 1}")
        plt.imshow(pre, vmin=0, vmax=5, cmap="viridis", interpolation="none")
        plt.axis("off")
        plt.colorbar()
    plt.tight_layout()
    plt.savefig(fr"D:\MEMO\MEMO-END2\images\slide006_core125.png")
    plt.show()
    plt.close()