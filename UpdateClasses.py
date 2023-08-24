import torch
from typing import Optional, Tuple
from monai.transforms import MapTransform

import matplotlib.pyplot as plt
import numpy as np

def as_one_hot(tensor: torch.Tensor, labels: Optional[Tuple[int, ...]] = (0, 1, 2, 3, 4, 5)) -> torch.Tensor:
    """
    This function take a tensor (label image) of shape [1,H,W] and the list of the present values
    and return this label to the one_hot format with shape [N,H,W] with N = number of classes,
    and with values equal to 1.0 or 0.0.

    Parameters
    ----------
    tensor : torch.Tensor
        The label with a [1,512,512] shape and values contained in (0, 1, 2, 3, 4, 5).
    labels : Optional[List[int]]=(0, 1, 2, 3, 4, 5)
        A list with each labels.

    Returns
    -------
    labels_one_hot : torch.Tensor
        Return the label to a one_hot format (shape = [len(labels),512,512] and values are
                                              0.0 or 1.0.)
    """
    combined_one_hot = []

    for class_label in labels:
        # torch.where(tensor of boolean, x, y) replace the True values by x and
        # the y values by y
        binary_label = torch.where(
            torch.eq(tensor, class_label),
            torch.tensor(1.0), torch.tensor(0.0))
        combined_one_hot.append(binary_label)

    # stack the list of tensor along the first dimension to have shape [N,H,W]
    labels_one_hot = torch.cat(combined_one_hot, dim=0)

    if 7 in tensor:
        # print(f"There is a 7 in the tensor")
        labels_one_hot = torch.ones_like(labels_one_hot)*7

    return labels_one_hot

class ChangeClassD(MapTransform):

    def __call__(self, data):
        for key in self.keys:
            tmp = data[key].as_tensor()
            if 2 in tmp:
                # print(f"There is a 2 in the tensor")
                tmp[tmp == 2] = 1
                data[key].set_array(tmp)
            if 6 in tmp:
                # print(f"There is a 6 in the tensor")
                tmp[tmp == 6] = 1

            if 5 in tmp:
                tmp[tmp == 5] = 4
            if 1 in tmp:
                tmp[tmp == 1] = 0

            tmp = as_one_hot(tmp, labels=(0, 3, 4))
            data[key].set_array(tmp)
        return data

class MakeMajVoteD(MapTransform):

    def __call__(self, data, validation=False):
        all_maps = []
        if not data["make_majority_vote"]:
            return data

        for key in self.keys:
            if 7 in data[key].as_tensor(): continue
            all_maps.append(data[key].as_tensor())
        all_maps = torch.stack(all_maps, dim=0)
        maj_vote_img = torch.mode(all_maps, dim=0)
        for key in self.keys:
            data[key].set_array(maj_vote_img[0])
        return data
