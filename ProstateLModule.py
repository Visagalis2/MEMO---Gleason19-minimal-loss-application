import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

from pathlib import Path
from dataclasses import dataclass
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
# from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, random_split
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.visualize import plot_2d_or_3d_image
from typing import List, Dict, Tuple, Union, Optional
from statistics import mode

class ProstateLModule(LightningModule):

    def __init__(self, lr, batch_size, num_per_group, norm_num_groups, loss="DiceLoss", save_dir="Saving_files", tune=True):
        super().__init__()
        self.save_hyperparameters()
        print(norm_num_groups)
        self.model = UNet(spatial_dims=2,
                          in_channels=3,
                          out_channels=3,  
                          channels=(32, 64, 128, 256),
                          strides=(2, 2, 2),
                          norm=("group", {"num_groups": norm_num_groups}),  # following https://sci-hub.se/10.1109/LRA.2019.2896518
                          num_res_units=0,
                          bias=False,
                          adn_ordering="NA")
        if loss == "DiceLoss":
            self.loss = DiceLoss(softmax=True, include_background=True, reduction="none")
        elif loss == "CrossEntropyLoss":
            self.loss = torch.nn.CrossEntropyLoss()
        self.dice_metric = DiceMetric(include_background=True, reduction="none", ignore_empty=False)
        self.annotators = ["Maps1", "Maps3", "Maps4", "Maps5"]
        self.save_dir = save_dir

        self.chosen_annotator = {"Maps1": 0, "Maps3": 0, "Maps4": 0, "Maps5": 0}
        self.training_dice_score = []
        self.validation_dice_score = []
        self.test_dice_score = []
        self.training_loss = []
        self.validation_loss = []
        self.test_loss = []
        self.training_agreement = []
        self.validation_agreement = []
        self.test_agreement = []
        self.df_chosen_annotator = pd.DataFrame(columns=["Maps1", "Maps3", "Maps4", "Maps5"])

        val = {"val_loss": None, "val_dice": None,
             "perc_score": None,}
        test = {"test_loss": None, "test_dice": None,
             "perc_score": None,}
        self.df_all_val_data = pd.DataFrame(val, index=[0])
        self.df_all_test_data = pd.DataFrame(test, index=[0])
        tr = {"tr_loss": None, "tr_dice": None,
              "maps": str([-1,-1,-1,-1]),
             "perc_score": None,}
        self.df_all_tr_data = pd.DataFrame(tr, index=[0])


        self.csv_path = str(Path(self.save_dir).joinpath("csv"))
        self.images_path = str(Path(self.save_dir).joinpath("images"))
        os.makedirs(self.csv_path, exist_ok=True)
        os.makedirs(self.images_path, exist_ok=True)

        self.lr = lr
        self.batch_size = batch_size
        self.num_per_group = num_per_group
        self.num_group = batch_size // num_per_group
        self.tune = tune


    def make_pairwise_agreement_maps(self, preds: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """
            Calculate the pairwise agreement between predicted `preds` and ground truth `ground_truth`.

            Args:
                preds (torch.Tensor): The predicted tensor with the model's output.
                ground_truth (torch.Tensor): The ground truth tensor.

            Returns:
                score (torch.Tensor(float)): The pairwise agreement score between `preds` and `ground_truth`.
        """
        inter = preds == ground_truth
        score = torch.sum(inter) / inter.numel()
        return score

    def bar_plot(self,title: str = "") -> None:
        """
        Create a bar plot using the data from the `chosen_annotator` dictionary and save the plot as an image.

        Args:
            title (str, optional): Title for the bar plot. Default is an empty string.

        Returns:
            None
        """
        data = self.chosen_annotator
        values = list(data.values())
        keys = list(data.keys())
        path = Path(self.images_path).joinpath(f"{self.batch_size}_{self.num_per_group}_annotators_epoch_{self.current_epoch}.png")
        plt.figure(figsize=(8,6), dpi=300)
        plt.bar(keys, values)
        plt.title(title)
        plt.savefig(path)
        plt.close()

    def execute_loss(self, logits: torch.Tensor, y: List[torch.Tensor], val: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the loss for the given logits and ground truth annotations. Return a list of annotator indices that
        produced the minimum loss for each group.

        Args:
            logits (torch.Tensor): The model's output logits.
            y (List[torch.Tensor]): A list of ground truth tensors for different annotators.
            val (bool, optional): If True, calculates the validation loss for the first annotator in the `y` list because
                                  there is only consensus ground truth in the validation set.
                                  Default is False.

        Returns:
            final_loss (torch.Tensor): The final loss calculated as the mean of the minimum loss from each group.
            annotators_per_group (Optional[torch.Tensor]): If `val=False`, returns a list containing the index of the
                                                        annotator with the minimum loss for each group. If `val=True`,
                                                        returns None.
        """

        if val:
            return self.loss(logits, y[0]).mean().unsqueeze(0)

        batch_loss_reduced = torch.ones(self.num_group)
        annotators_per_group = torch.zeros(self.num_group, dtype=torch.int)

        for i in range(self.num_group):
            # Divise the batch into groups of size `num_per_group`
            block = torch.ones((self.num_per_group, 4))
            for j in range(self.num_per_group):
                # within the sub-batch, compute the loss for each annotator
                im_index = i * self.num_per_group + j
                for k, anno in enumerate(y):
                    if 7 not in anno[im_index]:
                        output = logits[im_index].unsqueeze(0)
                        ground_truth = anno[im_index].unsqueeze(0)
                        block[j][k] = self.loss(output, ground_truth).mean().unsqueeze(0)
                    else:
                        block[j][k] = torch.tensor([7.0], requires_grad=True) # 7 > max(Dice loss)
            group_mean = block.mean(dim=0)
            # get the minimum mean loss for each group and keep the index of the chosen annotator
            batch_loss_reduced[i] = group_mean.min()
            annotators_per_group[i] = group_mean.argmin()
        final_loss: torch.Tensor = batch_loss_reduced.mean()
        return final_loss, annotators_per_group

    def execute_metrics(self, preds: torch.Tensor, y: List[torch.Tensor], annotators_per_group: Optional[List[int]] = None, val: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate evaluation metrics (dice and pairwise agreement) for the predicted `preds` and ground truth
        annotations `y`.
        The best annotator (with the minimum loss) for each group can be provided using `annotators_per_group`.

        Args:
            preds (torch.Tensor): The model's predicted tensor.
            y (List[torch.Tensor]): A list of ground truth tensors for different annotators.
            annotators_per_group (Optional[List[int]], optional): A list containing the index of the annotator with the
                                                                  minimum loss for each group. Default is None.
            val (bool, optional): If True, calculates the validation metrics for the first annotator in the `y` list.
                                  Default is False.

        Returns:
            batch_dice_reduced (torch.Tensor): The mean dice score over all groups.
            batch_agreement_reduced (torch.Tensor): The mean pairwise agreement score over all groups.
        """

        if val:
            return self.dice_metric(preds, y[0]).mean().unsqueeze(0), self.make_pairwise_agreement_maps(preds, y[0]).unsqueeze(0)

        batch_dice_reduced = torch.zeros((self.num_group))
        batch_agreement_reduced = torch.zeros((self.num_group))

        for i in range(self.num_group):
            block = torch.zeros((2, self.num_per_group))
            best_annotator_block = int(annotators_per_group[i].item())
            for j in range(self.num_per_group):
                im_index = i * self.num_per_group + j
                anno = y[best_annotator_block]
                if 7 not in anno[im_index]:
                    output = preds[im_index].unsqueeze(0)
                    ground_truth = anno[im_index].unsqueeze(0)
                    dice = self.dice_metric(output, ground_truth).mean().unsqueeze(0)
                    block[0][j] = self.dice_metric(output, ground_truth).mean().unsqueeze(0)
                    block[1][j] = torch.tensor([self.make_pairwise_agreement_maps(output, ground_truth)], device=self.device)
                else:
                    block[0][j] = torch.tensor([0.0])
                    block[1][j] = torch.tensor([0.0])
            group_mean = block.mean(dim=1)  # mean over j -> on the block
            batch_dice_reduced[i] = group_mean[0]
            batch_agreement_reduced[i] = group_mean[1]
        return batch_dice_reduced.mean().unsqueeze(0), batch_agreement_reduced.mean().unsqueeze(0)


    def on_train_end(self) -> None:
        """
            This method is called at the end of the training process. It saves the `chosen_annotator` dictionary,
            `df_all_val_data`, and `df_all_tr_data` DataFrames as CSV files.

            Returns:
                None
        """
        self.df_chosen_annotator.to_csv(Path(self.csv_path).joinpath("chosen_annotator.csv"))
        self.df_all_val_data.to_csv(Path(self.csv_path).joinpath("all_val_data.csv"))
        self.df_all_tr_data.to_csv(Path(self.csv_path).joinpath("all_tr_data.csv"))

    def on_test_end(self) -> None:
        """
            This method is called at the end of the testing process. It saves the `chosen_annotator` dictionary,
            `df_all_val_data`, and `df_all_tr_data` DataFrames as CSV files.

            Returns:
                None
        """
        self.df_all_test_data.to_csv(Path(self.csv_path).joinpath("all_test_data.csv"))

    def on_train_epoch_start(self) -> None:
        """
        This method is called at the start of each training epoch. It logs the learning rate (`lr`) to the output.

        Returns:
            None
        """
        if not self.tune:
            pass  # there is no update

        self.log("lr", self.lr)

    def on_train_epoch_end(self) -> None:
        """
        This method is called at the end of each training epoch. It calculates and logs the mean dice score
        (`Tr_mean_epoch_dice_score`), mean agreement (`Tr_mean_epoch_agreement`), and mean loss (`Tr_mean_epoch_loss`)
        over the training epoch.

        Additionally, it creates a bar plot for the training annotators and saves it as an image. It also appends the
        current epoch's `chosen_annotator` dictionary to the `df_chosen_annotator` DataFrame and resets the
        `chosen_annotator`, `training_dice_score`, `training_loss`, and `training_agreement` lists.

        Returns:
            None
        """
        if len(self.training_dice_score) != 0:
            mean_dice = torch.stack(self.training_dice_score).mean()
            self.log('Tr_mean_epoch_dice_score', mean_dice.item(), prog_bar=True)
        if len(self.training_agreement) != 0:
            mean_agreement = torch.stack(self.training_agreement).mean()
            self.log('Tr_mean_epoch_agreement', mean_agreement.item(), prog_bar=True)
        if len(self.training_loss) != 0:
            mean_loss = torch.stack(self.training_loss).mean()
            self.log('Tr_mean_epoch_loss', mean_loss.item(), prog_bar=True)

        self.bar_plot(title="Training annotators")
        self.df_chosen_annotator = pd.concat([self.df_chosen_annotator,
                                              pd.DataFrame(self.chosen_annotator, index=[self.current_epoch])])

        self.chosen_annotator = {"Maps1": 0, "Maps3": 0, "Maps4": 0, "Maps5": 0}
        self.training_dice_score.clear()
        self.training_loss.clear()
        self.training_agreement.clear()

    def on_validation_epoch_end(self) -> None:
        """
        This method is called at the end of each validation epoch. It calculates and logs the mean dice score
        (`Val_mean_epoch_dice_score`), mean loss (`Val_mean_epoch_loss`), and mean agreement
        (`Val_mean_epoch_agreement`) over the validation epoch.

        Additionally, it clears the `validation_dice_score`, `validation_loss`, and `validation_agreement` lists.

        Returns:
            None
        """
        if len(self.validation_dice_score) != 0:
            mean_dice = torch.stack(self.validation_dice_score).mean()
            self.log('Val_mean_epoch_dice_score', mean_dice.item(), prog_bar=True)
        if len(self.validation_loss) != 0:
            mean_loss = torch.stack(self.validation_loss).mean()
            self.log('Val_mean_epoch_loss', mean_loss.item(), prog_bar=True)
        if len(self.validation_agreement) != 0:
            mean_agreement = torch.stack(self.validation_agreement).mean()
            self.log('Val_mean_epoch_agreement', mean_agreement.item(), prog_bar=True)

        self.validation_dice_score.clear()
        self.validation_loss.clear()
        self.validation_agreement.clear()

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for the training process.

        Returns:
            dict: A dictionary containing the optimizer and the learning rate scheduler.
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.tune:
            return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(self.optimizer,
                                               mode='min',
                                               patience=5,
                                               min_lr = 1e-6,
                                               factor=0.1,
                                               threshold=5e-4,
                                               verbose=True),
                "monitor": 'val_loss',
                },
            }
        else:
            return { "optimizer": self.optimizer }

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (Dict[str, torch.Tensor]): A batch of data containing 'im_pth', and the 4 batches of annotations.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The loss for the current training step.
        """

        # print(batch)
        x, y = batch['im_pth'], [batch['Maps1'], batch['Maps3'], batch['Maps4'], batch['Maps5']]
        logits = self.model(x)
        loss, annotators_per_group = self.execute_loss(logits, y)

        batch_annotators_count = [0 for i in range(4)]

        for anno_block in annotators_per_group:
            self.chosen_annotator[self.annotators[anno_block.item()]] += 1
            batch_annotators_count[anno_block] += 1

        preds = torch.nn.Softmax(dim=1)(logits.as_tensor())  # dim = 1 because we have BxCxHxW and we want to softmax over C
        preds = AsDiscrete(threshold=0.7)(preds).as_tensor()

        mean_dice_score, mean_agreement_score = self.execute_metrics(preds, y, annotators_per_group=annotators_per_group)

        self.training_dice_score.append(mean_dice_score)
        self.training_loss.append(loss)
        self.training_agreement.append(mean_agreement_score)

        values = {"tr_loss": loss.item(), "tr_dice": mean_dice_score.item(),
                  "maps": str(batch_annotators_count),
                  "perc_score": mean_agreement_score.item(),}

        last_index = self.df_all_tr_data.index[-1]

        self.df_all_tr_data = pd.concat((self.df_all_tr_data, pd.DataFrame(values, index=[last_index+1])))

        self.log('train_loss', loss.item(), prog_bar=True, batch_size=x.shape[0])
        self.log('train_dice', mean_dice_score.item(), prog_bar=True, batch_size=x.shape[0])

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (Dict[str, torch.Tensor]): A batch of data containing 'im_pth' and a batch with a batch with consensus
                                             annotations.
            batch_idx (int): Index of the current batch.

        Returns:
            None
        """
        # print(batch)
        x, y = batch['im_pth'], [batch['Maps1']]  # note that each maps is a maj vote of the 4 maps
        logits = sliding_window_inference(x, (512, 512), x.shape[0], self.model, mode='gaussian')
        loss = self.execute_loss(logits, y, val=True)

        preds = torch.nn.Softmax(dim=1)(logits.as_tensor())
        preds = AsDiscrete(threshold=0.7)(preds).as_tensor()

        mean_dice_score, mean_agreement_score = self.execute_metrics(preds, y, val=True)

        self.validation_dice_score.append(mean_dice_score)
        self.validation_loss.append(loss)
        self.validation_agreement.append(mean_agreement_score)  # There is only 4 majority vote seg in y

        values = {"val_loss": loss.item(), "val_dice": mean_dice_score.item(),
                 "perc_score": mean_agreement_score.item(),}

        last_index = self.df_all_val_data.index[-1]

        self.df_all_val_data = pd.concat((self.df_all_val_data, pd.DataFrame(values, index=[last_index+1])))

        self.log('val_loss', loss.item(), prog_bar=True, batch_size=x.shape[0])
        self.log('val_dice', mean_dice_score.item(), prog_bar=True, batch_size=x.shape[0])


    def test_step(self, batch, batch_idx):
        x, y = batch['im_pth'], [batch['Maps1']]  # note that each maps is a maj vote of the 4 maps
        logits = sliding_window_inference(x, (512, 512), x.shape[0], self.model, mode='gaussian')
        loss = self.execute_loss(logits, y, val=True)
        name = batch["save_pth"]
        save_path = r"D:\MEMO_DATA\Data\Preds\\" + name + ".pt"
        preds = torch.nn.Softmax(dim=1)(logits.as_tensor())
        torch.save(preds, save_path)
        preds = AsDiscrete(threshold=0.7)(preds).as_tensor()

        mean_dice_score, mean_agreement_score = self.execute_metrics(preds, y, val=True)

        self.test_dice_score.append(mean_dice_score)
        self.test_loss.append(loss)
        self.test_agreement.append(mean_agreement_score)  # There is only 4 majority vote seg in y

        values = {"test_loss": loss.item(), "test_dice": mean_dice_score.item(),
                  "perc_score": mean_agreement_score.item(), }

        last_index = self.df_all_test_data.index[-1]

        self.df_all_test_data = pd.concat((self.df_all_test_data, pd.DataFrame(values, index=[last_index + 1])))
        self.log('test_loss', loss.item(), prog_bar=True, batch_size=x.shape[0])