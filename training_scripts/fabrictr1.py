# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os

os.environ["USE_TORCH"] = "1"
import warnings
import datetime
import hashlib
import logging
import multiprocessing as mp
import time
from typing import TypeVar, Optional, Iterator
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union, Tuple, Dict, Any, Callable
import numpy as np
import psutil
import torch
import wandb
from fastprogress.fastprogress import master_bar, progress_bar
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR, OneCycleLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms import ColorJitter, Compose, Normalize
import cv2
from doctr.datasets import DetectionDataset
# from doctr.utils.geometry import get_img_shape
import pathlib as Path
from tqdm.auto import tqdm
from doctr import transforms as T
from doctr.datasets import DetectionDataset
from doctr.models import detection
from doctr.utils.metrics import LocalizationConfusion
from torchvision.utils import save_image
# from utils import plot_recorder
import matplotlib
import matplotlib.pyplot as plt
import copy
import math
from torch.distributed import ReduceOp
torch.cuda.empty_cache()

#
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# import albumentations as A
import pdb
import lightning as L
# pdb.set_trace()

# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# import PIL.Image
# PIL.Image.MAX_IMAGE_PIXELS = 933120000

def convert_to_abs_coords(targets, img_shape):
    height, width = img_shape[-2:]
    for idx, t in enumerate(targets):
        targets[idx]["boxes"][:, 0::2] = (t["boxes"][:, 0::2] * width).round()
        targets[idx]["boxes"][:, 1::2] = (t["boxes"][:, 1::2] * height).round()

    targets = [
        {
            "boxes": torch.from_numpy(t["boxes"]).to(dtype=torch.float32),
            "labels": torch.tensor(t["labels"]).to(dtype=torch.long),
        }
        for t in targets
    ]

    return targets

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    # return
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


__all__ = ["DistributedSampler", ]

T_co = TypeVar('T_co', covariant=True)


class Dataset(Generic[T_co]):
    
    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])


class Sampler(Generic[T_co]):
    
    def __init__(self, data_source: Optional[Sized] = None) -> None:
        if data_source is not None:
            import warnings

            warnings.warn("`data_source` argument is not used and will be removed in 2.2.0."
                          "You may still have custom implementation that utilizes it.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    
class MixedDistributedSampler(Sampler[T_co]):
    
    def __init__(self, dataset1: Dataset, dataset2: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
        #     # Split to nearest available length that is evenly divisible.
        #     # This is to ensure each rank receives the same amount of data when
        #     # using this Sampler.
        #     self.num_samples = math.ceil(
        #         (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
        #     )
        # else:
        #     self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        total_length = len(self.dataset1) + len(self.dataset2)
        if self.drop_last and total_length % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (total_length - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(total_length / self.num_replicas)

        self.num_samples1 = int(self.num_samples * 0.6)
        self.num_samples2 = self.num_samples - self.num_samples1    
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    # def __iter__(self) -> Iterator[T_co]:

    #     indices1 = list(range(len(self.dataset1)))[:self.num_samples1]
    #     indices2 = list(range(len(self.dataset2)))[:self.num_samples2]

    #     # Combine the indices
    #     indices = indices1 + indices2

    #     if self.shuffle:
    #         # deterministically shuffle based on epoch and seed
    #         g = torch.Generator()
    #         g.manual_seed(self.seed + self.epoch)
    #         indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
    #     # else:
    #     #     indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

    #     if not self.drop_last:
    #         # add extra samples to make it evenly divisible
    #         padding_size = self.total_size - len(indices)
    #         if padding_size <= len(indices):
    #             indices += indices[:padding_size]
    #         else:
    #             indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
    #     else:
    #         # remove tail of data to make it evenly divisible.
    #         indices = indices[:self.total_size]
    #     assert len(indices) == self.total_size

    #     # subsample
    #     indices = indices[self.rank:self.total_size:self.num_replicas]
    #     assert len(indices) == self.num_samples

    #     return iter(indices)


    def __iter__(self) -> Iterator[T_co]:
        # Generate indices for both datasets
        indices1 = list(range(len(self.dataset1)))[:self.num_samples1]
        indices2 = list(range(len(self.dataset2)))[:self.num_samples2]

        # Combine the indices
        indices = indices1 + indices2

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # Collect samples from both datasets according to the combined indices
        samples = []
        for idx in indices:
            if idx < self.num_samples1:
                samples.append(self.dataset1[idx])
            else:
                samples.append(self.dataset2[idx - self.num_samples1])

        # Return an iterator of the samples
        return iter(samples)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

class MixedRandomSampler(Sampler):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        # self.ratio = ratio

        self.num_samples = max(len(dataset1), len(dataset2))
        self.dataset1_indices = torch.randperm(len(dataset1)).tolist()
        self.dataset2_indices = torch.randperm(len(dataset2)).tolist()

    def __iter__(self):
        num_samples1 = int(self.num_samples * 0.3)
        num_samples2 = self.num_samples - num_samples1

        # Repeat indices if necessary
        dataset1_indices = self.dataset1_indices * (num_samples1 // len(self.dataset1_indices)) + \
                           self.dataset1_indices[:num_samples1 % len(self.dataset1_indices)]
        dataset2_indices = self.dataset2_indices * (num_samples2 // len(self.dataset2_indices)) + \
                           self.dataset2_indices[:num_samples2 % len(self.dataset2_indices)]

        indices = dataset1_indices + dataset2_indices
        return iter(indices)

    def __len__(self):
        return self.num_samples



###########################################################################################
def crop_image_without_cutting_boxes2(image,boxes,probability=0):
    # image = cv2.imread(image)
    image = np.array(image)
    image = np.transpose(image, (1, 2, 0))
    boxes = np.array(boxes)
    if np.random.rand() < probability:
        return image, boxes
    # Step 1: Load the image and its corresponding bounding boxes
    # This is assumed to be done before this function is called

    # Step 2: Calculate the minimum bounding rectangle that contains all bounding boxes
    x_min = np.min(boxes[:, 0])
    y_min = np.min(boxes[:, 1])
    x_max = np.max(boxes[:, 2])
    y_max = np.max(boxes[:, 3])

    crop_x_min = x_min  # Add some padding
    crop_y_min = max(0, y_min - 10)  # Add some padding
    crop_x_max = x_max  # Add some padding
    crop_y_max = min(image.shape[0], y_max + 10)  # Add some padding

    crop_ratio = np.random.uniform(0.08, 0.3)
    height, width, _ = image.shape
    # crop_width = int(width * crop_ratio)
    # crop_height = int(height*(1-crop_ratio))
    crop_width = int(width)
    crop_height = int(height*(crop_ratio))
    # Check for negative or zero sizes
    if crop_width <= 0 or crop_height <= 0:
        print("Invalid crop dimensions. Skipping crop.")
        return image, boxes

    transform = A.Compose([
        A.RandomCrop(width=crop_width, height=crop_height, always_apply=True),
        A.BBoxSafeRandomCrop(erosion_rate = 0.2,always_apply=True)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    print("Original Bounding Boxes:", len(boxes))
    print("Original Image Shape:", image.shape)
    transformed = transform(image=image, bboxes=boxes, labels=np.ones(len(boxes)))
    
    print("Cropped Image Shape:", transformed['image'].shape)
    print("Adjusted Bounding Boxes:", len(transformed['bboxes']))

    image = np.transpose(transformed['image'], (2, 0, 1))

    # Step 5: Return the cropped image and the adjusted bounding boxes
    return transformed['image'], transformed['bboxes']


class CustomDetectionDataset(DetectionDataset):
    """Custom implementation of a text detection dataset with an additional transform.

    Args:
        img_folder: folder with all the images of the dataset
        label_path: path to the annotations of each image
        use_polygons: whether polygons should be considered as rotated bounding box (instead of straight ones)
        custom_transform: a custom transform function to be applied to each sample
        **kwargs: keyword arguments from `AbstractDataset`.
    """

    def __init__(
        self,
        img_folder: str,
        label_path: str,
        use_polygons: bool = False,
        custom_transform: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(img_folder, label_path, use_polygons, **kwargs)
        self.custom_transform = custom_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # Use parent class method to get the image and target
        img, target = super().__getitem__(index)

        # Apply the custom transform if provided
        if self.custom_transform is not None:
            img, target = self.custom_transform(img, target)

        return img, target

#######$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#########################$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import pickle
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_samples(images, targets: List[Dict[str, np.ndarray]]) -> None:
    # Unnormalize image
    nb_samples = min(len(images), 4)
    _, axes = plt.subplots(2, nb_samples, figsize=(20, 5))
    for idx in range(nb_samples):
        img = (255 * images[idx].numpy()).round().clip(0, 255).astype(np.uint8)
        if img.shape[0] == 3 and img.shape[2] != 3:
            img = img.transpose(1, 2, 0)

        target = np.zeros(img.shape[:2], np.uint8)
        tgts = targets[idx].copy()
        for boxes in tgts.values():
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * img.shape[1]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * img.shape[0]
            boxes[:, :4] = boxes[:, :4].round().astype(int)

            for box in boxes:
                if boxes.ndim == 3:
                    cv2.fillPoly(target, [np.int0(box)], 1)
                else:
                    target[int(box[1]) : int(box[3]) + 1, int(box[0]) : int(box[2]) + 1] = 1
        if nb_samples > 1:
            axes[0][idx].imshow(img)
            axes[1][idx].imshow(target.astype(bool))
        else:
            axes[0].imshow(img)
            axes[1].imshow(target.astype(bool))

    # Disable axis
    for ax in axes.ravel():
        ax.axis("off")
    plt.show()


def plot_recorder(lr_recorder, loss_recorder, beta: float = 0.95, **kwargs) -> None:
    """Display the results of the LR grid search.
    Adapted from https://github.com/frgfm/Holocron/blob/master/holocron/trainer/core.py

    Args:
    ----
        lr_recorder: list of LR values
        loss_recorder: list of loss values
        beta (float, optional): smoothing factor
        **kwargs: keyword arguments from `matplotlib.pyplot.show`
    """
    if len(lr_recorder) != len(loss_recorder) or len(lr_recorder) == 0:
        raise AssertionError("Both `lr_recorder` and `loss_recorder` should have the same length")

    # Exp moving average of loss
    smoothed_losses = []
    avg_loss = 0.0
    for idx, loss in enumerate(loss_recorder):
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_losses.append(avg_loss / (1 - beta ** (idx + 1)))

    # Properly rescale Y-axis
    data_slice = slice(
        min(len(loss_recorder) // 10, 10),
        # -min(len(loss_recorder) // 20, 5) if len(loss_recorder) >= 20 else len(loss_recorder)
        len(loss_recorder),
    )
    vals = np.array(smoothed_losses[data_slice])
    min_idx = vals.argmin()
    max_val = vals.max() if min_idx is None else vals[: min_idx + 1].max()  # type: ignore[misc]
    delta = max_val - vals[min_idx]

    plt.plot(lr_recorder[data_slice], smoothed_losses[data_slice])
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Training loss")
    plt.ylim(vals[min_idx] - 0.1 * delta, max_val + 0.2 * delta)
    plt.grid(True, linestyle="--", axis="x")
    plt.show(**kwargs)


def load_backbone(model, weights_path):
    pretrained_backbone_weights = pickle.load(open(weights_path, "rb"))
    model.feat_extractor.set_weights(pretrained_backbone_weights[0])
    model.fpn.set_weights(pretrained_backbone_weights[1])
    return model


class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def fit_one_epoch( model, train_loader, batch_transforms, optimizer, scheduler, mb, amp=True):
    if amp:
        scaler = torch.cuda.amp.GradScaler()

    model.train()
    # Iterate over the batches of the dataset
    pbar = tqdm(train_loader, position=1)
    return_loss, batch_cnt = 0, 0
    for images, targets in pbar:
        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)

        optimizer.zero_grad()
        if amp:
            # print("Using AMP")
            with torch.cuda.amp.autocast():
                train_loss = model(images, targets)["loss"]
                # print("Train loss")
            scaler.scale(train_loss).backward()
            # print("Backward")
            # Gradient clipping
            scaler.unscale_(optimizer)
            # print("Unscale")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # print("Clip")
            # Update the params
            scaler.step(optimizer)
            # print("Step")
            scaler.update()
            # print("Update")
        else:
            # print("Not using AMP")
            train_loss = model(images, targets)["loss"]
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        scheduler.step()
        # print("Scheduler step")

        pbar.set_description(f"Training loss: {train_loss.item():.6}")
        return_loss += train_loss.item()
        batch_cnt += 1
    return_loss /= batch_cnt
    return return_loss

# def fit_one_epoch(fabric2,model, train_loader, batch_transforms, optimizer, scheduler, mb):

#     scaler = torch.cuda.amp.GradientScaler()
#     fabric = fabric2
#     model.train()
#     # Iterate over the batches of the dataset
#     return_loss, batch_cnt = 0,0
#     # for images, targets in progress_bar(train_loader, parent=mb):
#     pbar = tqdm(train_loader, position=1)
    
#     for batch_idx, batch in enumerate(pbar):
#         images, targets = batch
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # device = torch.device("cuda:0")
#         if torch.cuda.is_available():
#             images = images.to(device)
#         images = batch_transforms(images)

#         optimizer.zero_grad()
        
#         is_last_batch = (batch_idx + 1) % 8 == 0

#         # train_loss = model(images, targets)["loss"]
#         # fabric.backward(train_loss)
#         # with fabric.no_backward_sync(model, enabled=not is_last_batch):
#         #     train_loss = model(images, targets)["loss"]
#         #     fabric.backward(train_loss)
#         train_loss = model(images, targets)["loss"]
#         train_loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
#         optimizer.step()
#         scheduler.step()

#         #master bar
#         # mb.child.comment = f"Training loss: {train_loss.item():.6}"
        
        
#     #     return_loss += train_loss.item()
#     #     batch_cnt += 1
   
#     # return_loss /= batch_cnt 
#     # return return_loss
#     return train_loss.item()



# def fit_one_epoch(fabric2, model, train_loader, batch_transforms, optimizer, scheduler, mb):
#     fabric = fabric2
#     model.train()
    
#     return_loss, batch_cnt = 0, 0
    
#     pbar = tqdm(train_loader, position=1)
    
#     for batch_idx, batch in enumerate(pbar):
#         images, targets = batch
        
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if torch.cuda.is_available():
#             images = images.to(device)  # Ensure device placement
        
#         images = batch_transforms(images)
        
#         is_last_batch = (batch_idx + 1) % 8 == 0  # Check if it's the last batch for accumulation
        
#         with fabric.no_backward_sync(model, enabled=not is_last_batch):
#             train_loss = model(images, targets)["loss"]
#             fabric.backward(train_loss)

#         return_loss += train_loss.item()
#         batch_cnt += 1

#         if is_last_batch:
#             fabric.clip_gradients(model, optimizer, max_norm=2.0, error_if_nonfinite=False)
#             optimizer.step()
#             optimizer.zero_grad()
#             scheduler.step()
        
#         pbar.set_description(f"Training loss: {train_loss.item():.6}")

#     return_loss /= batch_cnt 
#     return return_loss



# def fit_one_epoch(fabric2, model, train_loader, batch_transforms, optimizer, scheduler, mb):
#     st1 = time.time()
#     fabric = fabric2
    
#     model.train()
    
#     # Iterate over the batches of the dataset
#     return_loss, batch_cnt = 0,0
#     # for images, targets in progress_bar(train_loader, parent=mb):
    
#     pbar = tqdm(train_loader, position=1)
#     # for images, targets in train_loader:
#     # for batch in pbar:

#     for batch_idx, batch in enumerate(pbar):
#         # try:
#         st2 = time.time()
#         images, targets = batch
#         start_time = time.time()
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # device = torch.device("cuda:0")
#         # if torch.cuda.is_available():
#         #     images = images.to(device)

#         images = batch_transforms(images)
#         # optimizer.zero_grad()
        
#         is_accumulating = batch_idx % 8!= 0

#         with fabric.no_backward_sync(model,enabled=is_accumulating):
#             train_loss = model(images, targets)["loss"]
#             # train_loss.backward()
#             fabric.backward(train_loss)

#             # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
#             # fabric.clip_gradients(model, optimizer, max_norm=2.0, error_if_nonfinite=False)
#         # except Exception as e:
#         #     print(e)
#         #     continue

#         if not is_accumulating:
#             fabric.clip_gradients(model, optimizer, max_norm=2.0, error_if_nonfinite=False)
#             optimizer.step()
#             optimizer.zero_grad()
#             scheduler.step()
        
#         #master bar
#         # mb.child.comment = f"Training loss: {train_loss.item():.6}"
#         pbar.set_description(f"Training loss: {train_loss.item():.6}")
#         return_loss += train_loss.item()
#         batch_cnt += 1
#         print("Time taken for batch ",batch_idx," is ",time.time()-st2,"s")
   
#     return_loss /= batch_cnt 
    
#     return return_loss


@torch.no_grad()
def evaluate(model, val_loader, batch_transforms, val_metric, amp=False):
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    for images, targets in val_loader:
        # if torch.cuda.is_available():
        #     images = images.cuda()
        images = batch_transforms(images)
        out = model(images, targets, return_preds=True)
        # Compute metric
        loc_preds = out["preds"]
        # print("Loc preds",loc_preds)
        # print("Targets",targets)
        # for boxes_gt, boxes_pred in zip(targets, loc_preds):
        #     print("GT",boxes_gt) 
        #     print("PRED",boxes_pred)
        #     # Convert pred to boxes [xmin, ymin, xmax, ymax]  N, 4, 2 --> N, 4
        #     boxes_pred = np.concatenate((boxes_pred.min(axis=1), boxes_pred.max(axis=1)), axis=-1)
        #     val_metric.update(gts=boxes_gt, preds=boxes_pred[:, :4])
        for target, loc_pred in zip(targets, loc_preds):
            for boxes_gt, boxes_pred in zip(target.values(), loc_pred.values()):
                if args.rotation and args.eval_straight:
                    # Convert pred to boxes [xmin, ymin, xmax, ymax]  N, 4, 2 --> N, 4
                    boxes_pred = np.concatenate((boxes_pred.min(axis=1), boxes_pred.max(axis=1)), axis=-1)
                val_metric.update(gts=boxes_gt, preds=boxes_pred[:, :4])
        val_loss += out["loss"].item()
        batch_cnt += 1

        if(batch_cnt%100 == 0):
            print("Running eval for batch_cnt ",batch_cnt)

    val_loss /= batch_cnt
    recall, precision, mean_iou = val_metric.summary()
    return val_loss, recall, precision, mean_iou


def main(args):
    # fabric = L.Fabric(accelerator="cuda", devices=3, strategy='ddp', precision='16-mixed')

    # fabric = L.Fabric(accelerator="cuda", devices=1)
    # fabric.launch()
    print(args)

    torch.backends.cudnn.benchmark = True
    system_available_memory = int(psutil.virtual_memory().available / 1024**3)
    logging.info(f"System memory available: {system_available_memory}GB")

    st = time.time()
    
    print("Loading val dataset1")
    val_set_d1 = DetectionDataset(
        img_folder=os.path.join(args.val_path_d1, "images"),
        label_path=os.path.join(args.val_path_d1, "labels.json"),
        sample_transforms=T.SampleCompose(
            (
                [T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True)]
                if not args.rotation or args.eval_straight
                else []
            )
            + (
                [
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True),  # This does not pad
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation and not args.eval_straight
                else []
            )
        ),
        use_polygons= False,
    )
    
    print('Loading val dataset2')
    val_set_d2 = DetectionDataset(
        img_folder=os.path.join(args.val_path_d2, "images"),
        label_path=os.path.join(args.val_path_d2, "labels.json"),
        sample_transforms=T.SampleCompose(
            (
                [T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True)]
                if not args.rotation or args.eval_straight
                else []
            )
            + (
                [
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True),  # This does not pad
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation and not args.eval_straight
                else []
            )
        ),
        use_polygons= False,
    )

    combined_val_set = torch.utils.data.ConcatDataset([val_set_d1,val_set_d2])

    # val_sampler = torch.utils.data.distributed.DistributedSampler(
    # 	combined_val_set,
    # 	shuffle = True
    # )

    val_loader = DataLoader(
        combined_val_set,
        # val_set_d1,
        batch_size=args.batch_size,
        num_workers=args.workers,
        # sampler=val_sampler,
        pin_memory=torch.cuda.is_available(),
        collate_fn=val_set_d1.collate_fn,
        drop_last=True
    )
    # val_loader = DataLoader(
    #     combined_val_set,
    #     batch_size=args.batch_size,
    #     num_workers=args.workers,
    #     shuffle = True
    # )

    # print(f"Validation set loaded in {time.time() - st:.4}s ({len(combined_val_set)} samples in " f"{len(val_loader)} batches)")

    with open(os.path.join(args.val_path_d1, "labels.json"), "rb") as f:
        val_hash_d1 = hashlib.sha256(f.read()).hexdigest()
    
    with open(os.path.join(args.val_path_d2, "labels.json"), "rb") as f:
        val_hash_d2 = hashlib.sha256(f.read()).hexdigest()

    print("loading test dataset1")
    test_set_d1 = DetectionDataset(
        img_folder=os.path.join(args.test_path_d1, "images"),
        label_path=os.path.join(args.test_path_d1, "labels.json"),
        sample_transforms=T.SampleCompose(
            (
                [T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True)]
                if not args.rotation or args.eval_straight
                else []
            )
            + (
                [
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True),  # This does not pad
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation and not args.eval_straight
                else []
            )
        ),
        use_polygons= False,
    )

    print("loading test dataset2")
    # test_set_d2 = DetectionDataset(
    #     img_folder=os.path.join(args.test_path_d2, "images"),
    #     label_path=os.path.join(args.test_path_d2, "labels.json"),
    #     sample_transforms=T.SampleCompose(
    #         (
    #             [T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True)]
    #             if not args.rotation or args.eval_straight
    #             else []
    #         )
    #         + (
    #             [
    #                 T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True),  # This does not pad
    #                 T.RandomRotate(90, expand=True),
    #                 T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
    #             ]
    #             if args.rotation and not args.eval_straight
    #             else []
    #         )
    #     ),
    #     # use_polygons= False,
    # )
        
    # combined_test_set = torch.utils.data.ConcatDataset([test_set_d1,test_set_d2])

    # test_sampler = torch.utils.data.distributed.DistributedSampler(
    # 	combined_test_set,
    #     # test_set_d1,
    # 	shuffle = True
    # )

    test_loader = DataLoader(
        test_set_d1,
        batch_size=args.batch_size,
        num_workers=args.workers,
        # sampler=test_sampler,
        pin_memory=torch.cuda.is_available(),
        collate_fn=test_set_d1.collate_fn,
        drop_last=True
    )

    # print(f"Test set loaded in {time.time() - st:.4}s ({len(test_set_d1)} samples in " f"{len(test_loader)} batches)")

    with open(os.path.join(args.test_path_d1, "labels.json"), "rb") as f:
        test_hash_d1 = hashlib.sha256(f.read()).hexdigest()

    # with open(os.path.join(args.test_path_d2, "labels.json"), "rb") as f:
    #     test_hash_d2 = hashlib.sha256(f.read()).hexdigest()

    batch_transforms = Normalize(mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287))

    # local_rank = rank
    # device = torch.device("cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu")

    # model = detection.__dict__["db_resnet50"](pretrained=args.pretrained, assume_straight_pages= True)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    model = detection.__dict__["db_resnet50"](pretrained=args.pretrained, assume_straight_pages= True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Backbone freezing
    if args.freeze_backbone:
        model.feat_extractor.requires_grad_(False) #False->True to unfreeze all the layers
    
    # GPU
    torch.cuda.set_device(local_rank)
    model = model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if isinstance(args.resume, str):
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint)

    # Metrics
    val_metric = LocalizationConfusion(
        use_polygons= False,
        mask_shape=(args.input_size, args.input_size),
        use_broadcasting=True if system_available_memory > 62 else False,
    )
    test_metric = LocalizationConfusion(
        use_polygons= False,
        mask_shape=(args.input_size, args.input_size),
        use_broadcasting=True if system_available_memory > 62 else False,
    )

    if args.test_only:
        print("Running evaluation")
        val_loss, recall, precision, mean_iou = evaluate(model, val_loader, batch_transforms, val_metric, amp=args.amp)
        print(
            f"Validation loss: {val_loss:.6} (Recall: {recall:.2%} | Precision: {precision:.2%} | "
            f"Mean IoU: {mean_iou:.2%})"
        )
        return

    st = time.time()

    print("Loading train dataset1")
    easy_train_set_d1 = DetectionDataset(
        img_folder=os.path.join(args.easy_train_path_d1, "images"),
        label_path=os.path.join(args.easy_train_path_d1, "labels.json"),
        img_transforms=Compose(
            [
                # Augmentations
                T.RandomApply(T.ColorInversion(), 0.1),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            ]
        ),
        sample_transforms=T.SampleCompose(
            (
                [T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True)]
                if not args.rotation
                else []
            )
            + (
                [
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True),
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation
                else []
            )
        ),
        use_polygons= False,
    )

    medium_train_set_d1 = DetectionDataset(
        img_folder=os.path.join(args.medium_train_path_d1, "images"),
        label_path=os.path.join(args.medium_train_path_d1, "labels.json"),
        img_transforms=Compose(
            [
                # Augmentations
                T.RandomApply(T.ColorInversion(), 0.1),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            ]
        ),
        sample_transforms=T.SampleCompose(
            (
                [T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True)]
                if not args.rotation
                else []
            )
            + (
                [
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True),
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation
                else []
            )
        ),
        use_polygons=False,
    )

    hard_train_set_d1 = DetectionDataset(
        img_folder=os.path.join(args.hard_train_path_d1, "images"),
        label_path=os.path.join(args.hard_train_path_d1, "labels.json"),
        img_transforms=Compose(
            [
                # Augmentations
                T.RandomApply(T.ColorInversion(), 0.1),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            ]
        ),
        sample_transforms=T.SampleCompose(
            (
                [T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True)]
                if not args.rotation
                else []
            )
            + (
                [
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True),
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation
                else []
            )
        ),
        use_polygons=False,
    )

    print("Balancing Dataset1")
    e_d1 = len(easy_train_set_d1)
    m_d1 = len(medium_train_set_d1)
    h_d1 = len(hard_train_set_d1)
    N_d1 = e_d1+m_d1+h_d1
    # newmedium_d1 = e_d1* (N_d1-m_d1)/(N_d1-e_d1)
    newhard_d1 = e_d1* (N_d1-h_d1)/(N_d1-e_d1)
    # print("size of Easy, Newmedium, Newhard, hard, medium, Total docs (all for train set): ",e_d1,newmedium_d1,newhard_d1,h_d1,m_d1,N_d1)
    print("size of Easy, Newhard, hard, medium, Total docs (all for train set): ",e_d1,newhard_d1,h_d1,m_d1,N_d1)
    
    new_hard_train_set_d1 = copy.deepcopy(hard_train_set_d1)
    while(len(new_hard_train_set_d1) + h_d1 < newhard_d1):
        new_hard_train_set_d1 = torch.utils.data.ConcatDataset([new_hard_train_set_d1, hard_train_set_d1])
    tail_end_size_d1 = math.floor(newhard_d1 - len(new_hard_train_set_d1))
    tail_end_d1, extra_d1 = torch.utils.data.random_split(hard_train_set_d1,[tail_end_size_d1,h_d1 - tail_end_size_d1])
    new_hard_train_set_d1 = torch.utils.data.ConcatDataset([new_hard_train_set_d1, tail_end_d1])
    
    # new_medium_train_set_d1 = copy.deepcopy(medium_train_set_d1)
    # while(len(new_medium_train_set_d1) + h_d1 < newmedium_d1):
    #     new_medium_train_set_d1 = torch.utils.data.ConcatDataset([new_medium_train_set_d1, medium_train_set_d1])
    # tail_end_size_d1 = math.floor(newmedium_d1 - len(new_medium_train_set_d1))
    # tail_end_d1, extra_d1 = torch.utils.data.random_split(medium_train_set_d1,[tail_end_size_d1,m_d1 - tail_end_size_d1])
    # new_medium_train_set_d1 = torch.utils.data.ConcatDataset([new_medium_train_set_d1, tail_end_d1])
    
    # print("length of the new medium and hard train set: ",len(new_medium_train_set_d1),len(new_hard_train_set_d1))
    print("length of the new hard train set: ",len(new_hard_train_set_d1))

    # train_set_d1 = torch.utils.data.ConcatDataset([easy_train_set_d1, new_medium_train_set_d1,new_hard_train_set_d1])
    # train_set_d1 = torch.utils.data.ConcatDataset([easy_train_set_d1, new_hard_train_set_d1])
    train_set_d1 = hard_train_set_d1
    
    # Nnew_d1 = e_d1 + len(new_medium_train_set_d1) + len(new_hard_train_set_d1)
    # M_d1= math.floor(math.log(0.1, ((Nnew_d1-1)/Nnew_d1)))

    Nnew_d1 = e_d1 + len(new_hard_train_set_d1)
    M_d1= math.floor(math.log(0.1, ((Nnew_d1-1)/Nnew_d1)))

    print("Length of new trainset: ", Nnew_d1)


    print("Loading train dataset2")
    easy_train_set_d2 = DetectionDataset(
        img_folder=os.path.join(args.easy_train_path_d2, "images"),
        label_path=os.path.join(args.easy_train_path_d2, "labels.json"),
        img_transforms=Compose(
            [
                # Augmentations
                T.RandomApply(T.ColorInversion(), 0.1),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            ]
        ),
        sample_transforms=T.SampleCompose(
            (
                [T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True)]
                if not args.rotation
                else []
            )
            + (
                [
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True),
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation
                else []
            )
        ),
        use_polygons= False,
    )

    medium_train_set_d2 = DetectionDataset(
        img_folder=os.path.join(args.medium_train_path_d2, "images"),
        label_path=os.path.join(args.medium_train_path_d2, "labels.json"),
        img_transforms=Compose(
            [
                # Augmentations
                T.RandomApply(T.ColorInversion(), 0.1),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            ]
        ),
        sample_transforms=T.SampleCompose(
            (
                [T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True)]
                if not args.rotation
                else []
            )
            + (
                [
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True),
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation
                else []
            )
        ),
        use_polygons=False,
    )

    hard_train_set_d2 = DetectionDataset(
        img_folder=os.path.join(args.hard_train_path_d2, "images"),
        label_path=os.path.join(args.hard_train_path_d2, "labels.json"),
        img_transforms=Compose(
            [
                # Augmentations
                T.RandomApply(T.ColorInversion(), 0.1),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            ]
        ),
        sample_transforms=T.SampleCompose(
            (
                [T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True)]
                if not args.rotation
                else []
            )
            + (
                [
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True),
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation
                else []
            )
        ),
        use_polygons=False,
    )

    print("Balancing Dataset2")
    e_d2 = len(easy_train_set_d2)
    m_d2 = len(medium_train_set_d2)
    h_d2 = len(hard_train_set_d2)
    N_d2 = e_d2+m_d2+h_d2
    newmedium_d2 = e_d2* (N_d2-m_d2)/(N_d2-e_d2)
    newhard_d2 = e_d2* (N_d2-h_d2)/(N_d2-e_d2)
    print("size of Easy, Newmedium, Newhard, hard, medium, Total docs (all for train set): ",e_d2,newmedium_d2,newhard_d2,h_d2,m_d2,N_d2)
    
    new_hard_train_set_d2 = copy.deepcopy(hard_train_set_d2)
    while(len(new_hard_train_set_d2) + h_d2 < newhard_d2):
        new_hard_train_set_d2 = torch.utils.data.ConcatDataset([new_hard_train_set_d2, hard_train_set_d2])
    tail_end_size_d2 = math.floor(newhard_d2 - len(new_hard_train_set_d2))
    tail_end_d2, extra_d2 = torch.utils.data.random_split(hard_train_set_d2,[tail_end_size_d2,h_d2 - tail_end_size_d2])
    new_hard_train_set_d2 = torch.utils.data.ConcatDataset([new_hard_train_set_d2, tail_end_d2])
    
    new_medium_train_set_d2 = copy.deepcopy(medium_train_set_d2)
    while(len(new_medium_train_set_d2) + h_d2 < newmedium_d2):
        new_medium_train_set_d2 = torch.utils.data.ConcatDataset([new_medium_train_set_d2, medium_train_set_d2])
    tail_end_size_d2 = math.floor(newmedium_d2 - len(new_medium_train_set_d2))
    tail_end_d2, extra_d2 = torch.utils.data.random_split(medium_train_set_d2,[tail_end_size_d2,m_d2 - tail_end_size_d2])
    new_medium_train_set_d2 = torch.utils.data.ConcatDataset([new_medium_train_set_d2, tail_end_d2])
    
    print("length of the new medium and hard train set: ",len(new_medium_train_set_d2),len(new_hard_train_set_d2))
    
    train_set_d2 = torch.utils.data.ConcatDataset([easy_train_set_d2, new_medium_train_set_d2,new_hard_train_set_d2])
    
    Nnew_d2 = e_d2 + len(new_medium_train_set_d2) + len(new_hard_train_set_d2)
    M_d2= math.floor(math.log(0.1, ((Nnew_d2-1)/Nnew_d2)))

    print("Length of new trainset: ", Nnew_d2)

    print("#######$####$$$$$$$$###########")

    # print("Balancing both balanced datasets in the ratio 7:3")
    # splitBalanced_new_d2 = Nnew_d2*((Nnew_d1*0.7)/(Nnew_d2*0.3))
    # print("Number of samples to have in dataset2 after balancing for minibatch split",splitBalanced_new_d2)

    # new_train_set_d2 = copy.deepcopy(train_set_d2)
    # while(len(new_train_set_d2) + Nnew_d2 < splitBalanced_new_d2):
    #     new_train_set_d2 = torch.utils.data.ConcatDataset([new_train_set_d2, train_set_d2])

    # tail_end_size = math.floor(splitBalanced_new_d2 - len(new_train_set_d2))
    # tail_end, extra = torch.utils.data.random_split(train_set_d2,[tail_end_size,Nnew_d2 - tail_end_size])
    # new_train_set_d2 = torch.utils.data.ConcatDataset([new_train_set_d2, tail_end])

    # splitBalanced_train_set_d2 = torch.utils.data.ConcatDataset([new_train_set_d2, train_set_d2])  #trainset for dataset2   

    # print("Length of new final combined trainset: ", len(splitBalanced_train_set_d2))
    # print("Combining both datasets after final balancing in the ratio")
    # combined_train_set = torch.utils.data.ConcatDataset([train_set_d1,splitBalanced_train_set_d2])
    combined_train_set = torch.utils.data.ConcatDataset([train_set_d1,train_set_d2])

    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    # 	combined_train_set,
    # 	shuffle = True
    # )

    train_loader = DataLoader(
        combined_train_set,
        # train_set_d2,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        # sampler=RandomSampler(train_set_d1),
        pin_memory=torch.cuda.is_available(),
        collate_fn = val_set_d1.collate_fn
        # collate_fn=combined_train_set.collate_fn,
    )

    print("Combined Final len(train_loader):",len(train_loader))

    # print(f"Train set loaded in {time.time() - st:.4}s ({len(combined_train_set)} samples in " f"{len(train_loader)} batches)")
    
    print("Generating hashes for trainset")
    with open(os.path.join(args.easy_train_path_d1, "labels.json"), "rb") as f:
        easy_train_hash_d1 = hashlib.sha256(f.read()).hexdigest()

    with open(os.path.join(args.medium_train_path_d1, "labels.json"), "rb") as f:
        medium_train_hash_d1 = hashlib.sha256(f.read()).hexdigest()

    with open(os.path.join(args.hard_train_path_d1, "labels.json"), "rb") as f:
        hard_train_hash_d1 = hashlib.sha256(f.read()).hexdigest()


    with open(os.path.join(args.easy_train_path_d2, "labels.json"), "rb") as f:
        easy_train_hash_d2 = hashlib.sha256(f.read()).hexdigest()

    with open(os.path.join(args.medium_train_path_d2, "labels.json"), "rb") as f:
        medium_train_hash_d2 = hashlib.sha256(f.read()).hexdigest()

    with open(os.path.join(args.hard_train_path_d2, "labels.json"), "rb") as f:
        hard_train_hash_d2 = hashlib.sha256(f.read()).hexdigest()

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return

    # Backbone freezing
    # if args.freeze_backbone:
    #     for p in model.feat_extractor.parameters():
    #         p.requires_grad = False

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        args.lr,
        betas=(0.95, 0.99),
        eps=1e-6,
        weight_decay=args.weight_decay,
    )

    # model,optimizer = fabric.setup(model,optimizer)
    # val_loader_fab = fabric.setup_dataloaders(val_loader)
    # test_loader_fab = fabric.setup_dataloaders(test_loader)
    # train_loader_fab = fabric.setup_dataloaders(train_loader)
    print("Fabric setup done")
    # if args.find_lr:
    #     lrs, losses = record_lr(model, train_loader, batch_transforms, optimizer, amp=args.amp)
    #     plot_recorder(lrs, losses)
    #     return
    
    scheduler = CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=args.lr / 25e4)

    # Training monitoring
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"db_resnet50_{current_time}" if args.name is None else args.name
    if isinstance(args.resume, str):
        file_name = os.path.basename(args.resume)
        exp_name = file_name[:file_name.find("_epoch")]
    # W&B
    # if args.wb and dist.get_rank() == 0:
    if args.wb:

        run = wandb.init(
            name=exp_name,
            id=exp_name,
            project="FT-doctrv1",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "batch_size_per_gpu": args.batch_size,
                "architecture": "db_resnet50",
                "input_size": args.input_size,
                "optimizer": "adam",
                "framework": "pytorch",
                "scheduler": "cosine",
                #changed
                # "easy_train_hash_d1": easy_train_hash_d1,
                # "medium_train_hash_d1": medium_train_hash_d1,
                # "hard_train_hash_d1": hard_train_hash_d1,
                
                "easy_train_hash_d2": easy_train_hash_d2,
                "medium_train_hash_d2": medium_train_hash_d2,
                "hard_train_hash_d2": hard_train_hash_d2,


                # "val_hash_d1": val_hash_d1,
                # "test_hash_d1": test_hash_d1,

                "val_hash_d2": val_hash_d2,
                # "test_hash_d2": test_hash_d2,
                "pretrained": args.pretrained,
                "rotation": False
            }, 
            resume="allow"
        )
        last_epoch = wandb.run.summary.get('epoch',0)
        print('Last Epoch:',last_epoch)

    ccc=0
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            ccc+=1
            # print(name)
    print("Number of params:",ccc)  

    min_loss = np.inf
    if args.early_stop:
        early_stopper = EarlyStopper(patience=args.early_stop_epochs, min_delta=args.early_stop_delta)

    mb = master_bar(range(args.epochs))

    total_time = 0
    for epoch in mb:
        print("Entered epoch!!!!!!!!!!!!!!!!!!!!!!!!!!")
        t0 = time.time()
        
        # train_loss = fit_one_epoch(fabric, model, train_loader_fab, batch_transforms, optimizer, scheduler, mb, amp=args.amp)
        train_loss = fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, mb, amp=args.amp)

        dist.barrier()
        # fabric.barrier()

        epoch_time = time.time() - t0
        total_time+=epoch_time
        
        # images_per_sec = torch.tensor(len(train_loader)*args.batch_size/epoch_time).to(device)
        # dist.reduce(images_per_sec, 0)

        # val_loss, recall, precision, mean_iou = evaluate(model, val_loader_fab, batch_transforms, val_metric, amp=args.amp)
        val_loss, recall, precision, mean_iou = evaluate(model, val_loader, batch_transforms, val_metric, amp=args.amp)

        world_size = int(os.environ['WORLD_SIZE'])
        # world_size = fabric.world_size
        train_loss, val_loss, recall, precision, mean_iou = train_loss/world_size, val_loss/world_size, recall/world_size, precision/world_size, mean_iou/world_size
        
        train_loss = torch.tensor(train_loss)
        val_loss = torch.tensor(val_loss)
        recall = torch.tensor(recall)
        precision = torch.tensor(precision)
        mean_iou = torch.tensor(mean_iou)

        # print("b1  -  ",val_loss)
        train_loss = train_loss.cuda()
        val_loss = val_loss.cuda()
        recall = recall.cuda()
        precision = precision.cuda()
        mean_iou = mean_iou.cuda()

        # train_loss = fabric.to_device(train_loss)
        # val_loss = fabric.to_device(val_loss)
        # recall = fabric.to_device(recall)
        # precision = fabric.to_device(precision)
        # mean_iou = fabric.to_device(mean_iou)
        
        # print("b2  -  ",val_loss)

        dist.all_reduce(train_loss, op=ReduceOp.SUM)
        dist.all_reduce(val_loss, op=ReduceOp.SUM)
        dist.all_reduce(recall, op=ReduceOp.SUM)
        dist.all_reduce(precision, op=ReduceOp.SUM)
        dist.all_reduce(mean_iou, op=ReduceOp.SUM)

        # reduced_train_loss = fabric.all_reduce(train_loss, reduce_op="sum")
        # reduced_val_loss = fabric.all_reduce(val_loss, reduce_op="sum")
        # reduced_recall = fabric.all_reduce(recall, reduce_op="sum")
        # reduced_precision = fabric.all_reduce(precision, reduce_op="sum")
        # reduced_mean_iou = fabric.all_reduce(mean_iou, reduce_op="sum")

        print("Train Loss: ",train_loss)
        print("Val Loss: ",val_loss)
        print("Recall: ",recall)
        print("Precision: ",precision)
        print("Mean IoU: ",mean_iou)

        # print("Train Loss: ",reduced_train_loss)
        # print("Val Loss: ",reduced_val_loss)
        # print("Recall: ",reduced_recall)
        # print("Precision: ",reduced_precision)
        # print("Mean IoU: ",reduced_mean_iou)


        if dist.get_rank() == 0:
            # torch.save(model.state_dict(), f"./models/{exp_name}_epoch{epoch}.pt")
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'scheduler': scheduler.state_dict()
                        }, f"/home2/sreevatsa/models/{exp_name}_epoch{epoch}.pt")
            
            if val_loss < min_loss :
                print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
                min_loss = val_loss
                counter = 0
            else:
                counter+=1

            #check if training should be stopped
            # if counter == patience:
            #     print("Validation loss hasn't improved in", patience, "epochs. Early stopping.")
            #     break

            log_msg = f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} "
            if any(val is None for val in (recall, precision, mean_iou)):
                log_msg += "(Undefined metric value, caused by empty GTs or predictions)"
            else:
                log_msg += f"(Recall: {recall:.2%} | Precision: {precision:.2%} | Mean IoU: {mean_iou:.2%})"
            mb.write(log_msg)
            
            # if args.early_stop and early_stopper.early_stop(reduced_val_loss):
            if args.early_stop and early_stopper.early_stop(val_loss):
                print("Training halted early due to reaching patience limit.")
                break
            # W&B
            if args.wb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "average_recall": recall,
                        "average_precision": precision,
                        "average_mean_iou": mean_iou,
                        "epoch": epoch
                    }
                )

    if args.wb and dist.get_rank() == 0:
        run.finish()

        # state = {"model":model, "optimizer":optimizer, "epoch":epoch, "scheduler":scheduler}

        # if reduced_val_loss < min_loss:
        #     print(f"Validation loss decreased {min_loss:.6} --> {reduced_val_loss:.6}: saving state...")
        #     fabric.save(f"./models/{exp_name}_epoch{epoch}.pt", state)
        #     min_loss = reduced_val_loss

        # log_msg = f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} "
        # if any(val is None for val in (recall, precision, mean_iou)):
        #     log_msg += "(Undefined metric value, caused by empty GTs or predictions)"
        # else:
        #     log_msg += f"(Recall: {recall:.2%} | Precision: {precision:.2%} | Mean IoU: {mean_iou:.2%})"
        # print(log_msg)

        # if args.early_stop and early_stopper.early_stop(reduced_val_loss):
        #     print("Training halted early due to reaching patience limit.")
        #     break
        


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR training script for text detection (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    #changed
    parser.add_argument("--easy_train_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset4/train/Easy", help="path to training data folder")
    parser.add_argument("--medium_train_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset4/train/Medium", help="path to training data folder")
    parser.add_argument("--hard_train_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset4/train/Hard", help="path to training data folder")
    parser.add_argument("--val_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset4/val", help="path to validation data folder")
    parser.add_argument("--test_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset4/test", help="path to test data folder")
    
    # parser.add_argument("--easy_train_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset1/extract/scratch/abhaynew/newfolder/train/Easy", help="path to training data folder")
    # parser.add_argument("--medium_train_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset1/extract/scratch/abhaynew/newfolder/train/Medium", help="path to training data folder")
    # parser.add_argument("--hard_train_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset1/extract/scratch/abhaynew/newfolder/train/Hard", help="path to training data folder")
    # parser.add_argument("--val_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset1/extract/scratch/abhaynew/newfolder/val", help="path to validation data folder")
    # parser.add_argument("--test_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset1/extract/scratch/abhaynew/newfolder/test", help="path to test data folder")
    
    #dataset2
    parser.add_argument("--easy_train_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset5_8124/train/Easy", help="path to training data folder")
    parser.add_argument("--medium_train_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset5_8124/train/Medium", help="path to training data folder")
    parser.add_argument("--hard_train_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset5_8124/train/Hard", help="path to training data folder")
    parser.add_argument("--val_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset5_8124/val", help="path to validation data folder")
    parser.add_argument("--test_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset5_8124/test", help="path to test data folder")
    


    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="batch size for training")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("--input_size", type=int, default=512, help="model input size, H = W")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam)")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("-j", "--workers", type=int, default=0, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument("--rotation", type=str, default=False, help="Accept rotated images")
    parser.add_argument("--eval_straight", type=str, default=True, help="process straight images")
    parser.add_argument("--find_lr", type=str, default=True, help="process straight images")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Run the validation loop")
    parser.add_argument(
        "--freeze-backbone", dest="freeze_backbone", action="store_true", help="freeze model backbone for fine-tuning"
    )
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping")
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    parser.add_argument("--early-stop-epochs", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--early-stop-delta", type=float, default=0.01, help="Minimum Delta for early stopping")
    parser.add_argument(
        "--show-samples", dest="show_samples", action="store_true", help="Display unormalized training samples"
    )
    parser.add_argument("--wb", dest="wb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--push-to-hub", dest="push_to_hub", action="store_true", help="Push to Huggingface Hub")
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Load pretrained parameters before starting the training",
    )
    # DDP configs:
    # parser.add_argument('--world-size', default=-1, type=int, 
    #                     help='number of nodes for distributed training')
    # parser.add_argument('--rank', default=-1, type=int, 
    #                     help='node rank for distributed training')
    # parser.add_argument('--dist-url', default='env://', type=str, 
    #                     help='url used to set up distributed training')
    # parser.add_argument('--dist-backend', default='nccl', type=str, 
    #                     help='distributed backend')
    # parser.add_argument('--local_rank', default=0, type=int, 
    #                     help='local rank for distributed training')
    args = parser.parse_args()

    return args

#
# def init_distributed():

#    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
#     dist_url = "env://" # default

#     # only works with torch.distributed.launch // torch.run
#     rank = int(os.environ["RANK"])
#     world_size = int(os.environ['WORLD_SIZE'])
#     local_rank = int(os.environ['LOCAL_RANK'])

#     dist.init_process_group(
#             backend="nccl",
#             init_method=dist_url,
#             world_size=world_size,
#             rank=rank)

#     # this will make all .cuda() calls work properly
#     torch.cuda.set_device(local_rank)
#     # synchronizes all the threads to reach this point before moving on
#     dist.barrier()
#     setup_for_distributed(rank == 0)

def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    is_master = (rank == 0)

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier(device_ids=[local_rank])
    if is_master:
        print("DDP is working correctly with {} processes.".format(world_size))
    setup_for_distributed(rank == 0)

if __name__ == "__main__":
    args = parse_args()
    #
    init_distributed()
    # init_processes(backend='nccl')
    main(args)
    warnings.filterwarnings("ignore")
    # wandb.finish()



