# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os

os.environ["USE_TORCH"] = "1"

import datetime
import hashlib
import logging
import multiprocessing as mp
import time
from typing import TypeVar, Optional, Iterator
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union, Tuple, Dict, Any
import numpy as np
import psutil
import torch
import wandb
from fastprogress.fastprogress import master_bar, progress_bar
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR, OneCycleLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms import ColorJitter, Compose, Normalize

from doctr import transforms as T
from doctr.datasets import DetectionDataset
from doctr.models import detection
from doctr.utils.metrics import LocalizationConfusion
from torchvision.utils import save_image
# from utils import plot_recorder
import torch
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

import pdb
# pdb.set_trace()

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`. Subclasses could also
    optionally implement :meth:`__getitems__`, for speedup batched samples
    loading. This method accepts list of indices of samples of batch and returns
    list of samples.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs an index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    # def __getitems__(self, indices: List) -> List[T_co]:
    # Not implemented to prevent false-positives in fetcher check in
    # torch.utils.data._utils.fetch._MapDatasetFetcher

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py




class Sampler(Generic[T_co]):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices or lists of indices (batches) of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    Args:
        data_source (Dataset): This argument is not used and will be removed in 2.2.0.
            You may still have custom implementation that utilizes it.

    Example:
        >>> # xdoctest: +SKIP
        >>> class AccedingSequenceLengthSampler(Sampler[int]):
        >>>     def __init__(self, data: List[str]) -> None:
        >>>         self.data = data
        >>>
        >>>     def __len__(self) -> int:
        >>>         return len(self.data)
        >>>
        >>>     def __iter__(self) -> Iterator[int]:
        >>>         sizes = torch.tensor([len(x) for x in self.data])
        >>>         yield from torch.argsort(sizes).tolist()
        >>>
        >>> class AccedingSequenceLengthBatchSampler(Sampler[List[int]]):
        >>>     def __init__(self, data: List[str], batch_size: int) -> None:
        >>>         self.data = data
        >>>         self.batch_size = batch_size
        >>>
        >>>     def __len__(self) -> int:
        >>>         return (len(self.data) + self.batch_size - 1) // self.batch_size
        >>>
        >>>     def __iter__(self) -> Iterator[List[int]]:
        >>>         sizes = torch.tensor([len(x) for x in self.data])
        >>>         for batch in torch.chunk(torch.argsort(sizes), len(self)):
        >>>             yield batch.tolist()

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source: Optional[Sized] = None) -> None:
        if data_source is not None:
            import warnings

            warnings.warn("`data_source` argument is not used and will be removed in 2.2.0."
                          "You may still have custom implementation that utilizes it.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #
    # Many times we have an abstract class representing a collection/iterable of
    # data, e.g., `torch.utils.data.Sampler`, with its subclasses optionally
    # implementing a `__len__` method. In such cases, we must make sure to not
    # provide a default implementation, because both straightforward default
    # implementations have their issues:
    #
    #   + `return NotImplemented`:
    #     Calling `len(subclass_instance)` raises:
    #       TypeError: 'NotImplementedType' object cannot be interpreted as an integer
    #
    #   + `raise NotImplementedError()`:
    #     This prevents triggering some fallback behavior. E.g., the built-in
    #     `list(X)` tries to call `len(X)` first, and executes a different code
    #     path if the method is not found or `NotImplemented` is returned, while
    #     raising a `NotImplementedError` will propagate and make the call fail
    #     where it could have used `__iter__` to complete the call.
    #
    # Thus, the only two sensible things to do are
    #
    #   + **not** provide a default `__len__`.
    #
    #   + raise a `TypeError` instead, which is what Python uses when users call
    #     a method that is not defined on an object.
    #     (@ssnl verifies that this works on at least Python 3.7.)


class MixedDistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

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

###########################################################################################



def fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, mb):

    model.train()
    # Iterate over the batches of the dataset
    return_loss, batch_cnt = 0,0
    for images, targets in progress_bar(train_loader, parent=mb):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            images = images.to(device)
        images = batch_transforms(images)

        optimizer.zero_grad()
        
        train_loss = model(images, targets)["loss"]
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step()

        #master bar
        mb.child.comment = f"Training loss: {train_loss.item():.6}"
        
        return_loss += train_loss.item()
        batch_cnt += 1
   
    return_loss /= batch_cnt 
    return return_loss


@torch.no_grad()
def evaluate(model, val_loader, batch_transforms, val_metric, amp=False):
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    for images, targets in val_loader:
        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)
        out = model(images, targets, return_preds=True)
        # Compute metric
        loc_preds = out["preds"]
        for boxes_gt, boxes_pred in zip(targets, loc_preds):
            # Convert pred to boxes [xmin, ymin, xmax, ymax]  N, 4, 2 --> N, 4
            # boxes_pred = np.concatenate((boxes_pred.min(axis=1), boxes_pred.max(axis=1)), axis=-1)
            val_metric.update(gts=boxes_gt, preds=boxes_pred[:, :4])

        val_loss += out["loss"].item()
        batch_cnt += 1

        if(batch_cnt%100 == 0):
            print("Running eval for batch_cnt ",batch_cnt)

    val_loss /= batch_cnt
    recall, precision, mean_iou = val_metric.summary()
    return val_loss, recall, precision, mean_iou


def main(args):
    args.rotation = False
    args.eval_straight = True
    print(args)

    torch.backends.cudnn.benchmark = True
    system_available_memory = int(psutil.virtual_memory().available / 1024**3)

    
    # print("Current GPU memory usage (in bytes):", torch.cuda.memory_allocated())
    # print("Maximum GPU memory usage (in bytes):", torch.cuda.max_memory_allocated())

    st = time.time()
    # Load both train and val data generators
    #dataset1
    print("#######################################")
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
                    T.Resize(args.input_size, preserve_aspect_ratio=True),  # This does not pad
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation and not args.eval_straight
                else []
            )
        ),
        use_polygons= False,
    )
    # print(type(val_set_d1))
    # print(len(val_set_d1))
    # aa,bb = val_set_d1[0]
    # print(type(aa))
    # # print(aa.shape)
    # print(type(bb))

    # print(val_set_d1[0])

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
                    T.Resize(args.input_size, preserve_aspect_ratio=True),  # This does not pad
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

    val_sampler = torch.utils.data.distributed.DistributedSampler(
    	combined_val_set,
    	shuffle = True
    )

    val_loader = DataLoader(
        combined_val_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=val_sampler,
        # pin_memory=torch.cuda.is_available(),
        collate_fn=val_set_d1.collate_fn,
        drop_last=True
    )

    print(f"Validation set loaded in {time.time() - st:.4}s ({len(combined_val_set)} samples in " f"{len(val_loader)} batches)")
    
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
                    T.Resize(args.input_size, preserve_aspect_ratio=True),  # This does not pad
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation and not args.eval_straight
                else []
            )
        ),
        use_polygons= False,
    )

    # print("loading test dataset2")
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
    #                 T.Resize(args.input_size, preserve_aspect_ratio=True),  # This does not pad
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

    test_sampler = torch.utils.data.distributed.DistributedSampler(
    	# combined_test_set,
        test_set_d1,
    	shuffle = True
    )
    test_loader = DataLoader(
        # combined_test_set,
        test_set_d1,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=test_sampler,
        # pin_memory=torch.cuda.is_available(),
        collate_fn=test_set_d1.collate_fn,
        drop_last=True
    )

    print(f"test set loaded in {time.time() - st:.4}s ({len(test_set_d1)} samples in " f"{len(test_loader)} batches)")

    with open(os.path.join(args.val_path_d1, "labels.json"), "rb") as f:
        val_hash_d1 = hashlib.sha256(f.read()).hexdigest()
    
    with open(os.path.join(args.val_path_d2, "labels.json"), "rb") as f:
        val_hash_d2 = hashlib.sha256(f.read()).hexdigest()

    #combine both hashes
    combined_val_hash = hashlib.sha256((val_hash_d1 + val_hash_d2).encode()).hexdigest()

    with open(os.path.join(args.test_path_d1, "labels.json"), "rb") as f:
        test_hash_d1 = hashlib.sha256(f.read()).hexdigest()

    with open(os.path.join(args.test_path_d2, "labels.json"), "rb") as f:
        test_hash_d2 = hashlib.sha256(f.read()).hexdigest()

    #combine both hashes
    combined_test_hash = hashlib.sha256((test_hash_d1 + test_hash_d2).encode()).hexdigest()

    # with open(os.path.join(args.val_path, "labels.json"), "rb") as f:
    #     val_hash = hashlib.sha256(f.read()).hexdigest()
    # with open(os.path.join(args.test_path, "labels.json"), "rb") as f:
    #     test_hash = hashlib.sha256(f.read()).hexdigest()

    batch_transforms = Normalize(mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287))
    # batch_transforms = Normalize(mean=(0.5,), std=(1.,))


    # Load doctr model
    model = detection.__dict__["db_resnet50"](pretrained=args.pretrained, assume_straight_pages= True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Backbone freezing
    if args.freeze_backbone:
        model.feat_extractor.requires_grad_(False) #freeze_backbone=True;model....(False)->freezing 
                                                   #freeze_backbone=False;model....(False/True)->unfreezing
    
    # GPU
    # dist_url = "env://"  # default

    # rank = 0
    # world_size = 1
    # local_rank = 0

    # if 'RANK' in os.environ:
    #     rank = int(os.environ['RANK'])
    # if 'WORLD_SIZE' in os.environ:
    #     world_size = int(os.environ['WORLD_SIZE'])
    # if 'LOCAL_RANK' in os.environ:
    #     local_rank = int(os.environ['LOCAL_RANK'])

    # dist.init_process_group(backend='nccl', init_method=dist_url, world_size=world_size, rank=rank)
    # torch.cuda.set_device(local_rank)

    # Initialize DDP with the default parameters
    # torch.distributed.init_process_group(backend='nccl', init_method=dist_url)

    torch.cuda.set_device(local_rank)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])

    # dist.barrier()

    # Resume weights
    if isinstance(args.resume, str):
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        # print(checkpoint)
        # print(model)
        model.load_state_dict(checkpoint,strict = 'true')

    params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {params*4/1024/1024/1024:.4f} GB")

    print('Before loading data')
    print("Current GPU memory usage (in bytes):", torch.cuda.memory_allocated())
    print("Maximum GPU memory usage (in bytes):", torch.cuda.max_memory_allocated())
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs")
    #     model = torch.nn.DataParallel(model)
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # model= torch.nn.DataParallel(model)
    # model.to(device)

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
        print("Running evaluation for validation")
        val_loss, recall, precision, mean_iou = evaluate(model, val_loader, batch_transforms, val_metric)
        print(f"Validation loss: {val_loss} (Recall: {recall} | Precision: {precision} | Mean IoU: {mean_iou})")

        print("Running evaluation for test")
        test_loss, recall, precision, mean_iou = evaluate(model, test_loader, batch_transforms, test_metric)
        print(f"test loss: {test_loss} (Recall: {recall} | Precision: {precision} | Mean IoU: {mean_iou})")
        return

    st = time.time()
    # Load both train and val data generators
    #dataset1

    ############%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
                    T.Resize(args.input_size, preserve_aspect_ratio=True),
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
                    T.Resize(args.input_size, preserve_aspect_ratio=True),
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
                    T.Resize(args.input_size, preserve_aspect_ratio=True),
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation
                else []
            )
        ),
        use_polygons=False,
    )


    #class balancing for dataset1
    e_d1 = len(easy_train_set_d1)
    m_d1 = len(medium_train_set_d1)
    h_d1 = len(hard_train_set_d1)
    N_d1 = e_d1+m_d1+h_d1

    newmedium_d1 = e_d1* (N_d1-m_d1)/(N_d1-e_d1)
    newhard_d1 = e_d1* (N_d1-h_d1)/(N_d1-e_d1)
    print("size of Easy, Newmedium, Newhard, hard, medium, Total docs (all for train set): ",e_d1,newmedium_d1,newhard_d1,h_d1,m_d1,N_d1)

    #CB for hard dataset1
    new_hard_train_set_d1 = copy.deepcopy(hard_train_set_d1)

    while(len(new_hard_train_set_d1) + h_d1 < newhard_d1):
        new_hard_train_set_d1 = torch.utils.data.ConcatDataset([new_hard_train_set_d1, hard_train_set_d1])

    tail_end_size = math.floor(newhard_d1 - len(new_hard_train_set_d1))
    tail_end, extra = torch.utils.data.random_split(hard_train_set_d1,[tail_end_size,h_d1 - tail_end_size])
    new_hard_train_set_d1 = torch.utils.data.ConcatDataset([new_hard_train_set_d1, tail_end])
    
    #CB for medium dataset1
    new_medium_train_set_d1 = copy.deepcopy(medium_train_set_d1)

    while(len(new_medium_train_set_d1) + m_d1 < newmedium_d1):
        new_medium_train_set_d1 = torch.utils.data.ConcatDataset([new_medium_train_set_d1, medium_train_set_d1])

    tail_end_size = math.floor(newmedium_d1 - len(new_medium_train_set_d1))
    tail_end, extra = torch.utils.data.random_split(medium_train_set_d1,[tail_end_size,m_d1 - tail_end_size])
    new_medium_train_set_d1 = torch.utils.data.ConcatDataset([new_medium_train_set_d1, tail_end])


    print("length of the new medium and hard train set: ",len(new_medium_train_set_d1),len(new_hard_train_set_d1))
    train_set_d1 = torch.utils.data.ConcatDataset([easy_train_set_d1, new_medium_train_set_d1,new_hard_train_set_d1])  #trainset for dataset1
    
    Nnew_d1 = e_d1 + len(new_medium_train_set_d1) + len(new_hard_train_set_d1)
    M_d1 = math.floor(math.log(0.1, ((Nnew_d1-1)/Nnew_d1)))

    print("Length of new trainset: ", Nnew_d1)
    # print("Length of new trainset, number of documents seen in an epoch: ", Nnew, M)
    #
    ######%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
                    T.Resize(args.input_size, preserve_aspect_ratio=True),
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
                    T.Resize(args.input_size, preserve_aspect_ratio=True),
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
                    T.Resize(args.input_size, preserve_aspect_ratio=True),
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation
                else []
            )
        ),
        use_polygons=False,
    )

    #class balancing for dataset2
    e_d2 = len(easy_train_set_d2)
    m_d2 = len(medium_train_set_d2)
    h_d2 = len(hard_train_set_d2)
    N_d2 = e_d2+m_d2+h_d2

    newmedium_d2 = e_d2* (N_d2-m_d2)/(N_d2-e_d2)
    newhard_d2 = e_d2* (N_d2-h_d2)/(N_d2-e_d2)
    print("size of Easy, Newmedium, Newhard, hard, medium, Total docs (all for train set): ",e_d2,newmedium_d2,newhard_d2,h_d2,m_d2,N_d2)

    #CB for hard dataset1
    new_hard_train_set_d2 = copy.deepcopy(hard_train_set_d2)

    while(len(new_hard_train_set_d2) + h_d2 < newhard_d2):
        new_hard_train_set_d2 = torch.utils.data.ConcatDataset([new_hard_train_set_d2, hard_train_set_d2])

    tail_end_size = math.floor(newhard_d2 - len(new_hard_train_set_d2))
    tail_end, extra = torch.utils.data.random_split(hard_train_set_d2,[tail_end_size,h_d2 - tail_end_size])
    new_hard_train_set_d2 = torch.utils.data.ConcatDataset([new_hard_train_set_d2, tail_end])
    
    #CB for medium dataset1
    new_medium_train_set_d2 = copy.deepcopy(medium_train_set_d2)

    while(len(new_medium_train_set_d2) + m_d2 < newmedium_d2):
        new_medium_train_set_d2 = torch.utils.data.ConcatDataset([new_medium_train_set_d2, medium_train_set_d2])

    tail_end_size = math.floor(newmedium_d2 - len(new_medium_train_set_d2))
    tail_end, extra = torch.utils.data.random_split(medium_train_set_d2,[tail_end_size,m_d2 - tail_end_size])
    new_medium_train_set_d2 = torch.utils.data.ConcatDataset([new_medium_train_set_d2, tail_end])


    print("length of the new medium and hard train set: ",len(new_medium_train_set_d2),len(new_hard_train_set_d2))
    train_set_d2 = torch.utils.data.ConcatDataset([easy_train_set_d2, new_medium_train_set_d2,new_hard_train_set_d2])  #trainset for dataset1
    
    Nnew_d2 = e_d2 + len(new_medium_train_set_d2) + len(new_hard_train_set_d2)
    M_d2 = math.floor(math.log(0.1, ((Nnew_d2-1)/Nnew_d2)))

    print("Length of new trainset: ", Nnew_d2)
    # print("Length of new trainset, number of documents seen in an epoch: ", Nnew, M)
    #
    #######%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    # 	train_set,
    # 	shuffle = True
    # )
    #
    combined_train_set = torch.utils.data.ConcatDataset([train_set_d1,train_set_d2])
    mixed_sampler = MixedDistributedSampler(train_set_d1,train_set_d2,shuffle=True)

    # train_loader = DataLoader(
    #     train_set,
    #     batch_size=args.batch_size,
    #     num_workers=args.workers,
    #     sampler=train_sampler,
    #     drop_last=True,
    #     pin_memory=True,
    #     collate_fn=val_set.collate_fn
    # )

    train_loader = DataLoader(
        combined_train_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=mixed_sampler,
        drop_last=True,
        pin_memory=True,
        collate_fn=val_set_d1.collate_fn
    )
    print(len(train_loader))
    print(f"Train set loaded in {time.time() - st:.4}s ({len(combined_train_set)} samples in " f"{len(train_loader)} batches)")
    
    print('After loading data')
    print("Current GPU memory usage (in bytes):", torch.cuda.memory_allocated())
    print("Maximum GPU memory usage (in bytes):", torch.cuda.max_memory_allocated())

    #changed
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
        return

    # Optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        args.lr,
        betas=(0.95, 0.99),
        eps=1e-6,
        weight_decay=args.weight_decay,
    )

    # Scheduler (using cosine)
    scheduler = CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=args.lr / 25e4)


    # Training monitoring
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"db_resnet50_{current_time}" if args.name is None else args.name
    if isinstance(args.resume, str):
        file_name = os.path.basename(args.resume)
        exp_name = file_name[:file_name.find("_epoch")]
    # W&B
    if args.wb and dist.get_rank() == 0:

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
                "easy_train_hash_d1": easy_train_hash_d1,
                "medium_train_hash_d1": medium_train_hash_d1,
                "hard_train_hash_d1": hard_train_hash_d1,
                
                "easy_train_hash_d2": easy_train_hash_d2,
                "medium_train_hash_d2": medium_train_hash_d2,
                "hard_train_hash_d2": hard_train_hash_d2,


                "val_hash_d1": val_hash_d1,
                "test_hash_d1": test_hash_d1,

                "val_hash_d2": val_hash_d2,
                "test_hash_d2": test_hash_d2,
                "pretrained": args.pretrained,
                "rotation": False
            }, 
            resume="allow"
        )
        last_epoch = wandb.run.summary.get('epoch',0)
        print('Last Epoch:',last_epoch)
    print(model)
    print(' ')
    print(model.named_parameters())
    print(' ')
    ccc=0
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            ccc+=1
            # print(name)
    print(ccc)  

    # Create loss queue
    min_loss = np.inf
    patience = 5

    # Training loop
    mb = master_bar(range(args.epochs))
    for epoch in mb:
        train_loss = fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, mb)
        # Validation loop at the end of each epoch
        val_loss, recall, precision, mean_iou = evaluate(model, val_loader, batch_transforms, val_metric)
        
        # print("b0 - ", train_loss)
        # print("b1 - ", type(train_loss))

        # print("b0  -  ",val_loss)
        world_size = int(os.environ['WORLD_SIZE'])
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

        # print("b2  -  ",val_loss)

        dist.all_reduce(train_loss, op=ReduceOp.SUM)
        dist.all_reduce(val_loss, op=ReduceOp.SUM)
        dist.all_reduce(recall, op=ReduceOp.SUM)
        dist.all_reduce(precision, op=ReduceOp.SUM)
        dist.all_reduce(mean_iou, op=ReduceOp.SUM)
        
        # print("b3  -  ",val_loss)
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), f"./models/{exp_name}_epoch{epoch}.pt")
            
            if val_loss < min_loss :
                print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
                min_loss = val_loss
                counter = 0
            else:
                counter+=1

            #check if training should be stopped
            if counter == patience:
                print("Validation loss hasn't improved in", patience, "epochs. Early stopping.")
                break

            log_msg = f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} "
            if any(val is None for val in (recall, precision, mean_iou)):
                log_msg += "(Undefined metric value, caused by empty GTs or predictions)"
            else:
                log_msg += f"(Recall: {recall:.2%} | Precision: {precision:.2%} | Mean IoU: {mean_iou:.2%})"
            mb.write(log_msg)
            # W&B
            if args.wb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "average_recall": recall,
                        "average_precision": precision,
                        "average_mean_iou": mean_iou,
                        "epoch": epoch + 10
                    }
                )

    if args.wb and dist.get_rank() == 0:
        run.finish()



def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR training script for text detection (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    #changed
    # parser.add_argument("--easy_train_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset1/scratch/abhaynew/newfolder/train/Easy", help="path to training data folder")
    # parser.add_argument("--medium_train_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset1/scratch/abhaynew/newfolder/train/Medium", help="path to training data folder")
    # parser.add_argument("--hard_train_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset1/scratch/abhaynew/newfolder/train/Hard", help="path to training data folder")
    # parser.add_argument("--val_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset1/scratch/abhaynew/newfolder/val", help="path to validation data folder")
    # parser.add_argument("--test_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset1/scratch/abhaynew/newfolder/test", help="path to test data folder")
    
    # #dataset2
    # parser.add_argument("--easy_train_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset2/train/Easy", help="path to training data folder")
    # parser.add_argument("--medium_train_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset2/train/Medium", help="path to training data folder")
    # parser.add_argument("--hard_train_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset2/train/Hard", help="path to training data folder")
    # parser.add_argument("--val_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset2/val", help="path to validation data folder")
    # parser.add_argument("--test_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset2/test", help="path to test data folder")

    parser.add_argument("--easy_train_path_d1", type=str,default="//data3/sreevatsa/Datasets/dataset1/extract/scratch/abhaynew/newfolder/train/Easy", help="path to training data folder")
    parser.add_argument("--medium_train_path_d1", type=str,default="/data3/sreevatsa/Datasets/dataset1/extract/scratch/abhaynew/newfolder/train/Medium", help="path to training data folder")
    parser.add_argument("--hard_train_path_d1", type=str,default="/data3/sreevatsa/Datasets/dataset1/extract/scratch/abhaynew/newfolder/train/Hard", help="path to training data folder")
    parser.add_argument("--val_path_d1", type=str,default="/data3/sreevatsa/Datasets/dataset1/extract/scratch/abhaynew/newfolder/val", help="path to validation data folder")
    parser.add_argument("--test_path_d1", type=str,default="/data3/sreevatsa/Datasets/dataset1/extract/scratch/abhaynew/newfolder/test", help="path to test data folder")
    
    #dataset2
    parser.add_argument("--easy_train_path_d2", type=str,default="/data3/sreevatsa/Datasets/dataset2/train/Easy", help="path to training data folder")
    parser.add_argument("--medium_train_path_d2", type=str,default="/data3/sreevatsa/Datasets/dataset2/train/Medium", help="path to training data folder")
    parser.add_argument("--hard_train_path_d2", type=str,default="/data3/sreevatsa/Datasets/dataset2/train/Hard", help="path to training data folder")
    parser.add_argument("--val_path_d2", type=str,default="/data3/sreevatsa/Datasets/dataset2/val", help="path to validation data folder")
    parser.add_argument("--test_path_d2", type=str,default="/data3/sreevatsa/Datasets/dataset2/test", help="path to test data folder")
    


    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch_size", type=int, default=40, help="batch size for training")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("--input_size", type=int, default=512, help="model input size, H = W")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam)")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("-j", "--workers", type=int, default=8, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Run the validation loop")
    parser.add_argument(
        "--freeze-backbone", dest="freeze_backbone", action="store_true", help="freeze model backbone for fine-tuning"
    )
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
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int, 
                        help='local rank for distributed training')
    args = parser.parse_args()

    return args

#
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

# def init_distributed():
#     # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
#     dist_url = "env://" # default

#     rank = 0
#     world_size = 1
#     local_rank = 0

#     if 'RANK' in os.environ:
#         rank = int(os.environ['RANK'])
#     if 'WORLD_SIZE' in os.environ:
#         world_size = int(os.environ['WORLD_SIZE'])
#     if 'LOCAL_RANK' in os.environ:
#         local_rank = int(os.environ['LOCAL_RANK'])

#     print(local_rank)
#     dist.init_process_group(backend="nccl", init_method=dist_url, world_size=world_size, rank=rank)
#     # dist.init_process_group(backend = "nccl")
#     # this will make all .cuda() calls work properly
#     torch.cuda.set_device(local_rank)
#     # synchronizes all the threads to reach this point before moving on
#     dist.barrier()
#     if is_master:
#         print("DDP is working correctly with {} processes.".format(world_size))
#     setup_for_distributed(rank == 0)

# def init_distributed():

#     rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     is_master = (rank == 0)

#     dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
#     is_distributed = dist.is_initialized()

#     # Check if DDP is working correctly
#     # rank = dist.get_rank()
#     # world_size = dist.get_world_size()
#     # is_master = (rank == 0)
    
#     if is_master:
#         print("DDP is working correctly with {} processes.".format(world_size))
    
#     dist.barrier()


# LOCAL_RANK = int(os.environ['LOCAL_RANK'])
# WORLD_SIZE = int(os.environ['WORLD_SIZE'])
# WORLD_RANK = int(os.environ['RANK'])

def run(backend):
    tensor = torch.zeros(1)
    
    # Need to put tensor on a GPU device for nccl backend
    if backend == 'nccl':
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)

    if WORLD_RANK == 0:
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            print('worker_{} sent data to Rank {}\n'.format(0, rank_recv))
    else:
        dist.recv(tensor=tensor, src=0)
        print('worker_{} has received data from rank {}\n'.format(WORLD_RANK, 0))

def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    run(backend)


if __name__ == "__main__":
    args = parse_args()
    #
    init_distributed()
    # init_processes(backend='nccl')
    main(args)
    # wandb.finish()
