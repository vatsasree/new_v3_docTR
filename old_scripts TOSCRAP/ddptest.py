import os

os.environ["USE_TORCH"] = "1"

import datetime
import hashlib
import logging
import multiprocessing as mp
import time

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
    dist.barrier()
    if is_master:
        print("DDP is working correctly with {} processes.".format(world_size))
    setup_for_distributed(rank == 0)
    # Print memory usage of each GPU
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        print(f"Memory usage on GPU {i}: {torch.cuda.memory_allocated(i)} / {torch.cuda.memory_reserved(i)}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR training script for text detection (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    #changed
    parser.add_argument("--easy_train_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset1/scratch/abhaynew/newfolder/train/Easy", help="path to training data folder")
    parser.add_argument("--medium_train_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset1/scratch/abhaynew/newfolder/train/Medium", help="path to training data folder")
    parser.add_argument("--hard_train_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset1/scratch/abhaynew/newfolder/train/Hard", help="path to training data folder")
    parser.add_argument("--val_path_d1", type=str,default="/data3/sreevatsa/Datasets/dataset2/val", help="path to validation data folder")
    parser.add_argument("--test_path_d1", type=str,default="/scratch/sreevatsa/Datasets/dataset1/scratch/abhaynew/newfolder/test", help="path to test data folder")
    
    #dataset2
    parser.add_argument("--easy_train_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset2/train/Easy", help="path to training data folder")
    parser.add_argument("--medium_train_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset2/train/Medium", help="path to training data folder")
    parser.add_argument("--hard_train_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset2/train/Hard", help="path to training data folder")
    parser.add_argument("--val_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset2/val", help="path to validation data folder")
    parser.add_argument("--test_path_d2", type=str,default="/scratch/sreevatsa/Datasets/dataset2/test", help="path to test data folder")
    


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
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parse_args()
    #
    init_distributed()
    # init_processes(backend='nccl')
    #main(args)
    # wandb.finish()
