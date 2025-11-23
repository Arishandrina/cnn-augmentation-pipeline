import random
from collections import Counter
from typing import Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import SVHN


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_svhn_for_eda(data_root: str) -> Tuple[SVHN, SVHN]:
    transform_eda = T.ToTensor()
    train = SVHN(root=data_root, split="train", download=True, transform=transform_eda)
    test = SVHN(root=data_root, split="test", download=True, transform=transform_eda)
    return train, test


def compute_channel_stats(dataset, batch_size: int = 512, num_workers: int = 4) -> Tuple[List[float], List[float]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    mean = torch.zeros(3)
    var = torch.zeros(3)
    batches = 0
    for x, _ in loader:
        batch_mean = x.mean(dim=(0, 2, 3))
        batch_var = x.var(dim=(0, 2, 3), unbiased=False)
        mean = (mean * batches + batch_mean) / (batches + 1)
        var = (var * batches + batch_var) / (batches + 1)
        batches += 1
    std = torch.sqrt(var)
    return mean.tolist(), std.tolist()


def class_distribution(dataset) -> Dict[int, int]:
    counts = Counter()
    for _, y in dataset:
        counts[int(y)] += 1
    return dict(counts)


def make_transforms(mean: List[float], std: List[float]):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return transform, transform  



# сама внутри строит трансформации через make_transforms(mean, std) (baseline без аугментаций, фиксированный pipeline)
def get_dataloaders_with_split(data_root: str, mean: List[float], std: List[float], batch_size: int, val_fraction: float, 
                               seed: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_t, eval_t = make_transforms(mean, std)

    train_full_a = SVHN(root=data_root, split="train", download=True, transform=train_t)
    train_full_b = SVHN(root=data_root, split="train", download=False, transform=eval_t) # для val свой transform
    test = SVHN(root=data_root, split="test", download=True, transform=eval_t)

    num_val = int(len(train_full_a) * val_fraction)
    num_train = len(train_full_a) - num_val

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_full_a), generator=g).tolist()
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    train = Subset(train_full_a, train_idx)
    val = Subset(train_full_b, val_idx)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader



def get_dataloaders_with_custom_train_transform(data_root: str, train_transform, eval_transform, batch_size: int, 
                val_fraction: float, seed: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_full_a = SVHN(root=data_root, split="train", download=True, transform=train_transform)
    train_full_b = SVHN(root=data_root, split="train", download=False, transform=eval_transform)
    test = SVHN(root=data_root, split="test", download=True, transform=eval_transform)

    num_val = int(len(train_full_a) * val_fraction)
    num_train = len(train_full_a) - num_val

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_full_a), generator=g).tolist()
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]

    train = Subset(train_full_a, train_idx)
    val = Subset(train_full_b, val_idx)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader