import torch
import torchvision.transforms as T
from typing import List


class AddGaussianNoise:
    def __init__(self, noise_std: float = 0.03):
        self.noise_std = noise_std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x += noise
        return torch.clamp(x, 0.0, 1.0)
    


def get_eval_transform(mean: List[float], std: List[float]) -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])



def stage1_basic(mean: List[float], std: List[float]) -> T.Compose:
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15, interpolation=T.InterpolationMode.BILINEAR, fill=0),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])



def stage2_geometry(mean: List[float], std: List[float]) -> T.Compose:
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), 
                       interpolation=T.InterpolationMode.BILINEAR, fill=0),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])



def stage3_color(mean: List[float], std: List[float]) -> T.Compose:
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), 
                       interpolation=T.InterpolationMode.BILINEAR, fill=0),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])



def stage4_blur_noise(mean: List[float], std: List[float]) -> T.Compose:
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), 
                       interpolation=T.InterpolationMode.BILINEAR, fill=0),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
        T.ToTensor(),
        AddGaussianNoise(noise_std=0.03),
        T.Normalize(mean=mean, std=std)
    ])



def stage5_occlusion(mean: List[float], std: List[float]) -> T.Compose:
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), 
                       interpolation=T.InterpolationMode.BILINEAR, fill=0),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
        T.ToTensor(),
        AddGaussianNoise(noise_std=0.03),
        T.Normalize(mean=mean, std=std),
        T.RandomErasing(p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random', inplace=False)
    ])



class ProbTransform:
    def __init__(self, base_t: T.Compose, aug_t: T.Compose, p_aug: float):
        self.base_t = base_t
        self.aug_t = aug_t
        self.p_aug = float(p_aug)
    def __call__(self, img):
        return (self.aug_t if torch.rand(1).item() < self.p_aug else self.base_t)(img)