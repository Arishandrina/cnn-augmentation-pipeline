from typing import Dict, List
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F



def plot_split_distributions(train_ds, test_ds, class_names=None) -> None:
    ct = Counter(int(y) for _, y in train_ds)
    cte = Counter(int(y) for _, y in test_ds)
    xs = list(range(10))
    train_vals = [ct.get(i, 0) for i in xs]
    test_vals  = [cte.get(i, 0) for i in xs]
    width = 0.4

    plt.figure(figsize=(8,4))
    plt.bar([i - width/2 for i in xs], train_vals, width=width, label="train")
    plt.bar([i + width/2 for i in xs], test_vals,  width=width, label="test")
    plt.xticks(xs, [str(i) for i in xs] if class_names is None else class_names)
    plt.title("Распределение классов: train vs test")
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_examples_per_class(dataset, classes: List[int], samples_per_class: int = 6) -> None:
    indices_by_class = defaultdict(list)
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        y = int(y)
        if len(indices_by_class[y]) < samples_per_class:
            indices_by_class[y].append(idx)
        if all(len(v) >= samples_per_class for v in indices_by_class.values()) and len(indices_by_class) == len(classes):
            break

    rows = len(classes)
    cols = samples_per_class
    plt.figure(figsize=(2.0 * cols, 1.8 * rows))
    for row, cls in enumerate(classes):
        for col, idx in enumerate(indices_by_class[cls]):
            img, _ = dataset[idx]
            img_np = img.permute(1, 2, 0).numpy()
            plt.subplot(rows, cols, row * cols + col + 1)
            plt.imshow(img_np)
            plt.axis("off")
            if col == 0:
                plt.ylabel(f"class {cls}", rotation=90, fontsize=10)
    plt.suptitle("Примеры изображений по классам", y=0.995)
    plt.tight_layout()
    plt.show()



def compute_brightness_contrast_sharpness(dataset, num_samples=3000):
    idxs = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    brightness, contrast, sharpness = [], [], []

    # Лапласиан 3x3
    lap = torch.tensor([[0., 1., 0.],
                        [1., -4., 1.],
                        [0., 1., 0.]]).view(1,1,3,3)

    for i in idxs:
        x, _ = dataset[i]          # [C,H,W], в [0,1]
        # яркость/контраст на gray
        gray = (0.299*x[0] + 0.587*x[1] + 0.114*x[2]).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        b = gray.mean().item()
        c = gray.std(unbiased=False).item()
        # резкость: дисперсия отклика лапласиана
        resp = F.conv2d(gray, lap, padding=1)
        s = resp.pow(2).mean().item()
        brightness.append(b)
        contrast.append(c)
        sharpness.append(s)

    return np.array(brightness), np.array(contrast), np.array(sharpness)


def plot_brightness_contrast_sharpness(bright, contr, sharp) -> None:
    plt.figure(figsize=(10,3))
    plt.subplot(1,3,1); plt.hist(bright, bins=40, color="#1f77b4", density=True); plt.title("Яркость (ср. по пикселям)")
    plt.subplot(1,3,2); plt.hist(contr,  bins=40, color="#ff7f0e", density=True); plt.title("Контраст (std)")
    plt.subplot(1,3,3); plt.hist(sharp,  bins=40, color="#2ca02c", density=True); plt.title("Резкость (Var Laplacian)")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(4.5,4))
    plt.scatter(bright, contr, s=6, alpha=0.3)
    plt.xlabel("Яркость"); plt.ylabel("Контраст"); plt.title("Яркость vs Контраст")
    plt.tight_layout(); plt.show()



def plot_class_means(dataset, classes=range(10), per_class_limit=500):
    sums = {c: torch.zeros(3,32,32) for c in classes}
    counts = {c: 0 for c in classes}
    for i in range(len(dataset)):
        x, y = dataset[i]
        y = int(y)
        if y in sums and counts[y] < per_class_limit:
            sums[y] += x
            counts[y] += 1
        if all(counts[c] >= per_class_limit for c in classes):
            break

    cols = 5
    rows = int(np.ceil(len(classes)/cols))
    plt.figure(figsize=(2.2*cols, 2.2*rows))
    for j, c in enumerate(classes):
        mean_img = (sums[c] / max(1, counts[c])).permute(1,2,0).numpy()
        mean_img = np.clip(mean_img, 0, 1)
        plt.subplot(rows, cols, j+1)
        plt.imshow(mean_img); plt.axis("off"); plt.title(str(c))
    plt.suptitle("Средние изображения по классам", y=0.98)
    plt.tight_layout(); plt.show()



def plot_channel_histograms(dataset, num_samples: int = 4000, phase: str = "Train") -> None:
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    pixels = []
    for idx in indices:
        x, _ = dataset[idx]
        pixels.append(x.reshape(3, -1).numpy())
    if not pixels:
        return
    pixels = np.concatenate(pixels, axis=1)  # [3, N]
    plt.figure(figsize=(10, 4))
    for ch, color in enumerate(["r", "g", "b"]):
        sns.histplot(pixels[ch], bins=50, color=color, stat="density", element="step", fill=False, label=f"ch{ch}")
    plt.title(f"Распределение интенсивностей({phase})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()




def channel_correlations(dataset, num_samples=5000) -> np.ndarray:
    idxs = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    acc = []
    for i in idxs:
        x, _ = dataset[i]            # [3,H,W]
        acc.append(x.reshape(3, -1).numpy())
    X = np.concatenate(acc, axis=1)  # [3, N]
    corr = np.corrcoef(X)
    
    plt.figure(figsize=(3.5,3))
    sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap="coolwarm", xticklabels=["R","G","B"], yticklabels=["R","G","B"])
    plt.title("Корреляция каналов"); plt.tight_layout(); plt.show()
    return corr