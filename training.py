import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Optional
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm
from typing import Tuple, Dict


def _epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, 
           optimizer: Optional[Optimizer] = None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    preds_all = []
    targets_all = []

    total_correct = 0
    total_examples = 0
    phase = "Train" if is_train else "Val"
    pbar = tqdm(loader, desc=phase, leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_examples += targets.size(0)

        total_loss += loss.item() * targets.size(0)
        preds_all.append(preds.detach().cpu().numpy())
        targets_all.append(targets.detach().cpu().numpy())

        pbar.set_postfix({
            "loss": loss.item(),
            "accuracy": total_correct / total_examples
        })

    avg_loss = total_loss / total_examples
    acc = total_correct / total_examples

    y_pred = np.concatenate(preds_all)
    y_true = np.concatenate(targets_all)

    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0, labels=list(range(10))
    )

    metrics = {
        "loss": avg_loss,
        "accuracy": acc,
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
    }
    return metrics, y_true, y_pred


def train_epoch(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, 
                optimizer: Optimizer) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    return _epoch(model, loader, device, criterion, optimizer)


def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, 
               criterion: nn.Module) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    return _epoch(model, loader, device, criterion, optimizer=None)


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
    device: torch.device, criterion: nn.Module, optimizer: Optimizer, 
    epochs: int) -> Dict[str, Dict[str, float]]:

    best_f1 = 0.0
    best_state = None
    history = {"train": {}, "val": {}}

    for epoch in range(1, epochs + 1):
        train_metrics, _, _ = train_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics, _, _ = eval_epoch(model, val_loader, device, criterion)

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} f1={train_metrics['f1_macro']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1_macro']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, 
             criterion: nn.Module) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    metrics, y_true, y_pred = eval_epoch(model, loader, device, criterion)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    return metrics, y_true, y_pred, cm