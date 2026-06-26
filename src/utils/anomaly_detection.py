"""Anomaly detection utilities using Mahalanobis distance and reconstruction error."""
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Mahalanobis distance
# ---------------------------------------------------------------------------

def compute_distribution_params(
    embeddings: np.ndarray, reg: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and regularised precision matrix from normal embeddings."""
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    cov = (centered.T @ centered) / max(len(embeddings) - 1, 1)
    cov += reg * np.eye(cov.shape[0])
    precision = np.linalg.inv(cov)
    return mean, precision


def mahalanobis_scores(
    embeddings: np.ndarray, mean: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    """Return Mahalanobis distance for each embedding."""
    diff = embeddings - mean  # N x D
    scores = np.einsum("nd,dd,nd->n", diff, precision, diff)
    return np.sqrt(np.maximum(scores, 0.0))


def per_class_mahalanobis_scores(
    embeddings: np.ndarray,
    class_labels: np.ndarray,
    params_per_class: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Minimum Mahalanobis distance to any class distribution."""
    n = len(embeddings)
    num_classes = len(params_per_class)
    dist_matrix = np.zeros((n, num_classes))
    for c, (mean, prec) in params_per_class.items():
        dist_matrix[:, c] = mahalanobis_scores(embeddings, mean, prec)
    return dist_matrix.min(axis=1)


def fit_per_class_params(
    embeddings: np.ndarray, class_labels: np.ndarray, reg: float = 1e-6
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Fit one Gaussian distribution per class on normal validation embeddings."""
    params = {}
    for c in np.unique(class_labels):
        mask = class_labels == c
        params[int(c)] = compute_distribution_params(embeddings[mask], reg=reg)
    return params


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def find_threshold(normal_scores: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(normal_scores, percentile))


def classify(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores > threshold).astype(int)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_anomaly_detection(
    scores: np.ndarray,
    true_labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    preds = classify(scores, threshold)
    auroc = roc_auc_score(true_labels, scores)
    ap = average_precision_score(true_labels, scores)
    cm = confusion_matrix(true_labels, preds)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    return {
        "AUROC": float(auroc),
        "AP": float(ap),
        "F1": float(f1),
        "Precision": float(precision),
        "Recall": float(recall),
        "Accuracy": float(accuracy),
        "Threshold": float(threshold),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def get_roc_curve(scores: np.ndarray, true_labels: np.ndarray):
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    return fpr, tpr, thresholds


# ---------------------------------------------------------------------------
# Embedding extraction helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, dataloader, device: str = "cuda"):
    """Extract embeddings and labels from a dataloader using model.get_embedding()."""
    model.eval()
    model.to(device)
    all_embeddings, all_class_labels, all_anomaly_labels = [], [], []

    for imgs, class_labels, anomaly_labels in dataloader:
        imgs = imgs.to(device)
        embeddings = model.get_embedding(imgs).cpu()
        all_embeddings.append(embeddings)
        all_class_labels.append(class_labels)
        all_anomaly_labels.append(anomaly_labels)

    return (
        torch.cat(all_embeddings).numpy(),
        torch.cat(all_class_labels).numpy(),
        torch.cat(all_anomaly_labels).numpy(),
    )


@torch.no_grad()
def extract_recon_errors(model, dataloader, device: str = "cuda"):
    """Extract per-image reconstruction errors for the U-Net autoencoder."""
    import torch.nn.functional as F
    model.eval()
    model.to(device)
    all_errors, all_embeddings, all_class_labels, all_anomaly_labels = [], [], [], []

    for imgs, class_labels, anomaly_labels in dataloader:
        imgs = imgs.to(device)
        recon, embeddings = model(imgs)
        errors = F.mse_loss(recon, imgs, reduction="none").mean(dim=(1, 2, 3)).cpu()
        all_errors.append(errors)
        all_embeddings.append(embeddings.cpu())
        all_class_labels.append(class_labels)
        all_anomaly_labels.append(anomaly_labels)

    return (
        torch.cat(all_errors).numpy(),
        torch.cat(all_embeddings).numpy(),
        torch.cat(all_class_labels).numpy(),
        torch.cat(all_anomaly_labels).numpy(),
    )
