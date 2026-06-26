"""DBSCAN clustering + PCA/t-SNE dimensionality reduction utilities."""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_pca(
    embeddings: np.ndarray,
    n_components: int = 50,
    for_plot: bool = False,
) -> Tuple[np.ndarray, PCA]:
    """PCA reduction. Returns (reduced, fitted_pca)."""
    n_comp = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    reduced = pca.fit_transform(embeddings)
    if for_plot and n_comp > 2:
        reduced = reduced[:, :2]
    return reduced, pca


def reduce_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    pca_first: bool = True,
) -> np.ndarray:
    """t-SNE reduction, optionally preceded by PCA to 50 dims."""
    data = embeddings
    if pca_first and embeddings.shape[1] > 50:
        data, _ = reduce_pca(embeddings, n_components=50)
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(data) - 1),
        random_state=42,
        max_iter=1000,
    )
    return tsne.fit_transform(data)


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

def run_dbscan(
    embeddings_2d: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> np.ndarray:
    """Fit DBSCAN and return cluster labels (-1 = noise = potential anomaly)."""
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    return clusterer.fit_predict(embeddings_2d)


def tune_dbscan(
    embeddings_2d: np.ndarray,
    eps_values: list = None,
    min_samples_values: list = None,
) -> Tuple[float, int, float]:
    """Grid-search DBSCAN params by silhouette score (ignores noise points)."""
    if eps_values is None:
        eps_values = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    if min_samples_values is None:
        min_samples_values = [3, 5, 10, 15]

    best_score, best_eps, best_min = -1.0, eps_values[0], min_samples_values[0]
    for eps in eps_values:
        for ms in min_samples_values:
            labels = run_dbscan(embeddings_2d, eps=eps, min_samples=ms)
            valid = labels != -1
            if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
                continue
            score = silhouette_score(embeddings_2d[valid], labels[valid])
            if score > best_score:
                best_score, best_eps, best_min = score, eps, ms

    return best_eps, best_min, best_score


# ---------------------------------------------------------------------------
# Quantitative DBSCAN evaluation
# ---------------------------------------------------------------------------

def dbscan_anomaly_metrics(
    dbscan_labels: np.ndarray,
    true_anomaly_labels: np.ndarray,
) -> dict:
    """
    Treat DBSCAN noise points (label == -1) as predicted anomalies.
    Compare against ground-truth anomaly labels.
    """
    predicted_anomaly = (dbscan_labels == -1).astype(int)
    tp = int(((predicted_anomaly == 1) & (true_anomaly_labels == 1)).sum())
    tn = int(((predicted_anomaly == 0) & (true_anomaly_labels == 0)).sum())
    fp = int(((predicted_anomaly == 1) & (true_anomaly_labels == 0)).sum())
    fn = int(((predicted_anomaly == 0) & (true_anomaly_labels == 1)).sum())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    noise_rate = (dbscan_labels == -1).mean()
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "Accuracy": float(accuracy),
        "noise_rate": float(noise_rate),
        "n_clusters": int(n_clusters),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

PALETTE = {-1: "red", 0: "steelblue", 1: "seagreen", 2: "darkorange", 3: "purple"}


def _scatter(ax, xy, labels, label_names, title, alpha=0.6, s=12):
    unique = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique), 1)))
    for i, lbl in enumerate(unique):
        mask = np.array(labels) == lbl
        ax.scatter(xy[mask, 0], xy[mask, 1], c=[colors[i]], label=label_names.get(lbl, str(lbl)),
                   alpha=alpha, s=s, linewidths=0)
    ax.set_title(title)
    ax.legend(loc="best", markerscale=2, fontsize=8)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")


def plot_2d_embeddings(
    embeddings_2d: np.ndarray,
    dbscan_labels: np.ndarray,
    true_anomaly_labels: np.ndarray,
    class_labels: Optional[np.ndarray] = None,
    class_names: Optional[list] = None,
    method: str = "PCA",
    figsize: Tuple[int, int] = (16, 5),
) -> plt.Figure:
    ncols = 3 if class_labels is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    fig.suptitle(f"Embedding space — {method}", fontsize=14, fontweight="bold")

    # DBSCAN clusters
    dbscan_names = {-1: "Noise (predicted anomaly)"}
    for lbl in np.unique(dbscan_labels):
        if lbl >= 0:
            dbscan_names[lbl] = f"Cluster {lbl}"
    _scatter(axes[0], embeddings_2d, dbscan_labels.tolist(), dbscan_names, f"DBSCAN — {method}")

    # Ground truth anomaly labels
    gt_names = {0: "Normal", 1: "Anomaly"}
    _scatter(axes[1], embeddings_2d, true_anomaly_labels.tolist(), gt_names, "Ground Truth")

    # Class labels (optional)
    if class_labels is not None and class_names is not None:
        cls_names = {i: n for i, n in enumerate(class_names)}
        _scatter(axes[2], embeddings_2d, class_labels.tolist(), cls_names, "Class Labels")

    plt.tight_layout()
    return fig


def plot_explained_variance(pca: PCA, figsize: Tuple[int, int] = (8, 4)) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title("PCA — per-component variance")

    axes[1].plot(range(1, len(pca.explained_variance_ratio_) + 1),
                 np.cumsum(pca.explained_variance_ratio_), marker="o", ms=3)
    axes[1].axhline(0.95, color="red", linestyle="--", label="95% variance")
    axes[1].set_xlabel("# Components")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title("PCA — cumulative variance")
    axes[1].legend()

    plt.tight_layout()
    return fig
