"""Callback que guarda train/val loss y accuracy por época para análisis de overfitting."""
from typing import Dict, List
import matplotlib.pyplot as plt
import pytorch_lightning as pl


class MetricsHistory(pl.Callback):
    def __init__(self):
        self.train_loss: List[float] = []
        self.val_loss:   List[float] = []
        self.train_acc:  List[float] = []
        self.val_acc:    List[float] = []

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if "train_loss" in m:
            self.train_loss.append(float(m["train_loss"]))
        if "train_acc" in m:
            self.train_acc.append(float(m["train_acc"]))

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if "val_loss" in m:
            self.val_loss.append(float(m["val_loss"]))
        if "val_acc" in m:
            self.val_acc.append(float(m["val_acc"]))

    def plot(self, run_name: str = "", save_path: str = None) -> plt.Figure:
        has_acc = len(self.train_acc) > 0
        ncols = 2 if has_acc else 1
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 4))
        if ncols == 1:
            axes = [axes]

        epochs = range(1, len(self.train_loss) + 1)

        # Loss
        axes[0].plot(epochs, self.train_loss, label="Train loss", marker="o", ms=3)
        if self.val_loss:
            val_ep = range(1, len(self.val_loss) + 1)
            axes[0].plot(val_ep, self.val_loss, label="Val loss", marker="s", ms=3)
        axes[0].set_xlabel("Época")
        axes[0].set_ylabel("Loss")
        axes[0].set_title(f"Loss — {run_name}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Gap análisis (overfitting)
        if self.train_loss and self.val_loss:
            n = min(len(self.train_loss), len(self.val_loss))
            gap = [v - t for t, v in zip(self.train_loss[:n], self.val_loss[:n])]
            ax2 = axes[0].twinx()
            ax2.fill_between(range(1, n + 1), gap, alpha=0.15, color="red", label="Gap (val-train)")
            ax2.set_ylabel("Gap (overfitting)", color="red")
            ax2.tick_params(axis="y", colors="red")

        # Accuracy
        if has_acc:
            axes[1].plot(epochs, self.train_acc, label="Train acc", marker="o", ms=3)
            if self.val_acc:
                val_ep = range(1, len(self.val_acc) + 1)
                axes[1].plot(val_ep, self.val_acc, label="Val acc", marker="s", ms=3)
            axes[1].set_xlabel("Época")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title(f"Accuracy — {run_name}")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        return fig
