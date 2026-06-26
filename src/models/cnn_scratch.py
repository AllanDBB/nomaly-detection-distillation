"""Model A — CNN classifier trained from scratch (no pretrained weights)."""
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn_base import CNNBase


class CNNScratchModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = CNNBase(num_classes=num_classes, hidden_dim=hidden_dim)
        self.criterion = nn.CrossEntropyLoss()

        # Storage for test embeddings
        self._test_embeddings: List[torch.Tensor] = []
        self._test_anomaly_labels: List[torch.Tensor] = []
        self._test_class_labels: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        return self.model(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_embedding(x)

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        imgs, class_labels, _ = batch
        logits, _ = self(imgs)
        loss = self.criterion(logits, class_labels)
        acc = (logits.argmax(1) == class_labels).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        imgs, class_labels, _ = batch
        logits, _ = self(imgs)
        loss = self.criterion(logits, class_labels)
        acc = (logits.argmax(1) == class_labels).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        imgs, class_labels, anomaly_labels = batch
        logits, embeddings = self(imgs)
        loss = self.criterion(logits, class_labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self._test_embeddings.append(embeddings.detach().cpu())
        self._test_anomaly_labels.append(anomaly_labels.detach().cpu())
        self._test_class_labels.append(class_labels.detach().cpu())
        return {"embeddings": embeddings, "anomaly_labels": anomaly_labels}

    def on_test_epoch_end(self) -> None:
        self._test_embeddings = []
        self._test_anomaly_labels = []
        self._test_class_labels = []

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        t_max = self.trainer.max_epochs if (self.trainer and self.trainer.max_epochs) else 50
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
