"""Model B — CNN classifier trained via teacher-student knowledge distillation.

Teacher : pretrained ResNet-18 (ImageNet weights, fully frozen).
Student : same CNNBase architecture as Model A.

Distillation uses two losses:
  1. Feature-level distillation: MSE between student and teacher intermediate
     activations after global average pooling (conv2_x/layer1 and conv3_x/layer2).
     Dimension already matches (64 and 128), so no projector is needed.
  2. Hard-label cross-entropy on the 10-class classification task.

Combined loss = alpha * CE + beta * feat_loss1 + gamma * feat_loss2
"""
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as tv_models

from src.models.cnn_base import CNNBase


class CNNDistilledModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        alpha: float = 0.5,   # weight for CE loss
        beta: float = 0.3,    # weight for conv2_x feature distillation
        gamma: float = 0.2,   # weight for conv3_x feature distillation
    ):
        super().__init__()
        self.save_hyperparameters()

        self.student = CNNBase(num_classes=num_classes, hidden_dim=hidden_dim)
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_feat = nn.MSELoss()
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Frozen teacher — pretrained ResNet-18
        teacher = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        for p in teacher.parameters():
            p.requires_grad = False
        self.teacher = teacher

        # Hooks to capture intermediate teacher features
        self._teacher_feats: Dict[str, torch.Tensor] = {}
        self.teacher.layer1.register_forward_hook(self._hook("layer1"))
        self.teacher.layer2.register_forward_hook(self._hook("layer2"))

        # Storage for test results
        self._test_embeddings: List[torch.Tensor] = []
        self._test_anomaly_labels: List[torch.Tensor] = []
        self._test_class_labels: List[torch.Tensor] = []

    def _hook(self, name: str):
        def fn(module, input, output):
            self._teacher_feats[name] = output
        return fn

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        return self.student(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.student.get_embedding(x)

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        imgs, class_labels, _ = batch

        # Student forward with intermediate features
        student_logits, embedding, feat1_s, feat2_s = self.student(imgs, return_intermediates=True)

        # Teacher forward (frozen — gradients not computed)
        with torch.no_grad():
            self.teacher(imgs)
        feat1_t = self._teacher_feats["layer1"]  # 64 x H x W (same spatial as student feat1)
        feat2_t = self._teacher_feats["layer2"]  # 128 x H x W

        # Global-average-pool features before comparing
        s1 = self.gap(feat1_s).flatten(1)  # 64-dim
        t1 = self.gap(feat1_t).flatten(1)
        s2 = self.gap(feat2_s).flatten(1)  # 128-dim
        t2 = self.gap(feat2_t).flatten(1)

        feat_loss1 = self.criterion_feat(s1, t1)
        feat_loss2 = self.criterion_feat(s2, t2)
        ce_loss = self.criterion_ce(student_logits, class_labels)

        loss = (
            self.hparams.alpha * ce_loss
            + self.hparams.beta * feat_loss1
            + self.hparams.gamma * feat_loss2
        )

        acc = (student_logits.argmax(1) == class_labels).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_ce_loss", ce_loss, on_step=False, on_epoch=True)
        self.log("train_feat_loss", feat_loss1 + feat_loss2, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        imgs, class_labels, _ = batch
        logits, _ = self.student(imgs)
        loss = self.criterion_ce(logits, class_labels)
        acc = (logits.argmax(1) == class_labels).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        imgs, class_labels, anomaly_labels = batch
        logits, embeddings = self.student(imgs)
        loss = self.criterion_ce(logits, class_labels)
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
            self.student.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        t_max = self.trainer.max_epochs if (self.trainer and self.trainer.max_epochs) else 50
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
