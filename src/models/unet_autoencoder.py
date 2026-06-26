"""Model C — U-Net autoencoder trained from scratch.

Reconstructs normal training images. The bottleneck embedding (after global
average pool) is used for anomaly detection via Mahalanobis distance.
Reconstruction error (MSE) can also serve as an anomaly score.
"""
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3, features: List[int] = None, embedding_dim: int = 256):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder path
        in_ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(in_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = f

        # Bottleneck: features[-1] → features[-1]*2
        bottleneck_ch = features[-1] * 2
        self.bottleneck = DoubleConv(features[-1], bottleneck_ch)

        # Embedding projection from bottleneck
        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_ch, embedding_dim),
        )

        # Decoder path
        in_ch = bottleneck_ch
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(in_ch, f, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))
            in_ch = f

        self.final_conv = nn.Conv2d(features[0], in_channels, kernel_size=1)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        for enc, pool in zip(self.encoders, self.pools):
            x = pool(enc(x))
        bottleneck = self.bottleneck(x)
        return self.embedding_head(bottleneck)

    def forward(self, x: torch.Tensor):
        skip_connections = []

        # Encode
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        embedding = self.embedding_head(x)

        # Decode with skip connections
        skip_connections = skip_connections[::-1]
        for upconv, dec, skip in zip(self.upconvs, self.decoders, skip_connections):
            x = upconv(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        reconstruction = self.final_conv(x)
        return reconstruction, embedding


class UNetAutoencoderModule(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        features: List[int] = None,
        embedding_dim: int = 256,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        if features is None:
            features = [64, 128, 256, 512]

        self.model = UNetAutoencoder(
            in_channels=in_channels,
            features=features,
            embedding_dim=embedding_dim,
        )

        # Storage for test results
        self._test_embeddings: List[torch.Tensor] = []
        self._test_anomaly_labels: List[torch.Tensor] = []
        self._test_class_labels: List[torch.Tensor] = []
        self._test_recon_errors: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        return self.model(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_embedding(x)

    # ------------------------------------------------------------------
    def _recon_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, target)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        imgs, _, _ = batch
        recon, _ = self(imgs)
        loss = self._recon_loss(recon, imgs)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        imgs, _, _ = batch
        recon, _ = self(imgs)
        loss = self._recon_loss(recon, imgs)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx) -> Dict[str, Any]:
        imgs, class_labels, anomaly_labels = batch
        recon, embeddings = self(imgs)
        recon_error = F.mse_loss(recon, imgs, reduction="none").mean(dim=(1, 2, 3))
        self.log("test_recon_loss", recon_error.mean(), on_step=False, on_epoch=True)
        self._test_embeddings.append(embeddings.detach().cpu())
        self._test_anomaly_labels.append(anomaly_labels.detach().cpu())
        self._test_class_labels.append(class_labels.detach().cpu())
        self._test_recon_errors.append(recon_error.detach().cpu())
        return {
            "embeddings": embeddings,
            "anomaly_labels": anomaly_labels,
            "recon_error": recon_error,
        }

    def on_test_epoch_end(self) -> None:
        self._test_embeddings = []
        self._test_anomaly_labels = []
        self._test_class_labels = []
        self._test_recon_errors = []

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"},
        }
