"""Shared ResNet-18 inspired CNN backbone (conv1, conv2_x, conv3_x) + FC head."""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class CNNBase(nn.Module):
    """
    ResNet-18 first 3 conv stages (conv1, conv2_x, conv3_x) followed by
    a global average pool and a 2-layer FC classifier.

    The 128-dim vector after GAP is the embedding used for anomaly detection.
    The classifier output is used for training supervision.
    """

    EMBEDDING_DIM = 128  # fixed by conv3_x output channels

    def __init__(self, num_classes: int = 10, hidden_dim: int = 256):
        super().__init__()
        # conv1: 7x7, stride 2 + max pool stride 2
        # input 3x256x256 → 64x64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        # conv2_x: 2x BasicBlock(64, 64) → 64x64x64
        self.conv2_x = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )
        # conv3_x: BasicBlock(64→128, stride 2) + BasicBlock(128, 128) → 128x32x32
        self.conv3_x = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier on top of the 128-dim embedding
        self.classifier = nn.Sequential(
            nn.Linear(self.EMBEDDING_DIM, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        return self.gap(x).flatten(1)

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        x = self.conv1(x)
        feat1 = self.conv2_x(x)   # 64 x H/4 x W/4
        feat2 = self.conv3_x(feat1)  # 128 x H/8 x W/8
        embedding = self.gap(feat2).flatten(1)  # 128-dim
        logits = self.classifier(embedding)
        if return_intermediates:
            return logits, embedding, feat1, feat2
        return logits, embedding
