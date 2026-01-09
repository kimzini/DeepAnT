# Input -> Conv1 -> ReLU -> MaxPool -> Conv2 -> ReLU -> MaxPool -> (flatten) -> FC
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class PredictorParams:
    w: int  # history window
    conv_kernel_size: int
    p_w: int    # predicted window size
    feature_dim: int
    conv_channels: int
    pool_kernel_size: int

class TimeSeriesPredictor(nn.Module):
    def __init__(self, cfg: PredictorParams):
        super().__init__()
        self.cfg = cfg

        k = cfg.conv_kernel_size // 2 # 각 시점에서 양쪽 k개만 볼 수 있도록 설정

        self.conv1 = nn.Conv1d(cfg.feature_dim, cfg.conv_channels, kernel_size=cfg.conv_kernel_size, padding=k)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=cfg.pool_kernel_size)

        self.conv2 = nn.Conv1d(cfg.conv_channels, cfg.conv_channels, kernel_size=cfg.conv_kernel_size, padding=k)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=cfg.pool_kernel_size)

        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(cfg.feature_dim * cfg.p_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, w)
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)

        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)

        y = self.flatten(y)
        y = self.fc(y)
        y = y.view(x.size(0), self.cfg.feature_dim, self.cfg.p_w)
        # y: (B, C, p_w)
        return y