"""
fusion module

封装模型中的特征融合层（1x1 卷积），便于复用与单独调优。
默认行为与原来的直接 1x1 Conv 等价（conv-only），可通过参数启用归一化/激活/Dropout。
"""

import torch.nn as nn


class FusionModule(nn.Module):
    """
    融合模块：将两个分支的特征融合在一起
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.dropout_rate = dropout_rate

        self.fusion = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=self.bias),
            nn.GroupNorm(1, self.out_channels),
            nn.GELU(),
            nn.Dropout2d(self.dropout_rate),
        )

    def forward(self, x):
        x = self.fusion(x)
        return x
