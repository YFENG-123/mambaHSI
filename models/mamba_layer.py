"""
mamba_layer

封装 Mamba 层：将双向 S6Core 与 Pre-Norm + Dropout + 残差连接组合为一个可复用模块。

设计目标：
- 对输入序列同时执行正向与反向的 S6Core 处理以捕获双向信息。
- 在 S6Core 前使用 LayerNorm（Pre-Norm）以提升训练稳定性。
- 在融合后使用 Dropout 并保留残差以防止过拟合与信息丢失。

接口说明：
- 输入 x: 张量，形状 (batch, seq_len, d_model)
- 输出 y: 张量，形状 (batch, seq_len, d_model)

注意事项：
- 本模块依赖 `S6Core` 实现；确保 `S6Core` 的输入输出与本模块的 d_model 匹配。
"""

import torch
import torch.nn as nn

from .s6_core import S6Core


class MambaLayer(nn.Module):
    """
    Mamba 层：将双向 S6Core 与 Pre-Norm + Dropout + 残差连接组合为一个可复用模块。
    """

    def __init__(self, d_model: int = 64, dropout_rate: float = 0.3):
        super().__init__()
        self.d_model = d_model

        # 双向 S6Core（前向 + 反向）
        self.mamba_fwd = S6Core(d_model=d_model)
        self.mamba_bwd = S6Core(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保留残差
        residual = x

        # 双向处理：前向 + 翻转的后向
        x_fwd = self.mamba_fwd(x)
        x_bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])

        # 融合并应用 dropout + 残差
        x_mamba = x_fwd + x_bwd
        x_mamba = self.norm(x_mamba)
        x_mamba = self.gelu(x_mamba)
        x_mamba = self.dropout(x_mamba)
        x_mamba = x_mamba + residual
        x_mamba = self.norm(x_mamba)
        x_mamba = self.gelu(x_mamba)
        x_mamba = self.dropout(x_mamba)
        return x_mamba
