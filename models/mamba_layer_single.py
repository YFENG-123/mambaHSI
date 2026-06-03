"""
mamba_layer_single

单向 Mamba 层：仅使用前向 S6Core，与 Pre-Norm + Dropout + 残差连接组合。

设计目标：
- 仅对输入序列执行正向的 S6Core 处理（单向）
- 用于消融实验，对比双向 Mamba 的效果
- 在 S6Core 前使用 LayerNorm（Pre-Norm）以提升训练稳定性

接口说明：
- 输入 x: 张量，形状 (batch, seq_len, d_model)
- 输出 y: 张量，形状 (batch, seq_len, d_model)
"""

import torch
import torch.nn as nn

from .s6_core import S6Core


class MambaLayerSingle(nn.Module):
    """
    单向 Mamba 层：仅使用前向 S6Core，用于消融实验。
    """

    def __init__(self, d_model: int = 64, dropout_rate: float = 0.3):
        super().__init__()
        self.d_model = d_model

        # 单向 S6Core（仅前向）
        self.mamba_fwd = S6Core(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保留残差
        residual = x

        # 单向处理：仅前向
        x_mamba = self.mamba_fwd(x)
        
        # 应用 norm + dropout + 残差
        x_mamba = self.norm(x_mamba)
        x_mamba = self.gelu(x_mamba)
        x_mamba = self.dropout(x_mamba)

        x_mamba = x_mamba + residual
        x_mamba = self.norm(x_mamba)
        x_mamba = self.gelu(x_mamba)
        x_mamba = self.dropout(x_mamba)
        return x_mamba
