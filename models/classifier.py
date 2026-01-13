"""
classifier module

封装模型的分类头（MLP），便于独立调优与复用。
"""

import torch.nn as nn


class ClassifierModule(nn.Module):
    """
    分类器模块：Linear(in_dim -> hidden_dim) -> LayerNorm -> GELU -> Dropout -> Linear(hidden_dim -> out_dim)
    """

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, dropout_rate: float = 0.3
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(self, x):
        return self.net(x)
