"""
Mamba全局建模模块
使用双向Mamba进行全局上下文建模
"""

import torch
import torch.nn as nn
from .s6_core import S6Core


class MambaGlobalModule(nn.Module):
    """
    Mamba全局建模模块
    
    功能：
    - 使用双向Mamba（前向+后向）进行全局上下文建模
    - 通过残差连接保持梯度流动
    """
    
    def __init__(
        self,
        d_model=64,
        dropout_rate=0.5,
    ):
        """
        Args:
            d_model: 特征维度
            dropout_rate: Dropout比率
        """
        super().__init__()
        self.d_model = d_model
        
        # 双向Mamba核心
        self.mamba_fwd = S6Core(d_model=d_model)
        self.mamba_bwd = S6Core(d_model=d_model)
        
        # 归一化和Dropout
        self.mamba_norm = nn.LayerNorm(d_model)
        self.mamba_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x_fused):
        """
        Args:
            x_fused: 融合后的特征 (H, W, d_model)
        Returns:
            x_mamba: Mamba处理后的特征 (H*W, d_model)
        """
        # 转换为序列格式
        x_seq = x_fused.reshape(-1, self.d_model).unsqueeze(0)  # (1, H*W, d_model)
        
        # 残差连接
        residual = x_seq
        
        # 归一化
        x_norm = self.mamba_norm(x_seq)
        
        # 双向Mamba处理
        x_fwd = self.mamba_fwd(x_norm)
        x_bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        x_mamba = x_fwd + x_bwd
        
        # Dropout和残差连接
        x_mamba = self.mamba_dropout(x_mamba)
        x_mamba = x_mamba + residual
        
        # 移除batch维度
        x_mamba = x_mamba.squeeze(0)  # (H*W, d_model)
        
        return x_mamba

