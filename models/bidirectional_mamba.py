import torch
import torch.nn as nn
from .s6_core import S6Core


class BidirectionalMamba(nn.Module):
    """
    双向Mamba模块
    
    实现双向扫描以捕获全图上下文：
    - 前向流：使用S6Core处理原始序列
    - 后向流：翻转输入 -> S6Core处理 -> 翻转输出
    - 融合：前向和后向结果相加
    
    使用Pre-Norm Residual结构以提高训练稳定性。
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
        
        # Pre-Norm结构
        self.mamba_norm = nn.LayerNorm(d_model)
        self.mamba_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 (batch, seq_len, d_model)
        Returns:
            输出特征 (batch, seq_len, d_model)
        """
        # Pre-Norm Residual Block
        residual = x
        x_norm = self.mamba_norm(x)
        
        # 前向流
        x_fwd = self.mamba_fwd(x_norm)
        
        # 后向流 (翻转输入 -> 处理 -> 翻转输出)
        x_bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        
        # 融合: 相加
        x_mamba = x_fwd + x_bwd
        
        # Dropout和残差连接
        x_mamba = self.mamba_dropout(x_mamba)
        x_mamba = x_mamba + residual
        
        return x_mamba

