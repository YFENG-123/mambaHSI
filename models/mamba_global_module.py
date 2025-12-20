"""
Mamba全局建模模块
使用双向Mamba进行全局上下文建模
"""

import torch.nn as nn
from .s6_core import S6Core


class MambaGlobalModule(nn.Module):
    """
    Mamba全局建模模块 - Enhanced Version (V4.24)
    
    改进：
    - 增强双向Mamba结构：添加多层Mamba处理，提升全局特征建模能力
    - 添加前馈网络：增强特征表达能力
    - 使用可学习权重的残差连接：平衡不同层的影响
    - 添加LayerNorm稳定训练：提升训练稳定性
    
    功能：
    - 使用双向Mamba（前向+后向）进行全局上下文建模
    - 通过多层处理和残差连接保持梯度流动
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
        
        # 单层双向Mamba核心（精简版）
        self.mamba_fwd = S6Core(d_model=d_model)
        self.mamba_bwd = S6Core(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # 移除前馈网络以进一步降低内存占用，保留核心Mamba功能
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x_fused):
        """
        Args:
            x_fused: 融合后的特征 (H, W, d_model)
        Returns:
            x_mamba: Mamba处理后的特征 (H*W, d_model)
        """
        # 转换为序列格式
        x_seq = x_fused.reshape(-1, self.d_model).unsqueeze(0)  # (1, H*W, d_model)
        
        # 双向Mamba处理（精简版：单层，移除FFN）
        x_norm = self.norm(x_seq)
        x_fwd = self.mamba_fwd(x_norm)
        x_bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        x_mamba = x_fwd + x_bwd
        x_mamba = self.dropout(x_mamba)
        x_mamba = x_seq + x_mamba  # 残差连接
        
        # 移除前馈网络以降低内存占用
        
        # 移除batch维度
        x_mamba = x_mamba.squeeze(0)  # (H*W, d_model)
        
        return x_mamba

