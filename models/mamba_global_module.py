"""
Mamba全局建模模块
使用双向Mamba进行全局上下文建模
"""

import torch
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
        
        # 第一层：双向Mamba核心
        self.mamba_fwd_1 = S6Core(d_model=d_model)
        self.mamba_bwd_1 = S6Core(d_model=d_model)
        self.norm_1 = nn.LayerNorm(d_model)
        
        # 前馈网络：增强特征表达能力
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=True),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model, bias=True),
            nn.Dropout(dropout_rate),
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        
        # 第二层：双向Mamba核心（增强全局建模）
        self.mamba_fwd_2 = S6Core(d_model=d_model)
        self.mamba_bwd_2 = S6Core(d_model=d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        
        # 可学习的残差连接权重
        self.alpha_1 = nn.Parameter(torch.tensor(0.5))  # 第一层权重
        self.alpha_2 = nn.Parameter(torch.tensor(0.5))  # 第二层权重
        
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
        
        # 残差连接
        residual = x_seq
        
        # 第一层：双向Mamba处理
        x_norm_1 = self.norm_1(x_seq)
        x_fwd_1 = self.mamba_fwd_1(x_norm_1)
        x_bwd_1 = self.mamba_bwd_1(x_norm_1.flip(dims=[1])).flip(dims=[1])
        x_mamba_1 = x_fwd_1 + x_bwd_1
        x_mamba_1 = self.dropout(x_mamba_1)
        x_mamba_1 = x_seq + self.alpha_1 * x_mamba_1  # 可学习权重的残差连接
        
        # 前馈网络：增强特征表达能力
        x_ffn = self.norm_ffn(x_mamba_1)
        x_ffn = self.ffn(x_ffn)
        x_ffn = x_mamba_1 + x_ffn  # 残差连接
        
        # 第二层：双向Mamba处理（增强全局建模）
        x_norm_2 = self.norm_2(x_ffn)
        x_fwd_2 = self.mamba_fwd_2(x_norm_2)
        x_bwd_2 = self.mamba_bwd_2(x_norm_2.flip(dims=[1])).flip(dims=[1])
        x_mamba_2 = x_fwd_2 + x_bwd_2
        x_mamba_2 = self.dropout(x_mamba_2)
        x_mamba_2 = x_ffn + self.alpha_2 * x_mamba_2  # 可学习权重的残差连接
        
        # 移除batch维度
        x_mamba = x_mamba_2.squeeze(0)  # (H*W, d_model)
        
        return x_mamba

