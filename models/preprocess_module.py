"""
预处理模块
负责对输入的高光谱图像进行归一化处理
"""

import torch
import torch.nn as nn


class PreprocessModule(nn.Module):
    """
    预处理模块
    
    功能：
    - 对输入的高光谱图像进行LayerNorm归一化
    - 为后续的特征提取做准备
    """
    
    def __init__(self, bands=200):
        """
        Args:
            bands: 光谱波段数
        """
        super().__init__()
        self.preprocess = nn.LayerNorm(bands)
    
    def forward(self, x):
        """
        Args:
            x: 输入高光谱图像 (H, W, bands)
        Returns:
            归一化后的图像 (H, W, bands)
        """
        return self.preprocess(x)

