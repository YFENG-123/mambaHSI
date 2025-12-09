import torch
import torch.nn as nn


class MultiScaleSquareDepthConv(nn.Module):
    """
    多尺度方形深度可分离卷积模块
    包含3x3、5x5、7x7三种大小的方形深度可分离卷积
    """
    def __init__(self, channels, dropout_rate=0.5):
        """
        Args:
            channels: 输入特征图的通道数
            dropout_rate: Dropout比率
        """
        super(MultiScaleSquareDepthConv, self).__init__()
        self.channels = channels
        
        # 3x3方形深度可分离卷积
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        
        # 5x5方形深度可分离卷积
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=2, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        
        # 7x7方形深度可分离卷积
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, stride=1, padding=3, groups=channels),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            拼接后的多尺度特征 (B, C*3, H, W)
        """
        # 分别使用3x3、5x5、7x7方形深度可分离卷积
        x_conv3x3 = self.conv3x3(x)  # (B, C, H, W)
        x_conv5x5 = self.conv5x5(x)  # (B, C, H, W)
        x_conv7x7 = self.conv7x7(x)  # (B, C, H, W)
        
        # 通道拼接
        x_concat = torch.cat([x_conv3x3, x_conv5x5, x_conv7x7], dim=1)  # (B, C*3, H, W)
        
        return x_concat

