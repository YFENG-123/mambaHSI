import torch.nn as nn


class MultiScaleAsymmetricDepthConv(nn.Module):
    """
    多尺度非对称深度可分离卷积模块
    使用13、15、17、19、21四种大小的非对称深度可分离卷积核进行深度可分离卷积
    """
    def __init__(self, channels, dropout_rate=0.5):
        """
        Args:
            channels: 输入特征图的通道数
            dropout_rate: Dropout比率
        """
        super(MultiScaleAsymmetricDepthConv, self).__init__()
        self.channels = channels
        
        # 13: 1x13和13x1非对称深度可分离卷积对
        self.dconv1_13 = nn.Conv2d(channels, channels, kernel_size=(1, 13), padding=(0, 6), groups=channels)
        self.dconv13_1 = nn.Conv2d(channels, channels, kernel_size=(13, 1), padding=(6, 0), groups=channels)
        
        # 15: 1x15和15x1非对称深度可分离卷积对
        self.dconv1_15 = nn.Conv2d(channels, channels, kernel_size=(1, 15), padding=(0, 7), groups=channels)
        self.dconv15_1 = nn.Conv2d(channels, channels, kernel_size=(15, 1), padding=(7, 0), groups=channels)
        
        # 17: 1x17和17x1非对称深度可分离卷积对
        self.dconv1_17 = nn.Conv2d(channels, channels, kernel_size=(1, 17), padding=(0, 8), groups=channels)
        self.dconv17_1 = nn.Conv2d(channels, channels, kernel_size=(17, 1), padding=(8, 0), groups=channels)
        
        # 19: 1x19和19x1非对称深度可分离卷积对
        self.dconv1_19 = nn.Conv2d(channels, channels, kernel_size=(1, 19), padding=(0, 9), groups=channels)
        self.dconv19_1 = nn.Conv2d(channels, channels, kernel_size=(19, 1), padding=(9, 0), groups=channels)
        
        # 21: 1x21和21x1非对称深度可分离卷积对
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        
        # 1x1卷积用于特征融合
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        # BatchNorm和激活函数
        self.bn = nn.BatchNorm2d(channels)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            融合后的多尺度特征 (B, C, H, W)
        """
        # 13: 1x13和13x1非对称深度可分离卷积对
        x_1_13 = self.dconv1_13(x)  # (B, C, H, W)
        x_13_1 = self.dconv13_1(x)  # (B, C, H, W)
        
        # 15: 1x15和15x1非对称深度可分离卷积对
        x_1_15 = self.dconv1_15(x)  # (B, C, H, W)
        x_15_1 = self.dconv15_1(x)  # (B, C, H, W)
        
        # 17: 1x17和17x1非对称深度可分离卷积对
        x_1_17 = self.dconv1_17(x)  # (B, C, H, W)
        x_17_1 = self.dconv17_1(x)  # (B, C, H, W)
        
        # 19: 1x19和19x1非对称深度可分离卷积对
        x_1_19 = self.dconv1_19(x)  # (B, C, H, W)
        x_19_1 = self.dconv19_1(x)  # (B, C, H, W)
        
        # 21: 1x21和21x1非对称深度可分离卷积对
        x_1_21 = self.dconv1_21(x)  # (B, C, H, W)
        x_21_1 = self.dconv21_1(x)  # (B, C, H, W)
        
        # 将所有多尺度特征相加
        x_sum = x_1_13 + x_13_1 + x_1_15 + x_15_1 + x_1_17 + x_17_1 + x_1_19 + x_19_1 + x_1_21 + x_21_1  # (B, C, H, W)
        
        # 通过1x1卷积进行特征融合
        x_fusion = self.conv(x_sum)  # (B, C, H, W)
        
        # BatchNorm、激活和Dropout
        x_fusion = self.bn(x_fusion)
        x_fusion = self.gelu(x_fusion)
        x_fusion = self.dropout(x_fusion)
        
        return x_fusion

