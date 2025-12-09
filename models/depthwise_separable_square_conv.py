import torch
import torch.nn as nn


class DepthwiseSeparableSquareConv(nn.Module):
    """
    深度可分离方形卷积模块
    使用可配置大小的深度可分离卷积
    输入输出通道数保持不变（深度可分离状态）
    """
    def __init__(self, channels, kernel_sizes=None, dropout_rate=0.5):
        """
        Args:
            channels: 输入输出通道数（深度可分离，通道数不变）
            kernel_sizes: 卷积核大小列表，默认为[3, 5, 7]
            dropout_rate: Dropout比率
        """
        super(DepthwiseSeparableSquareConv, self).__init__()
        self.channels = channels
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        
        self.kernel_sizes = kernel_sizes
        
        # 为每个卷积核大小创建深度可分离卷积
        self.square_convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
            self.square_convs.append(conv)

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            拼接后的多尺度特征 (B, C*len(kernel_sizes), H, W)
        """
        # 对每个卷积核大小进行深度可分离卷积
        multi_scale_features = []
        for conv in self.square_convs:
            feature = conv(x)  # (B, C, H, W)
            multi_scale_features.append(feature)
        
        # 通道拼接
        x_concat = torch.cat(multi_scale_features, dim=1)  # (B, C*len(kernel_sizes), H, W)
        
        return x_concat

