import torch
import torch.nn as nn


class CoordinateConv(nn.Module):
    """
    坐标卷积模块
    在特征图中添加坐标信息（x坐标和y坐标），帮助网络学习位置相关的特征
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            groups: 分组卷积的组数
            bias: 是否使用偏置
        """
        super(CoordinateConv, self).__init__()
        # 坐标卷积：输入通道数+2（x坐标和y坐标），输出通道数
        self.conv = nn.Conv2d(
            in_channels + 2,  # 原始通道数 + 2个坐标通道
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            输出特征图 (B, out_channels, H', W')
        """
        batch_size, channels, height, width = x.size()
        
        # 生成x坐标图：从-1到1的归一化坐标
        x_coords = torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype)
        x_coords = x_coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, W)
        x_coords = x_coords.expand(batch_size, 1, height, width)  # (B, 1, H, W)
        
        # 生成y坐标图：从-1到1的归一化坐标
        y_coords = torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype)
        y_coords = y_coords.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, H, 1)
        y_coords = y_coords.expand(batch_size, 1, height, width)  # (B, 1, H, W)
        
        # 将坐标信息与输入特征图拼接
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)  # (B, C+2, H, W)
        
        # 进行卷积操作
        output = self.conv(x_with_coords)  # (B, out_channels, H', W')
        
        return output


class CoordinateConv2d(nn.Module):
    """
    坐标卷积2D模块（简化版本）
    可以直接替代nn.Conv2d使用，自动添加坐标信息
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            groups: 分组卷积的组数
            bias: 是否使用偏置
        """
        super(CoordinateConv2d, self).__init__()
        self.coordinate_conv = CoordinateConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            输出特征图 (B, out_channels, H', W')
        """
        return self.coordinate_conv(x)

