import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    使用平均池化和最大池化两种方式生成通道注意力权重
    """

    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: 输入特征图的通道数
            reduction: 降维比例，用于构建共享MLP
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        self.sigmoid = nn.Sigmoid()

        # 共享的MLP（多层感知机），先降维再恢复
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            应用通道注意力权重后的特征图 (B, C, H, W)
        """
        # 平均池化路径
        avg_out = self.avg_pool(x)  # (B, C, 1, 1)
        avg_out = avg_out.view(avg_out.size(0), -1)  # (B, C)
        avg_out = self.mlp(avg_out)  # (B, C)

        # 最大池化路径
        max_out = self.max_pool(x)  # (B, C, 1, 1)
        max_out = max_out.view(max_out.size(0), -1)  # (B, C)
        max_out = self.mlp(max_out)  # (B, C)

        # 将两种池化结果相加并通过Sigmoid得到注意力权重
        attention_weights = avg_out + max_out
        attention_weights = self.sigmoid(attention_weights)
        attention_weights = attention_weights.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)

        # 应用注意力权重
        return x * attention_weights


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    使用平均池化和最大池化两种方式生成空间注意力权重
    """

    def __init__(self, kernel_size=7):
        """
        Args:
            kernel_size: 卷积核大小，用于生成空间注意力图
        """
        super(SpatialAttention, self).__init__()
        # 在通道维度上进行平均池化和最大池化
        self.avg_pool = torch.mean  # 通道维度平均池化
        self.max_pool = torch.max  # 通道维度最大池化

        # 卷积层：将拼接后的特征图转换为空间注意力图
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            应用空间注意力权重后的特征图 (B, C, H, W)
        """
        # 平均池化路径：在通道维度上取平均值
        avg_out = self.avg_pool(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # 最大池化路径：在通道维度上取最大值
        max_out, _ = self.max_pool(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # 拼接两种池化结果
        x_concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)

        # 通过卷积层生成空间注意力图
        attention_map = self.conv(x_concat)  # (B, 1, H, W)

        # 应用Sigmoid得到空间注意力权重
        attention_weights = self.sigmoid(attention_map)  # (B, 1, H, W)

        # 应用注意力权重
        return x * attention_weights


class MultiAttention(nn.Module):
    """
    多头注意力模块
    组合通道注意力和空间注意力，按顺序应用
    用于Net网络调用
    """
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        """
        Args:
            channels: 输入特征图的通道数
            reduction: 通道注意力的降维比例，用于构建共享MLP
            spatial_kernel_size: 空间注意力的卷积核大小
        """
        super(MultiAttention, self).__init__()
        
        # 通道注意力模块
        self.channel_attention = ChannelAttention(channels=channels, reduction=reduction)
        
        # 空间注意力模块
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            应用通道和空间注意力后的特征图 (B, C, H, W)
        """
        # 先应用通道注意力
        x = self.channel_attention(x)  # (B, C, H, W)
        
        # 再应用空间注意力
        x = self.spatial_attention(x)  # (B, C, H, W)
        
        return x

