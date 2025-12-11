import torch
import torch.nn as nn


class StripConvolution(nn.Module):
    """
    条状卷积模块（Strip Convolution）
    用于提取长距离的空间依赖关系，特别适合高光谱图像中的条状结构特征
    包括水平条状卷积和垂直条状卷积
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, dropout_rate=0.1):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 条状卷积核大小，默认为7
            dropout_rate: Dropout比率
        """
        super(StripConvolution, self).__init__()
        
        # 水平条状卷积：1xk，提取水平方向的长距离依赖
        self.horizontal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=(1, kernel_size), 
                stride=1, 
                padding=(0, kernel_size // 2),
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        # 垂直条状卷积：kx1，提取垂直方向的长距离依赖
        self.vertical_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=(kernel_size, 1), 
                stride=1, 
                padding=(kernel_size // 2, 0),
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        # 融合层：将水平和垂直特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, in_channels, H, W)
        Returns:
            融合后的特征 (B, out_channels, H, W)
        """
        # 水平条状卷积
        h_feat = self.horizontal_conv(x)  # (B, out_channels, H, W)
        
        # 垂直条状卷积
        v_feat = self.vertical_conv(x)  # (B, out_channels, H, W)
        
        # 拼接水平和垂直特征
        combined = torch.cat([h_feat, v_feat], dim=1)  # (B, out_channels*2, H, W)
        
        # 融合
        output = self.fusion(combined)  # (B, out_channels, H, W)
        
        return output


class MultiScaleStripConvolution(nn.Module):
    """
    多尺度条状卷积模块
    使用多个不同大小的条状卷积核提取不同尺度的长距离空间依赖
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[5, 7, 9], dropout_rate=0.1):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_sizes: 条状卷积核大小列表，默认为[5, 7, 9]
            dropout_rate: Dropout比率
        """
        super(MultiScaleStripConvolution, self).__init__()
        
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        
        # 计算每个尺度的输出通道数，确保总和等于out_channels
        channels_per_scale = out_channels // self.num_scales
        remainder = out_channels % self.num_scales
        
        # 为每个尺度创建条状卷积
        self.strip_convs = nn.ModuleList()
        for i, kernel_size in enumerate(kernel_sizes):
            # 将余数分配给前几个尺度
            scale_out_channels = channels_per_scale + (1 if i < remainder else 0)
            self.strip_convs.append(
                StripConvolution(in_channels, scale_out_channels, kernel_size, dropout_rate)
            )
        
        # 计算实际拼接后的通道数
        actual_combined_channels = sum(
            channels_per_scale + (1 if i < remainder else 0) 
            for i in range(self.num_scales)
        )
        
        # 融合多尺度特征（输入通道数为实际拼接后的通道数）
        self.fusion = nn.Sequential(
            nn.Conv2d(actual_combined_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, in_channels, H, W)
        Returns:
            多尺度融合后的特征 (B, out_channels, H, W)
        """
        # 提取多尺度特征
        multi_scale_features = []
        for strip_conv in self.strip_convs:
            feat = strip_conv(x)  # (B, out_channels//len(kernel_sizes), H, W)
            multi_scale_features.append(feat)
        
        # 拼接多尺度特征
        combined = torch.cat(multi_scale_features, dim=1)  # (B, out_channels, H, W)
        
        # 融合
        output = self.fusion(combined)  # (B, out_channels, H, W)
        
        return output

