import torch
import torch.nn as nn


class DepthwiseSeparableSquareConv(nn.Module):
    """
    深度可分离方形卷积模块
    使用可配置大小的深度可分离卷积
    输入输出通道数保持不变（深度可分离状态）
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=None, dropout_rate=0.5):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_sizes: 卷积核大小列表，默认为[3, 5, 7]
            dropout_rate: Dropout比率
        """
        super(DepthwiseSeparableSquareConv, self).__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
        
        self.kernel_sizes = kernel_sizes
        
        # 为每个卷积核大小创建深度可分离卷积
        self.square_convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            # 深度可分离卷积（不进行norm和dropout，保持原始特征）
            conv = nn.Conv2d(
                in_channels, 
                in_channels, 
                kernel_size=kernel_size, 
                stride=1, 
                padding=padding, 
                groups=in_channels, 
                bias=False
            )
            self.square_convs.append(conv)
        
        # 1x1卷积压缩层：将拼接后的特征压缩到out_channels
        intermediate_channels = in_channels * len(kernel_sizes)
        self.compress = nn.Sequential(
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, in_channels, H, W)
        Returns:
            压缩后的特征 (B, out_channels, H, W)
        """
        # 对每个卷积核大小进行深度可分离卷积
        multi_scale_features = []
        for conv in self.square_convs:
            feature = conv(x)  # (B, in_channels, H, W)
            multi_scale_features.append(feature)
        
        # 通道拼接
        x_concat = torch.cat(multi_scale_features, dim=1)  # (B, in_channels*len(kernel_sizes), H, W)
        
        # 压缩到out_channels
        x_out = self.compress(x_concat)  # (B, out_channels, H, W)
        
        return x_out

