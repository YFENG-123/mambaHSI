import torch
import torch.nn as nn


class MultiScaleAsymmetricDepthConv(nn.Module):
    """
    多尺度非对称深度可分离卷积模块
    使用可配置大小的非对称深度可分离卷积核进行深度可分离卷积
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=None, dropout_rate=0.5):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_sizes: 卷积核大小列表，默认为[13, 15, 17, 19, 21]
            dropout_rate: Dropout比率
        """
        super(MultiScaleAsymmetricDepthConv, self).__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [13, 15, 17, 19, 21]
        
        self.kernel_sizes = kernel_sizes
        
        # 为每个卷积核大小创建非对称深度可分离卷积对
        self.dconv1_k_list = nn.ModuleList()  # 存储1xk卷积
        self.dconvk_1_list = nn.ModuleList()  # 存储kx1卷积
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            # 1xk非对称深度可分离卷积
            self.dconv1_k_list.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), padding=(0, padding), groups=in_channels)
            )
            # kx1非对称深度可分离卷积
            self.dconvk_1_list.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), padding=(padding, 0), groups=in_channels)
            )
        
        # 1x1卷积压缩层：将拼接后的特征压缩到out_channels
        intermediate_channels = in_channels * len(kernel_sizes) * 2
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
        # 对每个卷积核大小进行非对称深度可分离卷积
        multi_scale_features = []
        for dconv1_k, dconvk_1 in zip(self.dconv1_k_list, self.dconvk_1_list):
            # 1xk和kx1非对称深度可分离卷积对
            x_1_k = dconv1_k(x)  # (B, in_channels, H, W)
            x_k_1 = dconvk_1(x)  # (B, in_channels, H, W)
            multi_scale_features.append(x_1_k)
            multi_scale_features.append(x_k_1)
        
        # 拼接所有多尺度特征
        x_concat = torch.cat(multi_scale_features, dim=1)  # (B, in_channels*len(kernel_sizes)*2, H, W)
        
        # 压缩到out_channels
        x_out = self.compress(x_concat)  # (B, out_channels, H, W)
        
        return x_out

