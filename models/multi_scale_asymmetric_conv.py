import torch
import torch.nn as nn


class MultiScaleAsymmetricDepthConv(nn.Module):
    """
    多尺度非对称深度可分离卷积模块
    使用可配置大小的非对称深度可分离卷积核进行深度可分离卷积
    """
    def __init__(self, channels, kernel_sizes=None, dropout_rate=0.5):
        """
        Args:
            channels: 输入特征图的通道数
            kernel_sizes: 卷积核大小列表，默认为[13, 15, 17, 19, 21]
            dropout_rate: Dropout比率
        """
        super(MultiScaleAsymmetricDepthConv, self).__init__()
        self.channels = channels
        
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
                nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, padding), groups=channels)
            )
            # kx1非对称深度可分离卷积
            self.dconvk_1_list.append(
                nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(padding, 0), groups=channels)
            )
        
        # 不进行融合，直接返回拼接的多尺度特征

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            拼接后的多尺度特征 (B, C*len(kernel_sizes)*2, H, W) - 保留所有非对称卷积对
        """
        # 对每个卷积核大小进行非对称深度可分离卷积
        multi_scale_features = []
        for dconv1_k, dconvk_1 in zip(self.dconv1_k_list, self.dconvk_1_list):
            # 1xk和kx1非对称深度可分离卷积对
            x_1_k = dconv1_k(x)  # (B, C, H, W)
            x_k_1 = dconvk_1(x)  # (B, C, H, W)
            multi_scale_features.append(x_1_k)
            multi_scale_features.append(x_k_1)
        
        # 拼接所有多尺度特征，不进行融合
        x_concat = torch.cat(multi_scale_features, dim=1)  # (B, C*len(kernel_sizes)*2, H, W)
        
        return x_concat

