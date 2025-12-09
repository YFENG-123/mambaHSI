import torch
import torch.nn as nn


class DepthwiseSeparableASPP(nn.Module):
    """
    深度可分离ASPP模块
    将ASPP中的普通卷积替换为深度可分离卷积
    输入输出通道数保持不变（深度可分离状态）
    """
    def __init__(self, channels, dilations=[1, 6, 12, 18], dropout_rate=0.1):
        """
        Args:
            channels: 输入输出通道数（深度可分离，通道数不变）
            dilations: 膨胀率列表，默认为[1, 6, 12, 18]
            dropout_rate: Dropout比率
        """
        super(DepthwiseSeparableASPP, self).__init__()
        
        # 多个不同膨胀率的3x3深度可分离卷积分支（保持通道数不变）
        self.dilated_convs = nn.ModuleList()
        for dilation in dilations:
            padding = dilation
            # 深度卷积（不进行norm和激活，保持原始特征）
            conv = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False
            )
            self.dilated_convs.append(conv)
        
        # 不进行融合，直接返回拼接的多尺度特征

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            拼接后的多尺度特征 (B, C*len(dilations), H, W) - 保留所有膨胀卷积分支
        """
        # 多个膨胀卷积分支
        dilated_features = []
        for dilated_conv in self.dilated_convs:
            feature = dilated_conv(x)  # (B, C, H, W)
            dilated_features.append(feature)
        
        # 拼接所有膨胀卷积分支，不进行融合
        x_concat = torch.cat(dilated_features, dim=1)  # (B, C*len(dilations), H, W)
        
        return x_concat

