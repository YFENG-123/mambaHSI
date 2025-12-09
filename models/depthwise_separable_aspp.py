import torch
import torch.nn as nn


class DepthwiseSeparableASPP(nn.Module):
    """
    深度可分离ASPP模块
    将ASPP中的普通卷积替换为深度可分离卷积
    输入输出通道数保持不变（深度可分离状态）
    """
    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18], dropout_rate=0.1):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
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
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False
            )
            self.dilated_convs.append(conv)
        
        # 1x1卷积压缩层：将拼接后的特征压缩到out_channels
        intermediate_channels = in_channels * len(dilations)
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
        # 多个膨胀卷积分支
        dilated_features = []
        for dilated_conv in self.dilated_convs:
            feature = dilated_conv(x)  # (B, in_channels, H, W)
            dilated_features.append(feature)
        
        # 拼接所有膨胀卷积分支
        x_concat = torch.cat(dilated_features, dim=1)  # (B, in_channels*len(dilations), H, W)
        
        # 压缩到out_channels
        x_out = self.compress(x_concat)  # (B, out_channels, H, W)
        
        return x_out

