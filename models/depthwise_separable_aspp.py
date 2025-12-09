import torch
import torch.nn as nn


class DepthwiseSeparableASPP(nn.Module):
    """
    深度可分离ASPP模块
    将ASPP中的普通卷积替换为深度可分离卷积
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
        
        # 1x1卷积分支（深度可分离）
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 多个不同膨胀率的3x3深度可分离卷积分支
        self.dilated_convs = nn.ModuleList()
        for dilation in dilations:
            padding = dilation
            conv = nn.Sequential(
                # 深度卷积
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    groups=in_channels,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels),
                # 点卷积
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.dilated_convs.append(conv)
        
        # 全局平均池化分支
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 融合所有分支
        num_branches = 1 + len(dilations) + 1  # 1x1 + dilations + global_pool
        self.fusion = nn.Sequential(
            nn.Conv2d(
                out_channels * num_branches,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            融合后的特征 (B, out_channels, H, W)
        """
        batch_size, channels, height, width = x.size()
        
        # 1x1卷积分支
        x_1x1 = self.conv1x1(x)  # (B, out_channels, H, W)
        
        # 多个膨胀卷积分支
        dilated_features = []
        for dilated_conv in self.dilated_convs:
            feature = dilated_conv(x)  # (B, out_channels, H, W)
            dilated_features.append(feature)
        
        # 全局平均池化分支
        x_global = self.global_pool(x)  # (B, out_channels, 1, 1)
        x_global = nn.functional.interpolate(
            x_global, size=(height, width), mode='bilinear', align_corners=False
        )  # (B, out_channels, H, W)
        
        # 拼接所有分支
        all_features = [x_1x1] + dilated_features + [x_global]
        x_concat = torch.cat(all_features, dim=1)  # (B, out_channels*num_branches, H, W)
        
        # 融合
        x_fusion = self.fusion(x_concat)  # (B, out_channels, H, W)
        
        return x_fusion

