"""
spatial_branch 模块

实现空间分支的多尺度卷积融合模块（SpatialBranchModule）。
该模块对输入的局部空间邻域分别使用 3x3、5x5、7x7 三个并行卷积分支进行特征提取，
同时加入残差连接保留原始输入特征，然后将所有分支输出拼接并通过 1x1 卷积进行通道压缩。

架构特点：
- 多尺度卷积：3x3、5x5、7x7 并行分支提取不同感受野的空间特征
- 残差连接：保留原始输入作为补充特征，避免信息丢失
- 融合策略：4个分支（3个卷积 + 1个残差）拼接后通过1x1卷积压缩

输入/输出约定：
- 输入 x: (B, C_in, H, W)
- 输出 out: (B, C_out, H, W)

设计说明：
- 使用 GroupNorm(1, C) 进行归一化，对每个通道单独处理
- Dropout 使用 2D 版本以适合空间特征的正则化
- 残差分支通过1x1卷积调整通道维度以保持一致性
"""

import torch
import torch.nn as nn


class SKFusion(nn.Module):
    """
    空间分支模块：多尺度卷积 + 残差连接融合

    架构流程：
    1. 多尺度卷积特征提取（3x3, 5x5, 7x7分支）
    2. 残差分支：原始输入的1x1投影
    3. 四分支特征拼接（3个卷积 + 1个残差）
    4. 1x1卷积融合到目标输出通道

    优化特点：
    - 多尺度特征：不同感受野的空间信息捕获
    - 残差连接：保留原始输入特征，避免梯度消失
    - 统一融合：所有分支在中间通道维度对齐后融合
    """

    def __init__(self, in_channels, out_channels, bias=False, dropout_rate=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.mid_channels = max(6, self.in_channels // 24)

        # 先将多尺度卷积输出到一个中间通道数（mid_channels），再通过1x1融合到目标输出通道
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.mid_channels,
                kernel_size=3,
                padding=1,
                bias=self.bias,
            ),
            nn.GroupNorm(1, self.mid_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.mid_channels,
                kernel_size=5,
                padding=2,
                bias=self.bias,
            ),
            nn.GroupNorm(1, self.mid_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.mid_channels,
                kernel_size=7,
                padding=3,
                bias=self.bias,
            ),
            nn.GroupNorm(1, self.mid_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )

        # 残差分支：将原始输入投影到中间通道数以匹配其他分支
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.mid_channels,
                kernel_size=1,
                bias=self.bias,
            ),
            nn.GroupNorm(1, self.mid_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )

        # 融合层：将4个分支的中间结果（3个卷积 + 1个残差）拼接后压缩到 out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(
                self.mid_channels * 4, self.out_channels, kernel_size=1, bias=self.bias
            ),
            nn.GroupNorm(1, self.out_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )

    def forward(self, x):
        # x shape: (B, C, H, W)

        # 多尺度卷积分支
        x3 = self.conv3(x)  # (B, mid_channels, H, W)
        x5 = self.conv5(x)  # (B, mid_channels, H, W)
        x7 = self.conv7(x)  # (B, mid_channels, H, W)

        # 残差分支：原始输入经过1x1投影
        x_res = self.conv1(x)  # (B, mid_channels, H, W)

        # 拼接所有分支（包括残差连接） (B, 4*mid_channels, H, W)
        x_cat = torch.cat([x3, x5, x7, x_res], dim=1)

        # 融合到目标输出通道
        out = self.fusion(x_cat)  # (B, out_channels, H, W)

        return out
