"""
spatial_branch 模块

实现空间分支的多尺度卷积融合模块（SpatialBranchModule）。
该模块对输入的局部空间邻域分别使用 3x3、5x5、7x7 三个并行卷积分支进行特征提取，
然后将分支输出拼接并通过 1x1 卷积进行通道压缩，最后经过归一化、激活与 dropout。
新增空间注意力机制（SEBlock）用于增强重要特征的权重分配。

输入/输出约定：
- 输入 x: (B, C_in, H, W)
- 输出 out: (B, C_out, H, W)

设计说明：
- 使用 GroupNorm(1, C_out) 来兼容 (B, C, H, W) 的归一化需求（等价于对通道做 LayerNorm 的效果）。
- Dropout 使用 2D 版本以更适合空间特征的正则化。
- SEBlock通过通道注意力机制自适应地重新校准特征响应。
"""

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    空间注意力机制（SEBlock）：通过通道维度进行特征重新校准

    设计思路：
    - 使用全局平均池化获取通道级全局信息
    - 通过两个全连接层建模通道间的依赖关系
    - 使用Sigmoid生成注意力权重进行特征重校准
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化：(B, C, H, W) -> (B, C, 1, 1)
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),  # 升维
            nn.Sigmoid()  # 生成注意力权重
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            加权后的特征图 (B, C, H, W)
        """
        return x * self.attention(x)


class SpatialBranchModule(nn.Module):
    """
    空间分支模块：多尺度卷积融合 + 空间注意力机制

    架构流程：
    1. 多尺度卷积特征提取（3x3, 5x5, 7x7）
    2. 特征拼接与融合（1x1卷积）
    3. 空间注意力重新校准（SEBlock）
    """

    def __init__(self, in_channels, out_channels, bias=False, dropout_rate=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.mid_channels = min(32, max(8, self.in_channels // 8))

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

        # 融合层：将3个分支的中间结果拼接后压缩到 out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(
                self.mid_channels * 3, self.out_channels, kernel_size=1, bias=self.bias
            ),
            nn.GroupNorm(1, self.out_channels),
            nn.GELU(),
            nn.Dropout2d(self.dropout_rate),
        )

        # 空间注意力机制：对融合后的特征进行通道重新校准
        self.spatial_attention = SpatialAttention(self.out_channels)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        # 拼接 (B, 3*C_out, H, W)
        x_cat = torch.cat([x3, x5, x7], dim=1)

        # 融合
        out = self.fusion(x_cat)

        # 应用空间注意力机制进行特征重新校准
        out = self.spatial_attention(out)

        return out
