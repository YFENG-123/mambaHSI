"""
spectral_branch 模块

实现光谱分支（SpectralBranchModule），用于对每个像素的光谱向量进行多尺度特征提取与融合。
该模块采用与空间分支相同的残差拼接融合策略：多尺度1D卷积 + 残差连接 + 通道拼接 + 1D卷积融合。

架构特点：
- 多尺度1D卷积：3x3、5x5、7x7并行分支提取不同感受野的光谱特征
- 残差连接：保留原始光谱信息，避免信息丢失
- 统一融合：在通道维度拼接后通过1D卷积融合到目标谱长
- 最终投影：将融合后的谱向量映射到输出通道数

设计要点：
- 采用与空间分支类似的残差拼接策略，但针对谱向量特性优化
- 通过全局平均池化提取分支特征，避免序列维度的大卷积操作
- 残差连接保留原始光谱信息，线性融合确保内存效率

输入/输出约定：
- 输入 x: (H, W, bands)
- 输出 out: (H, W, out_channels)
"""

import torch
import torch.nn as nn

# 模型版本: V4.18
# 说明: 光谱分支模块 — 谱轴平滑 + 全连接投影（去除复杂 SE 以提升稳定性）


class SpectralBranchModule(nn.Module):
    """
    光谱分支模块：多尺度1D卷积 + 残差连接融合（内存优化版）

    架构流程：
    1. 多尺度1D卷积特征提取（3x3, 5x5, 7x7分支）
    2. 残差分支：原始输入的1x1投影
    3. 各分支全局平均池化提取特征向量
    4. 特征向量拼接并线性融合
    5. 最终线性投影到输出通道数

    内存优化特点：
    - 全局池化避免序列维度大卷积，显著减少内存使用
    - 残差连接保留原始光谱信息
    - 线性融合保持特征丰富性同时控制计算复杂度
    """

    def __init__(self, in_channels, out_channels, bias=False, dropout_rate=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.dropout_rate = dropout_rate
        # 减少中间通道数以节省内存：原来是 in_channels//24，现在进一步减少
        self.conv_channels = 6

        # 对每个像素的光谱序列使用正常 1D 卷积（Conv1d(1 -> C)），然后在通道维度融合
        # 三个尺度的卷积输出中间通道数 self.conv_channels
        self.conv3 = nn.Sequential(
            nn.Conv1d(1, self.conv_channels, kernel_size=3, padding=1, bias=self.bias),
            nn.GroupNorm(1, self.conv_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(1, self.conv_channels, kernel_size=5, padding=2, bias=self.bias),
            nn.GroupNorm(1, self.conv_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(1, self.conv_channels, kernel_size=7, padding=3, bias=self.bias),
            nn.GroupNorm(1, self.conv_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )

        # 残差分支：将原始输入投影到中间通道数以匹配其他分支
        self.residual_proj = nn.Sequential(
            nn.Conv1d(1, self.conv_channels, kernel_size=1, bias=self.bias),
            nn.GroupNorm(1, self.conv_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )

        # 内存优化融合策略：每个分支分别进行全局平均池化，然后拼接并线性融合
        # 这样避免了在序列维度上的复杂卷积，大幅减少内存使用
        self.fusion_linear = nn.Sequential(
            nn.Linear(self.conv_channels * 4, self.out_channels, bias=self.bias),
            nn.LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )


    def forward(self, x):
        # x: (H, W, bands)
        h_out, w_out, bands = x.shape

        # 将每个像素的谱向量批量化为 (H*W, 1, bands)
        x_pixels = x.reshape(-1, bands).unsqueeze(1).contiguous()  # (H*W, 1, bands)

        # 多尺度1D卷积分支
        x3 = self.conv3(x_pixels)  # (H*W, C, bands)
        x5 = self.conv5(x_pixels)  # (H*W, C, bands)
        x7 = self.conv7(x_pixels)  # (H*W, C, bands)

        # 残差分支：原始输入经过1x1投影
        x_res = self.residual_proj(x_pixels)  # (H*W, C, bands)

        # 内存优化融合：每个分支分别全局平均池化，然后拼接
        # 对每个分支在谱轴维度进行全局平均池化
        x3_pooled = x3.mean(dim=-1)  # (H*W, C)
        x5_pooled = x5.mean(dim=-1)  # (H*W, C)
        x7_pooled = x7.mean(dim=-1)  # (H*W, C)
        x_res_pooled = x_res.mean(dim=-1)  # (H*W, C)

        # 在特征维度拼接所有分支
        x_cat = torch.cat([x3_pooled, x5_pooled, x7_pooled, x_res_pooled], dim=-1)  # (H*W, 4*C)

        # 通过线性层融合到谱向量维度
        x_out = self.fusion_linear(x_cat)  # (H*W, in_channels)

        return x_out.reshape(h_out, w_out, self.out_channels)
