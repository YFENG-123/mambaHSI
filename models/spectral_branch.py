"""
spectral_branch 模块

实现光谱分支（SpectralBranchModule），用于对每个像素的光谱向量进行平滑与特征映射。
该模块通过谱轴平滑（ReplicationPad1d + Conv1d）保护短谱带场景的稳定性，然后使用全连接 MLP
将原始谱长映射到中间表示（feat_channels），再投影到输出通道数。

设计要点：
- 谱轴预平滑以降低高频噪声并提升对极短谱带长度的鲁棒性。
- 以全连接（MLP）为主的光谱特征提取更适合逐像素的谱向量建模。
- 保持实现稳定性与可复现性，移除了复杂的 SE 结构以减少小样本过拟合风险。

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
    光谱分支模块：仅保留 MLP（输入谱向量 -> MLP -> 输出）
    """

    def __init__(self, in_channels, out_channels, bias=False, dropout_rate=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.conv_channels = max(8, self.in_channels // 24)

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

        # 在残差相加后进行归一化 + 激活 + dropout（逐像素）
        self.post_fuse = nn.Sequential(
            nn.LayerNorm(self.in_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )

        # 可学习的多尺度融合权重（4个分支：3x3, 5x5, 7x7, 原始信号）
        #self.fusion_weights = nn.Parameter(torch.ones(4), requires_grad=True)
        # 优化初始化：给予原始信号更高的权重，更好保持类别特征
        self.fusion_weights = nn.Parameter(torch.tensor([0.8, 0.8, 0.8, 1.2]), requires_grad=True)

        # Linear 输入维度为谱长度 self.in_channels
        self.project_fc = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels, bias=self.bias),
            nn.LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )

    def forward(self, x):
        # x: (H, W, bands)
        h_out, w_out, bands = x.shape

        # 将每个像素的谱向量批量化为 (H*W, 1, bands)
        x_pixels = x.reshape(-1, bands).unsqueeze(1).contiguous()  # (H*W, 1, bands)
        x3 = self.conv3(x_pixels)  # (H*W, C, bands)
        x5 = self.conv5(x_pixels)  # (H*W, C, bands)
        x7 = self.conv7(x_pixels)  # (H*W, C, bands)
        # 可学习的加权融合（使用softmax归一化的权重）
        weights = torch.softmax(self.fusion_weights, dim=0)
        x_fused_seq = (
            weights[0] * x3.sum(dim=1) +
            weights[1] * x5.sum(dim=1) +
            weights[2] * x7.sum(dim=1) +
            weights[3] * x_pixels.squeeze(1)
        )  # (H*W, bands)
        x_fused_seq = self.post_fuse(x_fused_seq)

        # 逐像素将谱序列投影到 out_channels（保持原模块输出接口）
        x_proj = self.project_fc(x_fused_seq)  # (H*W, out_channels)

        # 恢复为 (H, W, out_channels)
        out = x_proj.reshape(h_out, w_out, self.out_channels)
        return out
