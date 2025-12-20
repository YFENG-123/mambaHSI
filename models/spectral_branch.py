import torch
import torch.nn as nn
import torch.nn.functional as F


"""
模型版本: V4.18
说明: 光谱分支模块 — 增强SE注意力 + 残差连接，提升健壮性。
"""


class SpectralBranchModule(nn.Module):
    """光谱分支模块：Spectral Smoothing + Multi-Scale CNN + Enhanced SE Attention (V4.18)

    改进：
    - 在多尺度特征提取之前进行谱轴平滑，降低高频噪声影响。
    - 使用 3 路并行 Conv1d（kernel=3,5,7）提取多尺度光谱特征。
    - 通过可学习的注意力加权（pixel-wise attention pooling）对每一路做加权求和，最后拼接并投影到输出通道。
    - **增强SE注意力** (V4.18)：
      - 增加中间层深度（3层结构），提升表达能力
      - 添加残差连接：out = out * (1 + attn)，保留原始特征信息
      - 提升小样本类别的识别稳定性
    """

    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = 10  # 从12减少到10以进一步降低内存占用

        # 1. 多尺度特征提取（精简为2个分支：3,5）
        # 移除smoothing层以降低内存占用
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, self.feat_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(self.feat_channels),
            nn.GELU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(1, self.feat_channels, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(self.feat_channels),
            nn.GELU(),
        )

        # 3. 注意力池化层（pixel-wise）
        self.att_pool3 = nn.Conv1d(self.feat_channels, 1, kernel_size=1)
        self.att_pool5 = nn.Conv1d(self.feat_channels, 1, kernel_size=1)

        # 融合投影：2*feat_channels -> out_channels (精简版：12*2=24 -> out_channels)
        self.fc = nn.Sequential(
            nn.Linear(self.feat_channels * 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 移除SE注意力以降低内存占用，保留核心特征提取能力

    def forward(self, x):
        # x: (H, W, bands)
        h, w, bands = x.shape
        x_seq = x.reshape(-1, 1, bands)  # (N, 1, L)

        # 多尺度卷积（精简为2个分支，移除smoothing）
        b3 = self.branch3(x_seq)
        b5 = self.branch5(x_seq)

        # 注意力权重（按谱轴 softmax）
        w3 = F.softmax(self.att_pool3(b3), dim=-1)
        w5 = F.softmax(self.att_pool5(b5), dim=-1)

        # 加权求和 -> (N, C)
        p3 = torch.sum(b3 * w3, dim=-1)
        p5 = torch.sum(b5 * w5, dim=-1)

        feat = torch.cat([p3, p5], dim=1)  # (N, 20) = 10*2

        # 投影（移除SE注意力以降低内存占用）
        out = self.fc(feat)

        out = out.reshape(h, w, self.out_channels)
        return out

