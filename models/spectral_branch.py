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
        self.feat_channels = 24  # 回退到24，保持稳定性，feat_channels=26导致整体OA下降（95.94% vs 97.12%）（V4.16 Reverted）

        # 1. Spectral Smoothing Layer（谱轴预平滑）
        self.smoothing = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(1),
            nn.GELU(),
        )

        # 2. 多尺度特征提取（3,5,7）
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
        self.branch7 = nn.Sequential(
            nn.Conv1d(1, self.feat_channels, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(self.feat_channels),
            nn.GELU(),
        )

        # 3. 注意力池化层（pixel-wise）
        self.att_pool3 = nn.Conv1d(self.feat_channels, 1, kernel_size=1)
        self.att_pool5 = nn.Conv1d(self.feat_channels, 1, kernel_size=1)
        self.att_pool7 = nn.Conv1d(self.feat_channels, 1, kernel_size=1)

        # 融合投影：3*feat_channels -> out_channels (V4.16: 24*3=72 -> out_channels)
        self.fc = nn.Sequential(
            nn.Linear(self.feat_channels * 3, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # SE 注意力（增强版 - V4.18）
        # 3层结构，提升表达能力，添加残差连接增强健壮性
        se_dim = max(4, out_channels // 8)
        self.se = nn.Sequential(
            nn.Linear(out_channels, se_dim, bias=False),
            nn.LayerNorm(se_dim),  # 添加LayerNorm稳定训练
            nn.GELU(),  # 使用GELU替代ReLU，提升表达能力
            nn.Linear(se_dim, se_dim, bias=False),  # 增加中间层深度
            nn.LayerNorm(se_dim),
            nn.GELU(),
            nn.Linear(se_dim, out_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (H, W, bands)
        h, w, bands = x.shape
        x_seq = x.reshape(-1, 1, bands)  # (N, 1, L)

        # 谱轴预平滑
        x_smooth = self.smoothing(x_seq)

        # 多尺度卷积
        b3 = self.branch3(x_smooth)
        b5 = self.branch5(x_smooth)
        b7 = self.branch7(x_smooth)

        # 注意力权重（按谱轴 softmax）
        w3 = F.softmax(self.att_pool3(b3), dim=-1)
        w5 = F.softmax(self.att_pool5(b5), dim=-1)
        w7 = F.softmax(self.att_pool7(b7), dim=-1)

        # 加权求和 -> (N, C)
        p3 = torch.sum(b3 * w3, dim=-1)
        p5 = torch.sum(b5 * w5, dim=-1)
        p7 = torch.sum(b7 * w7, dim=-1)

        feat = torch.cat([p3, p5, p7], dim=1)  # (N, 72) = 24*3

        # 投影与增强SE注意力（带残差连接）
        out = self.fc(feat)
        attn = self.se(out)
        out = out * (1 + attn)  # 残差连接：保留原始特征，增强表达能力

        out = out.reshape(h, w, self.out_channels)
        return out

