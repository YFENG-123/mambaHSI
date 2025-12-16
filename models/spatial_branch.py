import torch
import torch.nn as nn


class SpatialBranchModule(nn.Module):
    """
    空间分支模块：多尺度卷积融合
    
    包含三个并行卷积分支：
    - 3x3 卷积
    - 5x5 卷积
    - 7x7 卷积
    
    使用可学习的加权融合将三个分支的特征融合，并加入LayerNorm和Dropout防止过拟合。
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        
        # 可学习的融合权重（三个分支的权重）
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)  # 初始化为均匀权重

        # 更通用的LayerNorm做法是 permute 后 norm 再 permute 回来，或者使用 BatchNorm2d / GroupNorm
        # 这里为了稳健性，使用 GroupNorm 或者 BatchNorm2d 可能更好，但题目语境倾向于 transformer 风格的 LayerNorm
        # 不过 pytorch 的 LayerNorm 通常作用在 last dimension。对于 (B, C, H, W)，需要 permute。
        self.norm_layer = nn.GroupNorm(1, out_channels) # GroupNorm(1, C) 等价于 LayerNorm 但支持 (B, C, H, W) 格式
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        
        # 使用softmax归一化权重，确保权重和为1
        weights = torch.softmax(self.fusion_weights, dim=0)
        
        # 可学习的加权融合
        out = weights[0] * x3 + weights[1] * x5 + weights[2] * x7
        
        # 归一化 + 激活 + Dropout
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout(out)
        
        return out

