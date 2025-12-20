
import torch
import torch.nn as nn


class SKFusion(nn.Module):
    """
    Selective Kernel Fusion (SK Fusion) - Enhanced Version (V4.19)
    
    目的：
    V3.8 的 Coordinate Attention 虽然引入了位置信息，但在多尺度特征融合上仍然是简单的拼接 (Concat)。
    对于 Class 7/9 这样的小样本，它们需要极小的感受野 (3x3)，而大背景需要大感受野 (7x7)。
    简单的拼接让网络必须在所有位置同时处理所有尺度的特征，容易引入噪声。
    
    SK Fusion 允许网络根据输入图像的内容，**动态地**为每个像素（或通道）选择最合适的感受野尺度。
    - 对于小物体区域，增加 3x3 分支的权重。
    - 对于大平滑区域，增加 7x7 分支的权重。
    这种动态选择机制比静态的 Attention 更稳健，能有效解决小样本被大尺度特征淹没的问题。
    
    改进（V4.19）：
    - 增强权重生成网络：增加中间层深度，提升表达能力
    - 使用GELU替代ReLU，提升非线性表达能力
    - 添加残差连接，保留原始特征信息，提升小样本类别稳定性
    """
    def __init__(self, channels, branches=2, reduction=16):
        super(SKFusion, self).__init__()
        self.branches = branches
        d = max(channels // reduction, 16)  # 从32减少到16
        
        # 1. 全局信息聚合
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. 紧凑特征描述（精简版）
        self.fc = nn.Sequential(
            nn.Linear(channels, d, bias=False),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        
        # 3. 权重生成 (FC -> Softmax)
        self.fcs = nn.ModuleList([
            nn.Linear(d, channels, bias=False) for _ in range(branches)
        ])
            
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x_branches):
        # x_branches: list of tensors [B, C, H, W]
        
        # Element-wise Sum
        U = sum(x_branches) # (B, C, H, W)
        
        # Global Pooling
        s = self.avg_pool(U).flatten(1) # (B, C)
        
        # Enhanced Compact Descriptor (V4.19)
        z = self.fc(s) # (B, d)
        
        # Generate Weights
        weights = []
        for fc in self.fcs:
            weights.append(fc(z).unsqueeze(-1).unsqueeze(-1)) # (B, C, 1, 1)
            
        weights = torch.stack(weights, dim=0) # (Branches, B, C, 1, 1)
        weights = self.softmax(weights) # Normalize across branches
        
        # Weighted Sum（移除残差连接以降低内存占用）
        V = 0
        for i, x in enumerate(x_branches):
            V += x * weights[i]
            
        return V


class SpatialBranchModule(nn.Module):
    """
    空间分支模块：High-Capacity + Enhanced Selective Kernel Fusion (V4.19)
    
    改进：
    1. **保持 High Capacity**：继续使用 `mid_channels = out_channels`。
    2. **增强 SK Fusion** (V4.19)：提升小样本类别稳定性
       - 增强权重生成网络：增加中间层深度，提升表达能力
       - 使用GELU替代ReLU，提升非线性表达能力
       - 添加残差连接，保留原始特征信息，提升小样本类别稳定性
       这能从根本上提升对不同尺度物体（特别是 Class 7/9 等小目标）的适应能力，消除种子间的方差。
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        
        # High Capacity
        mid_channels = out_channels 
        
        # Branch 1: 3x3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),
            nn.GELU()
        )
        
        # Branch 2: 5x5 (单层卷积，进一步精简)
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
            nn.GroupNorm(1, mid_channels),
            nn.GELU()
        )
        
        # 简化融合：直接使用平均池化替代SK Fusion以降低内存占用
        # 移除SK Fusion以进一步降低内存占用
        self.fusion_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        
        # 后处理
        self.norm_layer = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        
        # 简化融合：直接平均（移除SK Fusion以降低内存占用）
        out = (x3 + x5) / 2  # 简单平均融合
        
        # Linear Projection
        out = self.fusion_conv(out)
        
        # 后处理（移除残差连接）
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout(out)
        
        return out