
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, channels, branches=3, reduction=16):
        super(SKFusion, self).__init__()
        self.branches = branches
        d = max(channels // reduction, 32)
        
        # 1. 全局信息聚合
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 2. 增强的紧凑特征描述 (V4.19)
        # 增加中间层深度，提升表达能力
        self.fc = nn.Sequential(
            nn.Linear(channels, d, bias=False),
            nn.LayerNorm(d),  # 使用 LayerNorm 保持梯度稳定
            nn.GELU(),  # 使用GELU替代ReLU，提升表达能力
            nn.Linear(d, d, bias=False),  # 增加中间层深度
            nn.LayerNorm(d),
            nn.GELU(),
        )
        
        # 3. 权重生成 (FC -> Softmax)
        self.fcs = nn.ModuleList([])
        for i in range(branches):
            self.fcs.append(nn.Linear(d, channels, bias=False))
            
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
        
        # Weighted Sum with Residual Connection (V4.19)
        # 添加残差连接，保留原始特征信息，提升小样本类别稳定性
        V = 0
        for i, x in enumerate(x_branches):
            V += x * weights[i]
        
        # 残差连接：保留原始融合特征，增强表达能力
        V = V + U  # 添加残差连接
            
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
        
        # Branch 2: 5x5 (Stacked 3x3)
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),
            nn.GELU()
        )
        
        # Branch 3: 7x7 (Stacked 3x3)
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, mid_channels),
            nn.GELU()
        )
        
        # Selective Kernel Fusion
        self.sk_fusion = SKFusion(mid_channels, branches=3)
        
        # Fusion Projection (Optional, keeps dimensions consistent)
        self.fusion_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        
        # Shortcut for ResBlock
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.norm_layer = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        # x shape: (B, C, H, W)
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)
        
        # SK Fusion (Dynamic Selection)
        out = self.sk_fusion([x3, x5, x7])
        
        # Linear Projection
        out = self.fusion_conv(out)
        
        # 残差连接
        out = out + self.shortcut(x)
        
        # 后处理
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout(out)
        
        return out