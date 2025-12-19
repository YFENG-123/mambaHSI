import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """
    融合模块：Balanced Fusion + Denoising + Spatial-Aware Gated Fusion (V4.6)
    
    改进：
    1. **保留 Balanced GroupNorm**：维持 V3.4 的特征平衡设计。
    2. **新增 Denoising Conv (3x3 Depthwise)**：
       在融合后增加一个 3x3 深度卷积。
       - 原因：用户拒绝在 Mamba 内部添加 Local Conv，因为"空间分支已经提取了空间特征"。
       - 但是，将空间特征与光谱特征融合后，产生的新特征图可能在边界处存在不连续或高频噪声，这会干扰后续 Pure SSM 的序列建模。
       - 我们在 Fusion 模块末端添加这个轻量级卷积，作为"特征平滑/去噪"步骤。这不属于"特征提取"，而是"特征预处理"，确保送入 Mamba 的 tokens 具有良好的局部连续性。
    3. **可学习门控融合** (V4.5 Stable)：
       - 使用全局池化生成门控权重（稳定版本）
       - 公式：out = gate * x_fused + (1 - gate) * x_smooth
       - 相比直接残差连接，这种方式更稳定，让网络自动学习最佳混合比例
       - 回退原因：空间感知门控融合引入了过多复杂性，导致训练不稳定和性能下降
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        
        # 1. 平衡归一化 (Group=2, Split Spectral/Spatial)
        self.pre_norm = nn.GroupNorm(num_groups=2, num_channels=in_channels)
        
        # 2. 融合卷积 (1x1)
        self.fusion_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        
        # 3. 去噪/平滑卷积 (3x3 Depthwise)
        # 仅用于平滑局部邻域，不改变通道数
        self.denoise_conv = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1, 
            groups=out_channels, # Depthwise
            bias=False
        )
        self.denoise_norm = nn.GroupNorm(1, out_channels)
        
        # 4. 可学习门控融合 (V4.5 Stable)
        # 生成门控权重，用于平衡原始融合特征和平滑特征
        # 使用全局池化（稳定版本），避免空间卷积带来的复杂性
        self.gate_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.Sigmoid()  # 输出 [0, 1] 的权重
        )
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
            
    def forward(self, x):
        # x: (H, W, in_channels)
        x_in = x.permute(2, 0, 1).unsqueeze(0)
        
        # Norm & Fuse
        x_norm = self.pre_norm(x_in)
        x_fused = self.fusion_conv(x_norm) # (1, C, H, W)
        
        # Denoise / Smooth
        x_smooth = self.denoise_conv(x_fused)
        x_smooth = self.denoise_norm(x_smooth)
        
        # 可学习门控融合 (V4.5 Stable)
        # gate: [0, 1]，初始偏向平滑特征，网络会学习最佳混合比例
        gate = self.gate_conv(x_fused)  # (1, C, 1, 1)
        x_out = gate * x_fused + (1 - gate) * x_smooth
        x_out = self.act(x_out)
        
        out = x_out.squeeze(0).permute(1, 2, 0)
        out = self.dropout(out)
        
        return out
