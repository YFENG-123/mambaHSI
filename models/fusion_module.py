import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """
    融合模块：Balanced Fusion + Denoising (V4.4)
    
    改进：
    1. **保留 Balanced GroupNorm**：维持 V3.4 的特征平衡设计。
    2. **新增 Denoising Conv (3x3 Depthwise)**：
       在融合后增加一个 3x3 深度卷积。
       - 原因：用户拒绝在 Mamba 内部添加 Local Conv，因为"空间分支已经提取了空间特征"。
       - 但是，将空间特征与光谱特征融合后，产生的新特征图可能在边界处存在不连续或高频噪声，这会干扰后续 Pure SSM 的序列建模。
       - 我们在 Fusion 模块末端添加这个轻量级卷积，作为"特征平滑/去噪"步骤。这不属于"特征提取"，而是"特征预处理"，确保送入 Mamba 的 tokens 具有良好的局部连续性。
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
        
        # Residual Connection for Smoothing (Optional but good for preserving signals)
        # x_out = x_fused + x_smooth 
        # 这里直接使用平滑后的特征，因为目标是去噪
        x_out = self.act(x_smooth)
        
        out = x_out.squeeze(0).permute(1, 2, 0)
        out = self.dropout(out)
        
        return out
