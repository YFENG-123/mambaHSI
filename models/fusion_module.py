import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """
    融合模块：Balanced Fusion + Enhanced Channel Attention + Denoising + Gated Fusion (V4.24)
    
    改进：
    1. **保留 Balanced GroupNorm**：维持 V3.4 的特征平衡设计。
    2. **增强通道注意力机制 (V4.24)**：
       - 只保留通道注意力，移除空间注意力
       - 空间注意力对小样本类别（类别9）有负面影响，导致准确率从96.25%降至88.75%
       - 通道注意力更稳定，有助于提升小样本类别的识别能力
       - 添加残差连接，保留重要特征信息
       - 添加GroupNorm稳定训练，提升表达能力（V4.24）
    3. **保留 Denoising Conv (3x3 Depthwise)**：特征平滑/去噪步骤。
    4. **保留可学习门控融合** (V4.5 Stable)：稳定的特征混合机制。
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        
        # 1. 平衡归一化 (Group=2, Split Spectral/Spatial)
        self.pre_norm = nn.GroupNorm(num_groups=2, num_channels=in_channels)
        
        # 2. 简化通道注意力（进一步精简 - 移除中间层）
        # 通道注意力（Channel Attention）- 超精简版
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()  # 通道注意力权重
        )
        
        # 3. 融合卷积 (1x1)
        self.fusion_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        
        # 移除去噪卷积和门控融合以降低内存占用
        
        # 5. 输出层归一化 (V4.7+)
        # 在送入 Mamba 之前，对 (H, W, C) token 做 LayerNorm，进一步稳定不同 seed / 划分下的分布
        self.out_norm = nn.LayerNorm(out_channels)
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
            
    def forward(self, x):
        # x: (H, W, in_channels)
        x_in = x.permute(2, 0, 1).unsqueeze(0)
        
        # Norm
        x_norm = self.pre_norm(x_in)
        
        # Fuse
        x_fused = self.fusion_conv(x_norm) # (1, C, H, W)
        
        # 简化通道注意力（移除残差连接和去噪/门控融合）
        channel_attn = self.channel_attn(x_fused)  # (1, C, 1, 1)
        x_out = x_fused * channel_attn  # 直接应用注意力
        x_out = self.act(x_out)
        
        out = x_out.squeeze(0).permute(1, 2, 0)
        out = self.out_norm(out)
        out = self.dropout(out)
        
        return out
