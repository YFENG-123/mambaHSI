import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralBranchModule(nn.Module):
    """
    光谱分支模块：Spectral Smoothing + Multi-Scale CNN + Attention Pooling (V4.6 Reverted)
    
    回退说明：
    V4.9 的 Spatial-Spectral Smoothing (2D) 导致了严重的训练不稳定和验证集崩溃（Acc < 70%）。
    这表明强制的 2D 空间平滑可能破坏了光谱特征的完整性，或者引入了难以优化的参数。
    
    我们回退到 **V4.6** 版本，该版本表现最为稳定（OA ~97.5%，Class 9 ~83%）。
    
    微调：
    在 V4.6 的基础上，为 Smoothing Layer 增加一个 **残差连接 (Residual Connection)**。
    - Out = Raw + Smooth(Raw)
    - 目的：防止平滑层过度模糊信号，保证原始光谱信息能够直接传递，同时利用平滑层补充去噪后的特征。
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = 16 
        
        # 1. Spectral Smoothing Layer (Pre-processing) - 1D Depthwise
        self.smoothing = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=0, bias=False), 
            nn.BatchNorm1d(1),
            nn.GELU()
        )
        
        # 2. Multi-Scale Feature Extraction
        # Branch 1: Kernel 3
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, self.feat_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(self.feat_channels),
            nn.GELU()
        )
        
        # Branch 2: Kernel 5
        self.branch5 = nn.Sequential(
            nn.Conv1d(1, self.feat_channels, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(self.feat_channels),
            nn.GELU()
        )
        
        # Branch 3: Kernel 7
        self.branch7 = nn.Sequential(
            nn.Conv1d(1, self.feat_channels, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(self.feat_channels),
            nn.GELU()
        )
        
        # 3. Attention Pooling Layers
        self.att_pool3 = nn.Conv1d(self.feat_channels, 1, kernel_size=1)
        self.att_pool5 = nn.Conv1d(self.feat_channels, 1, kernel_size=1)
        self.att_pool7 = nn.Conv1d(self.feat_channels, 1, kernel_size=1)
        
        # 融合全连接层
        # Input: 3 branches * 16 channels = 48
        self.fc = nn.Sequential(
            nn.Linear(48, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # SE Attention
        self.se = nn.Sequential(
            nn.Linear(out_channels, out_channels // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 8, out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (H, W, bands)
        h, w, bands = x.shape
        x_seq = x.reshape(-1, 1, bands) # (N, 1, L)
        
        # Step 1: Smoothing with Residual
        # Res = Raw + Smooth
        x_smooth = self.smoothing(x_seq)
        x_in = x_seq + x_smooth
        
        # Step 2: Multi-Scale Conv
        b3 = self.branch3(x_in)
        b5 = self.branch5(x_in)
        b7 = self.branch7(x_in)
        
        # Step 3: Attention Pooling
        # Weights: (N, 1, L)
        w3 = F.softmax(self.att_pool3(b3), dim=-1)
        w5 = F.softmax(self.att_pool5(b5), dim=-1)
        w7 = F.softmax(self.att_pool7(b7), dim=-1)
        
        # Weighted Sum: (N, C)
        p3 = torch.sum(b3 * w3, dim=-1)
        p5 = torch.sum(b5 * w5, dim=-1)
        p7 = torch.sum(b7 * w7, dim=-1)
        
        # Concat
        feat = torch.cat([p3, p5, p7], dim=1) # (N, 48)
        
        # Projection
        out = self.fc(feat)
        
        # SE
        attn = self.se(out)
        out = out * attn
        
        out = out.reshape(h, w, self.out_channels)
        return out
