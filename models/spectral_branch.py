import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralBranchModule(nn.Module):
    """
    光谱分支模块：Spectral Smoothing + Multi-Scale CNN + Attention Pooling (V4.6)
    
    改进：
    1. **新增 Spectral Smoothing Layer**：
       在多尺度特征提取之前，增加一个 3x3 的深度卷积 (Depthwise Conv)。
       - 目的：对原始光谱数据进行平滑预处理，滤除高频噪声（Spikes）。
       - 原因：Class 9 的剧烈波动 (31% - 100%) 表明模型对输入光谱的微小扰动极其敏感。通过预先平滑，可以提升对噪声的鲁棒性。
       
    2. **保持 Learnable Attention Pooling**：
       V4.5 的 Attention Pooling 机制本身是好的，配合平滑后的输入，应能更稳定地聚焦关键波段。
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = 16 
        
        # 1. Spectral Smoothing Layer (Pre-processing)
        # Depthwise Conv1d: 独立对每个波段/通道进行平滑
        # 注意：这里的输入是 (N, 1, Bands)。我们需要平滑的是 Bands 维度。
        self.smoothing = nn.Sequential(
            nn.ReflectionPad1d(1), # 使用反射填充减少边界效应
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
        
        # Step 1: Smoothing
        x_smooth = self.smoothing(x_seq)
        
        # Step 2: Multi-Scale Conv
        b3 = self.branch3(x_smooth)
        b5 = self.branch5(x_smooth)
        b7 = self.branch7(x_smooth)
        
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