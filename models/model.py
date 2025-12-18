import torch
import torch.nn as nn
from .s6_core import S6Core
from .spatial_branch import SpatialBranchModule
from .spectral_branch import SpectralBranchModule
from .fusion_module import FusionModule


class MambaHSINet(nn.Module):
    """
    高光谱图像分类网络 (V2.5 - Robust Recovery)

    架构调整目标：
    1. **解决欠拟合**：恢复 Spatial Branch 的致密卷积能力 (Multi-Scale ResBlock) 和 Spectral Branch 的 MLP 深度。
    2. **解决抖动和不稳定**：使用 Residual Connections (残差连接) 贯穿全网，特别是 Spatial Branch。
    3. **保持小样本精度**：保留 SE Attention 和 Global Mamba。
    """

    def __init__(
        self,
        image_x=145,
        image_y=145,
        num_classes=17,
        bands=200,
        dropout_rate=0.5,
        d_model=64,
        classifier_hidden=64,
    ):
        super().__init__()
        self.bands = bands
        self.image_x = image_x
        self.image_y = image_y
        self.d_model = d_model

        # 预处理
        self.preprocess = nn.LayerNorm(bands)

        # 1. 光谱分支：2-Layer MLP + SE (强非线性 + 注意力)
        self.spectral_branch = SpectralBranchModule(bands, d_model, dropout_rate)

        # 2. 空间分支：Stacked Conv ResBlock (强局部特征 + 梯度稳定)
        self.spatial_branch = SpatialBranchModule(bands, d_model, dropout_rate)

        # 3. 融合模块：Simple Fusion
        self.fusion_module = FusionModule(in_channels=d_model * 2, out_channels=d_model, dropout_rate=dropout_rate)

        # 4. Mamba Global Block
        self.mamba_fwd = S6Core(d_model=d_model)
        self.mamba_bwd = S6Core(d_model=d_model)
        
        self.mamba_norm = nn.LayerNorm(d_model)
        self.mamba_dropout = nn.Dropout(dropout_rate)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x):
        h, w, _ = x.shape

        # 1. 预处理
        x_norm = self.preprocess(x)

        # 2. 双分支特征提取
        x_spec = self.spectral_branch(x_norm)
        
        x_spat = x_norm.permute(2, 0, 1).unsqueeze(0)
        x_spat = self.spatial_branch(x_spat)
        x_spat = x_spat.squeeze(0).permute(1, 2, 0)

        # 3. 融合
        x_cat = torch.cat([x_spec, x_spat], dim=-1)
        x_fused = self.fusion_module(x_cat)

        # 4. Mamba 全局建模
        x_seq = x_fused.reshape(-1, self.d_model).unsqueeze(0) # (1, H*W, d)
        
        residual = x_seq
        x_norm = self.mamba_norm(x_seq)

        x_fwd = self.mamba_fwd(x_norm)
        x_bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        x_mamba = x_fwd + x_bwd 
        
        x_mamba = self.mamba_dropout(x_mamba)
        x_mamba = x_mamba + residual

        x_mamba = x_mamba.squeeze(0) # (H*W, d)

        # 5. 分类
        output = self.classifier(x_mamba)
        output = output.reshape(h, w, -1)

        return output
