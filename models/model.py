import torch
import torch.nn as nn
from .preprocess_module import PreprocessModule
from .spectral_branch import SpectralBranchModule
from .spatial_branch import SpatialBranchModule
from .fusion_module import FusionModule
from .mamba_global_module import MambaGlobalModule
from .classifier import Classifier


class MambaHSINet(nn.Module):
    """
    高光谱图像分类网络

    架构调整目标：
    1. **解决欠拟合**：恢复 Spatial Branch 的致密卷积能力 (Multi-Scale ResBlock) 和 Spectral Branch 的 MLP 深度。
    2. **解决抖动和不稳定**：使用 Residual Connections (残差连接) 贯穿全网，特别是 Spatial Branch。
    3. **保持小样本精度**：保留 SE Attention 和 Global Mamba。
    
    六个主要步骤：
    1. 预处理模块：归一化输入
    2. 光谱分支模块：提取光谱维度的特征
    3. 空间分支模块：提取空间维度的特征
    4. 融合模块：融合双分支特征
    5. Mamba全局建模模块：全局上下文建模
    6. 分类模块：最终分类
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

        # 1. 预处理模块
        self.preprocess_module = PreprocessModule(bands=bands)

        # 2. 光谱分支模块
        self.spectral_branch = SpectralBranchModule(
            in_channels=bands, out_channels=d_model, dropout_rate=dropout_rate
        )

        # 3. 空间分支模块
        self.spatial_branch = SpatialBranchModule(
            in_channels=bands, out_channels=d_model, dropout_rate=dropout_rate
        )

        # 4. 融合模块
        self.fusion_module = FusionModule(
            in_channels=d_model * 2, out_channels=d_model, dropout_rate=dropout_rate
        )

        # 5. Mamba全局建模模块
        self.mamba_global_module = MambaGlobalModule(
            d_model=d_model, dropout_rate=dropout_rate
        )

        # 6. 分类模块
        self.classifier = Classifier(
            d_model=d_model,
            num_classes=num_classes,
            classifier_hidden=classifier_hidden,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        h, w, _ = x.shape

        # 1. 预处理
        x_norm = self.preprocess_module(x)

        # 2. 光谱分支特征提取
        x_spec = self.spectral_branch(x_norm)  # (H, W, d_model)

        # 3. 空间分支特征提取
        x_spat = x_norm.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        x_spat = self.spatial_branch(x_spat)  # (1, d_model, H, W)
        x_spat = x_spat.squeeze(0).permute(1, 2, 0)  # (H, W, d_model)

        # 4. 融合
        x_cat = torch.cat([x_spec, x_spat], dim=-1)  # (H, W, d_model * 2)
        x_fused = self.fusion_module(x_cat)

        # 5. Mamba全局建模
        x_mamba = self.mamba_global_module(x_fused)

        # 6. 分类
        output = self.classifier(x_mamba)
        output = output.reshape(h, w, -1)

        return output
