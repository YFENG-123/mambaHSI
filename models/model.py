"""
models.model

定义高光谱图像分类网络 `MambaHSINet`，采用光谱分支 + 空间分支 的双分支架构，
并在特征融合后引入 Mamba 层（双向 S6Core 封装）以增强空间-光谱上下文建模能力。

主要组成：
- 预处理（LayerNorm）用于对谱带通道进行归一化
- 光谱分支（`SpectralBranchModule`）：对光谱信息做逐像素的全连接投影
- 空间分支（`SpatialBranchModule`）：对局部空间邻域做多尺度卷积提取
- 特征融合（1x1 Conv）将两个分支的特征压缩为 d_model
- Mamba 层（`MambaLayer`）进行序列级别的长范围依赖建模
- 分类头（MLP）将每个像素的特征映射到类别分布

输入/输出形状约定：
- 输入 x: (H, W, bands)
- 输出: (H, W, num_classes)

注意：网络内部在不同模块间会进行 permute/reshape 以匹配 Conv2d / Linear / S6Core 的输入要求。
"""

import torch
import torch.nn as nn
from .spatial_branch import SpatialBranchModule
from .spectral_branch import SpectralBranchModule
from .mamba_layer import MambaLayer
from .fusion import FusionModule
from .classifier import ClassifierModule


class MambaHSINet(nn.Module):
    """
    高光谱图像分类网络（双分支架构 + 多尺度空间特征）

    架构流程：
    1. 归一化
    2. 双分支处理：
       - 光谱分支：全连接 (Linear)
       - 空间分支 (SpatialBranchModule)：
            - 并行 3x3, 5x5, 7x7 卷积
            - 拼接 -> 1x1 卷积压缩
    3. 拼接两个分支特征
    4. 1x1卷积融合
    5. Mamba处理 (S6Core)
    6. LayerNorm -> Dropout -> 分类器
    """

    def __init__(
        self,
        image_x=145,
        image_y=145,
        num_classes=17,
        bands=200,
        dropout_rate=0.3,
        d_model=64,
    ):
        super().__init__()
        self.bands = bands
        self.image_x = image_x
        self.image_y = image_y
        self.dropout_rate = dropout_rate
        self.d_model = d_model

        # 预处理层：归一化
        self.preprocess = nn.LayerNorm(bands)

        # 光谱分支：增强的线性映射 (Linear + GELU + Dropout)
        # 输入 (H, W, bands) -> 输出 (H, W, d_model)
        self.spectral_branch = SpectralBranchModule(
            bands, d_model, bias=False, dropout_rate=self.dropout_rate
        )

        # 空间分支：多尺度卷积模块
        # 输入 (bands, H, W) -> 输出 (d_model, H, W)
        self.spatial_branch = SpatialBranchModule(
            bands, d_model, bias=False, dropout_rate=self.dropout_rate
        )

        # 融合层：封装为单独模块（1x1 卷积）
        # 输入拼接后的特征 (2*d_model, H, W) -> 输出 (d_model, H, W)
        self.fusion = FusionModule(
            d_model * 2, d_model, bias=False, dropout_rate=self.dropout_rate
        )

        # Mamba 层封装（双向 + Pre-Norm + Dropout）
        self.mamba = MambaLayer(d_model=d_model, dropout_rate=self.dropout_rate)

        # 分类器（封装为单独模块）
        classifier_hidden = 128  # 隐藏层维度
        self.classifier = ClassifierModule(
            d_model, classifier_hidden, num_classes, dropout_rate=self.dropout_rate
        )

    def forward(self, x):
        """
        Args:
            x: 输入高光谱数据 (H, W, bands)
        Returns:
            分类结果 (H, W, num_classes)
        """
        h, w, _ = x.shape

        # 1. 预处理：归一化
        x_norm = self.preprocess(x)  # (H, W, bands)

        # 2. 双分支处理

        # 光谱分支 (消融实验：临时禁用)
        x_spec = self.spectral_branch(x_norm)  # (H, W, d_model)
        #x_spec = torch.zeros(h, w, self.d_model, device=x_norm.device, dtype=x_norm.dtype)

        # 空间分支 (消融实验：临时禁用)
        # 需要转换为卷积格式: (H, W, bands) -> (bands, H, W) -> (1, bands, H, W)
        x_spat = x_norm.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        x_spat = self.spatial_branch(x_spat)  # (1, d_model, H, W)
        x_spat = x_spat.squeeze(0).permute(1, 2, 0)  # (H, W, d_model)
        #x_spat = torch.zeros(h, w, self.d_model, device=x_norm.device, dtype=x_norm.dtype)

        # 3. 拼接
        x_cat = torch.cat([x_spec, x_spat], dim=-1)  # (H, W, 2*d_model)

        # 4. 1x1卷积融合
        # 转换维度: (H, W, 2*d_model) -> (1, 2*d_model, H, W)
        x_fused = x_cat.permute(2, 0, 1).unsqueeze(0)
        x_fused = self.fusion(x_fused)  # (1, d_model, H, W)
        x_fused = x_fused.squeeze(0).permute(1, 2, 0)  # (H, W, d_model)

        # 5. Mamba处理 (消融实验：临时禁用)
        # Reshape为序列: (H, W, d_model) -> (1, H*W, d_model)
        x_seq = x_fused.reshape(-1, self.d_model)  # (H*W, d_model)
        x_seq = x_seq.unsqueeze(0)  # (1, H*W, d_model)
        # 使用封装层处理（内部包含 Pre-Norm, 双向 S6Core, Dropout, 残差）
        x_mamba = self.mamba(x_seq)
        x_mamba = x_mamba.squeeze(0)  # (H*W, d_model)
        #x_mamba = x_fused.reshape(-1, self.d_model)  # (H*W, d_model) - 跳过Mamba直接传递

        # 6. 分类
        output = self.classifier(x_mamba)  # (H*W, num_classes)
        output = output.reshape(h, w, -1)  # (H, W, num_classes)

        return output
