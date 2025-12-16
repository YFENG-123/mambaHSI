import torch
import torch.nn as nn
from .spatial_branch import SpatialBranchModule
from .spectral_branch import SpectralBranchModule
from .bidirectional_mamba import BidirectionalMamba
from .class_aware_enhancement import ClassAwareFeatureEnhancement
from .classifier import Classifier


class MambaHSINet(nn.Module):
    """
    高光谱图像分类网络（双分支架构 + 多尺度空间特征 + 类别感知增强）

    架构流程：
    1. 归一化
    2. 双分支处理：
       - 光谱分支：全连接 (Linear)
       - 空间分支 (SpatialBranchModule)：
            - 并行 3x3, 5x5, 7x7 卷积
            - 拼接 -> 1x1 卷积压缩
    3. 拼接两个分支特征
    4. 1x1卷积融合
    5. Mamba处理 (BidirectionalMamba)
    6. 类别感知的特征增强 (ClassAwareFeatureEnhancement)
       - 使用类别原型增强特征表示
       - 特别关注少数类别的特征增强
    7. LayerNorm -> Dropout -> 分类器
    """

    def __init__(
        self,
        image_x=145,
        image_y=145,
        num_classes=17,
        bands=200,
        dropout_rate=0.5,
        d_model=64,
    ):
        super().__init__()
        self.bands = bands
        self.image_x = image_x
        self.image_y = image_y
        self.d_model = d_model

        # 预处理层：归一化（使用GroupNorm）
        # GroupNorm需要(B, C, H, W)格式，所以需要在forward中转换维度
        self.preprocess = nn.GroupNorm(1, bands)  # num_groups=1等价于LayerNorm

        # 光谱分支：增强的线性映射 (Linear + GELU + Dropout)
        # 输入 (H, W, bands) -> 输出 (H, W, d_model)
        self.spectral_branch = SpectralBranchModule(bands, d_model, dropout_rate)

        # 空间分支：多尺度卷积模块
        # 输入 (bands, H, W) -> 输出 (d_model, H, W)
        self.spatial_branch = SpatialBranchModule(bands, d_model, dropout_rate)

        # 融合层：1x1卷积
        # 输入拼接后的特征 (2*d_model, H, W) -> 输出 (d_model, H, W)
        self.fusion_conv = nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=True)

        # 双向 Mamba模块
        self.bidirectional_mamba = BidirectionalMamba(
            d_model=d_model,
            dropout_rate=dropout_rate,
        )

        # 类别感知的特征增强模块
        # 在双向Mamba后添加，使用类别原型增强特征表示
        self.class_aware_enhancement = ClassAwareFeatureEnhancement(
            d_model=d_model,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            temperature=0.1,  # 温度参数，控制注意力分布的锐度
            enhancement_scale=0.5,  # 增强强度，控制原型增强的权重
        )

        # 分类器模块
        self.classifier = Classifier(
            d_model=d_model,
            num_classes=num_classes,
            classifier_hidden=64,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        """
        Args:
            x: 输入高光谱数据 (H, W, bands)
        Returns:
            分类结果 (H, W, num_classes)
        """
        h, w, _ = x.shape

        # 1. 预处理：归一化（使用GroupNorm）
        # GroupNorm需要(B, C, H, W)格式，所以需要转换维度
        # (H, W, bands) -> (1, bands, H, W)
        x_for_norm = x.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        x_norm_tensor = self.preprocess(x_for_norm)  # (1, bands, H, W)
        # 转换回 (H, W, bands)
        x_norm = x_norm_tensor.squeeze(0).permute(1, 2, 0)  # (H, W, bands)

        # 2. 双分支处理

        # 光谱分支 (全连接)
        # Linear作用在最后一个维度
        x_spec = self.spectral_branch(x_norm)  # (H, W, d_model)

        # 空间分支 (多尺度卷积)
        # 需要转换为卷积格式: (H, W, bands) -> (bands, H, W) -> (1, bands, H, W)
        x_spat = x_norm.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        x_spat = self.spatial_branch(x_spat)  # (1, d_model, H, W)
        x_spat = x_spat.squeeze(0).permute(1, 2, 0)  # (H, W, d_model)

        # 3. 拼接
        x_cat = torch.cat([x_spec, x_spat], dim=-1)  # (H, W, 2*d_model)

        # 4. 1x1卷积融合
        # 转换维度: (H, W, 2*d_model) -> (1, 2*d_model, H, W)
        x_fused = x_cat.permute(2, 0, 1).unsqueeze(0)
        x_fused = self.fusion_conv(x_fused)  # (1, d_model, H, W)
        x_fused = x_fused.squeeze(0).permute(1, 2, 0)  # (H, W, d_model)

        # 5. Mamba处理 (双向 + Pre-Norm Residual)
        # Reshape为序列: (H, W, d_model) -> (1, H*W, d_model)
        x_seq = x_fused.reshape(-1, self.d_model)  # (H*W, d_model)
        x_seq = x_seq.unsqueeze(0)  # (1, H*W, d_model)

        # 移除显式位置编码，依赖Mamba自身的隐式位置建模能力

        # 双向Mamba处理
        x_mamba = self.bidirectional_mamba(x_seq)  # (1, H*W, d_model)
        x_mamba = x_mamba.squeeze(0)  # (H*W, d_model)

        # 类别感知的特征增强
        # 使用类别原型来增强特征表示，特别关注少数类别
        x_enhanced = self.class_aware_enhancement(x_mamba)  # (H*W, d_model)

        # 6. 分类
        output = self.classifier(x_enhanced)  # (H*W, num_classes)
        output = output.reshape(h, w, -1)  # (H, W, num_classes)

        return output
