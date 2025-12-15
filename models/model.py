import torch
import torch.nn as nn
from .s6_core import S6Core
from .spatial_branch import SpatialBranchModule
from .spectral_branch import SpectralBranchModule


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
        dropout_rate=0.5,
        d_model=64,
    ):
        super().__init__()
        self.bands = bands
        self.image_x = image_x
        self.image_y = image_y
        self.d_model = d_model

        # 预处理层：归一化
        self.preprocess = nn.LayerNorm(bands)

        # 光谱分支：增强的线性映射 (Linear + GELU + Dropout)
        # 输入 (H, W, bands) -> 输出 (H, W, d_model)
        self.spectral_branch = SpectralBranchModule(bands, d_model, dropout_rate)

        # 空间分支：多尺度卷积模块
        # 输入 (bands, H, W) -> 输出 (d_model, H, W)
        self.spatial_branch = SpatialBranchModule(bands, d_model, dropout_rate)

        # 融合层：1x1卷积
        # 输入拼接后的特征 (2*d_model, H, W) -> 输出 (d_model, H, W)
        self.fusion_conv = nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=True)

        # 优化：双向 Mamba (关键修改：必须使用双向扫描以捕获全图上下文)
        self.mamba_fwd = S6Core(d_model=d_model)
        self.mamba_bwd = S6Core(d_model=d_model)
        
        self.mamba_norm = nn.LayerNorm(d_model)
        self.mamba_dropout = nn.Dropout(dropout_rate)

        # 分类器
        classifier_hidden = 64  # 隐藏层维度
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden, num_classes),
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
        x_cat_conv = x_cat.permute(2, 0, 1).unsqueeze(0)
        x_fused = self.fusion_conv(x_cat_conv)  # (1, d_model, H, W)
        x_fused = x_fused.squeeze(0).permute(1, 2, 0)  # (H, W, d_model)

        # 5. Mamba处理 (双向 + Pre-Norm Residual)
        # Reshape为序列: (H, W, d_model) -> (1, H*W, d_model)
        x_seq = x_fused.reshape(-1, self.d_model)  # (H*W, d_model)
        x_seq = x_seq.unsqueeze(0)  # (1, H*W, d_model)

        # 移除显式位置编码，依赖Mamba自身的隐式位置建模能力

        # Pre-Norm Residual Block (这是最稳定的深层结构)
        residual = x_seq
        x_norm = self.mamba_norm(x_seq)

        # Bidirectional Mamba
        # 前向流
        x_fwd = self.mamba_fwd(x_norm)
        
        # 后向流 (翻转输入 -> 处理 -> 翻转输出)
        x_bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        
        # 融合: 相加
        x_mamba = x_fwd + x_bwd

        x_mamba = self.mamba_dropout(x_mamba)
        x_mamba = x_mamba + residual

        x_mamba = x_mamba.squeeze(0)  # (H*W, d_model)

        # 6. 分类
        output = self.classifier(x_mamba)  # (H*W, num_classes)
        output = output.reshape(h, w, -1)  # (H, W, num_classes)

        return output
