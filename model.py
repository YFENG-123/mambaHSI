import torch
import torch.nn as nn

from mamba_ssm import Mamba2


class Mamba2HSIClassifier(nn.Module):
    def __init__(
        self,
        image_x=145,
        image_y=145,
        num_classes=17,
        bands=200,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.bands = bands

        # 改进的预处理层
        self.preprocess = nn.Sequential(  # (H, W, bands) -> (H, W, bands)
            nn.LayerNorm(bands),
            nn.Dropout(dropout_rate),
        )

        # 1.光谱特征提取器（用全连接处理每个像素的波段序列）
        self.d_model = 256
        self.spectral_proj_linear = nn.Sequential(  # (H*W, bands) -> (H*W, d_model)
            nn.Linear(self.bands, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.spectral_mamba = Mamba2(d_model=self.d_model)
        self.spectral_mamba_norm = nn.Sequential(  # (H*W, d_model) -> (H*W, d_model)
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 2.空间特征提取器（卷积+mamba）
        self.spatial_dim = 256
        self.conv_layers = nn.Sequential(  # (H, W, bands) -> (H, W, spatial_dim)
            nn.Conv2d(self.bands, self.spatial_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool2d((image_x, image_y)),  # 确保空间尺寸不变
        )
        self.spatial_mamba = Mamba2(d_model=self.spatial_dim)
        self.spatial_mamba_norm = (  # (H*W, spatial_dim) -> (H*W, spatial_dim)
            nn.Sequential(
                nn.LayerNorm(self.spatial_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
        )

        # 3.融合特征
        self.fusion_dim = 256
        self.fusion_linear = (  # (H*W, d_model + spatial_dim) -> (H*W, fusion_dim)
            nn.Sequential(
                nn.Linear(self.d_model + self.spatial_dim, self.fusion_dim),
                nn.LayerNorm(self.fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
        )

        # 4.拼接后的Mamba处理（一次mamba）
        self.fusion_mamba = Mamba2(  # (H*W, fusion_dim) -> (H*W, fusion_dim)
            d_model=self.fusion_dim
        )
        self.fusion_mamba_norm = (
            nn.Sequential(  # (H*W, fusion_dim) -> (H*W, fusion_dim)
                nn.LayerNorm(self.fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            )
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 200),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(200, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        h, w, _ = x.shape
        # 预处理
        x_norm = self.preprocess(x)  # (H, W, bands)

        """
        第一部分：光谱特征提取（用全连接处理每个像素的波段序列）
        """
        # 将 (H, W, bands) reshape 为 (H*W, bands) 以便处理每个像素的波段序列
        x_flat = x_norm.reshape(-1, self.bands)  # (H*W, bands)
        # 将每个像素的波段序列投影到 d_model 维度
        x_proj = self.spectral_proj_linear(x_flat)  # (H*W, d_model)
        # 用 Mamba2 处理所有像素（将像素作为序列）
        # Mamba2 期望输入格式为 (batch, seq_len, d_model)
        x_proj_seq = x_proj.unsqueeze(0)  # (1, H*W, d_model)
        x_mamba = self.spectral_mamba(x_proj_seq)  # (1, H*W, d_model)
        x_mamba = x_mamba.squeeze(0)  # (H*W, d_model)
        # 使用 spectral_mamba_norm 进行归一化和激活
        spectral_features_flat = self.spectral_mamba_norm(x_mamba)  # (H*W, d_model)
        # 重新reshape回 (H, W, d_model)
        spectral_features = spectral_features_flat.reshape(
            h, w, self.d_model
        )  # (H, W, d_model)

        """
        第二部分：空间特征提取（卷积）
        """
        # Conv2d 期望输入格式为 (batch, channels, H, W)
        # 将 (H, W, bands) 转换为 (bands, H, W)，然后添加batch维度
        x_conv = x_norm.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        spatial_features_conv = self.conv_layers(x_conv)  # (1, spatial_dim, H, W)
        # 移除batch维度并转换回 (H, W, spatial_dim)
        spatial_features = spatial_features_conv.squeeze(0).permute(
            1, 2, 0
        )  # (H, W, spatial_dim)

        # 卷积后的Mamba处理
        spatial_features_flat = spatial_features.reshape(
            -1, self.spatial_dim
        )  # (H*W, spatial_dim)
        # 将空间特征reshape为序列格式 (1, H*W, spatial_dim) 以便Mamba2处理
        spatial_features_seq = spatial_features_flat.unsqueeze(
            0
        )  # (1, H*W, spatial_dim)
        spatial_features_mamba = self.spatial_mamba(
            spatial_features_seq
        )  # (1, H*W, spatial_dim)
        spatial_features_flat = spatial_features_mamba.squeeze(0)  # (H*W, spatial_dim)
        spatial_features_flat = self.spatial_mamba_norm(
            spatial_features_flat
        )  # (H*W, spatial_dim)
        spatial_features = spatial_features_flat.reshape(
            h, w, self.spatial_dim
        )  # (H, W, spatial_dim)

        """
        第三部分：拼接特征并做一次mamba
        """
        # 特征拼接
        x_combined = torch.cat(
            [spectral_features, spatial_features], dim=-1
        )  # (H, W, d_model + spatial_dim)

        # 将 (H, W, d_model + spatial_dim) reshape 为 (H*W, d_model + spatial_dim) 以便mamba处理
        x_combined_flat = x_combined.reshape(
            -1, self.d_model + self.spatial_dim
        )  # (H*W, d_model + spatial_dim)
        # 融合特征
        x_fusion_flat = self.fusion_linear(x_combined_flat)  # (H*W, fusion_dim)
        # Mamba处理：以像素为序列
        x_fusion_seq = x_fusion_flat.unsqueeze(0)  # (1, H*W, fusion_dim)
        x_fusion_mamba = self.fusion_mamba(x_fusion_seq)  # (1, H*W, fusion_dim)
        x_fusion_flat = x_fusion_mamba.squeeze(0)  # (H*W, fusion_dim)
        x_fusion_flat = self.fusion_mamba_norm(x_fusion_flat)  # (H*W, fusion_dim)
        x_fusion = x_fusion_flat.reshape(h, w, self.fusion_dim)  # (H, W, fusion_dim)

        # 分类
        x_classifier_flat = x_fusion.reshape(-1, self.fusion_dim)  # (H*W, fusion_dim)
        output_flat = self.classifier(x_classifier_flat)  # (H*W, num_classes)
        output = output_flat.reshape(h, w, -1)  # (H, W, num_classes)
        return output
