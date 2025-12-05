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

        # 1.光谱特征提取器：1x1卷积 -> Mamba
        self.d_model = 256
        self.spectral_conv1x1 = nn.Sequential(
            nn.Conv2d(self.bands, self.d_model, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.spectral_mamba = Mamba2(d_model=self.d_model)
        self.spectral_mamba_norm = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 2.空间特征提取器：3x3卷积 -> 1x1卷积
        self.spatial_dim = 256
        self.spatial_conv3x3 = nn.Sequential(
            nn.Conv2d(self.bands, self.spatial_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.spatial_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.spatial_conv1x1 = nn.Sequential(
            nn.Conv2d(self.spatial_dim, self.spatial_dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.spatial_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 3.特征融合：使用1x1卷积融合特征
        self.fusion_dim = 256
        # 使用1x1卷积将拼接的特征融合到fusion_dim维度
        self.fusion_conv1x1 = nn.Sequential(
            nn.Conv2d(
                self.d_model + self.spatial_dim,
                self.fusion_dim,
                kernel_size=1,
                padding=0,
            ),
            nn.BatchNorm2d(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        # 4.特征融合后：Mamba处理
        self.fusion_mamba = Mamba2(d_model=self.fusion_dim)
        self.fusion_mamba_norm = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 5.分类器
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
        第一部分：光谱特征提取 - 1x1卷积 -> Mamba
        """
        # 临时添加batch维度用于Conv2d: (H, W, bands) -> (1, bands, H, W)
        x_spectral = x_norm.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        # 1x1卷积
        x_spectral = self.spectral_conv1x1(x_spectral)  # (1, d_model, H, W)
        # 移除batch维度转回3D: (1, d_model, H, W) -> (H, W, d_model)
        x_spectral = x_spectral.squeeze(0).permute(1, 2, 0)  # (H, W, d_model)
        # Reshape为序列用于Mamba: (H, W, d_model) -> (H*W, d_model)
        x_spectral_flat = x_spectral.reshape(-1, self.d_model)  # (H*W, d_model)
        # Mamba处理: 临时添加batch维度
        x_spectral_seq = x_spectral_flat.unsqueeze(0)  # (1, H*W, d_model)
        x_spectral_mamba = self.spectral_mamba(x_spectral_seq)  # (1, H*W, d_model)
        x_spectral_flat = x_spectral_mamba.squeeze(0)  # (H*W, d_model)
        # 归一化
        spectral_features_flat = self.spectral_mamba_norm(
            x_spectral_flat
        )  # (H*W, d_model)
        # 转回3D格式
        spectral_features = spectral_features_flat.reshape(
            h, w, self.d_model
        )  # (H, W, d_model)

        """
        第二部分：空间特征提取 - 3x3卷积 -> 1x1卷积
        """
        # 临时添加batch维度用于Conv2d: (H, W, bands) -> (1, bands, H, W)
        x_spatial = x_norm.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        # 3x3卷积
        x_spatial = self.spatial_conv3x3(x_spatial)  # (1, spatial_dim, H, W)
        # 1x1卷积
        x_spatial = self.spatial_conv1x1(x_spatial)  # (1, spatial_dim, H, W)
        # 移除batch维度转回3D: (1, spatial_dim, H, W) -> (H, W, spatial_dim)
        spatial_features = x_spatial.squeeze(0).permute(1, 2, 0)  # (H, W, spatial_dim)

        """
        第三部分：特征融合 -> 1x1卷积 -> Mamba -> 分类
        """
        # 特征拼接: (H, W, d_model) + (H, W, spatial_dim) -> (H, W, d_model + spatial_dim)
        x_combined = torch.cat(
            [spectral_features, spatial_features], dim=-1
        )  # (H, W, d_model + spatial_dim)

        # 使用1x1卷积融合特征: (H, W, d_model + spatial_dim) -> (1, d_model + spatial_dim, H, W) -> (1, fusion_dim, H, W)
        x_combined = x_combined.permute(2, 0, 1).unsqueeze(
            0
        )  # (1, d_model + spatial_dim, H, W)
        x_fusion = self.fusion_conv1x1(x_combined)  # (1, fusion_dim, H, W)
        # 转回3D: (1, fusion_dim, H, W) -> (H, W, fusion_dim)
        x_fusion = x_fusion.squeeze(0).permute(1, 2, 0)  # (H, W, fusion_dim)

        # Mamba处理: (H, W, fusion_dim) -> (H*W, fusion_dim) -> (1, H*W, fusion_dim)
        x_fusion_flat = x_fusion.reshape(-1, self.fusion_dim)  # (H*W, fusion_dim)
        x_fusion_seq = x_fusion_flat.unsqueeze(0)  # (1, H*W, fusion_dim)
        x_fusion_mamba = self.fusion_mamba(x_fusion_seq)  # (1, H*W, fusion_dim)
        x_fusion_flat = x_fusion_mamba.squeeze(0)  # (H*W, fusion_dim)
        x_fusion_flat = self.fusion_mamba_norm(x_fusion_flat)  # (H*W, fusion_dim)

        # 分类
        output_flat = self.classifier(x_fusion_flat)  # (H*W, num_classes)
        output = output_flat.reshape(h, w, -1)  # (H, W, num_classes)
        return output
