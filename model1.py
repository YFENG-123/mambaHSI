import torch
import torch.nn as nn
from mamba_ssm import Mamba2


class Net1(nn.Module):
    def __init__(
        self,
        image_x=145,
        image_y=145,
        num_classes=17,
        bands=200,
        dropout_rate=0.3,
        d_model=256,
    ):
        super().__init__()
        self.bands = bands
        self.d_model = d_model

        # 预处理层：只归一化
        self.preprocess = nn.LayerNorm(bands)  # (H, W, bands) -> (H, W, bands)

        """
        光谱特征提取层
        """
        self.spectral_dim = 200  # 光谱特征维度
        # 1x1卷积：将bands维度投影到spectral_dim维度
        self.conv1x1_spectrum = nn.Sequential(
            nn.Conv2d(bands, self.spectral_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.spectral_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        """
        空间特征提取层
        """
        self.spatial_dim = 100  # 空间特征维度
        # 3x3卷积：空间特征提取，将bands维度投影到spatial_dim维度
        self.conv3x3_spatial = nn.Sequential(
            nn.Conv2d(bands, self.spatial_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.spatial_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 3x3卷积：进一步空间特征提取
        self.conv3x3_spatial_2 = nn.Sequential(
            nn.Conv2d(
                self.spatial_dim, self.spatial_dim, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(self.spatial_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        """
        特征融合层
        """
        # Mamba层
        self.mamba = Mamba2(d_model=self.spectral_dim + self.spatial_dim)
        self.mamba_norm = nn.Sequential(
            nn.LayerNorm(self.spectral_dim + self.spatial_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.spectral_dim + self.spatial_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        h, w, _ = x.shape
        # 预处理：归一化
        x_norm = self.preprocess(x)  # (H, W, bands)

        """
        光谱特征提取层
        """
        # 1x1卷积：光谱特征提取
        x_conv_spectrum = x_norm.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        x_conv_spectrum = self.conv1x1_spectrum(x_conv_spectrum)  # (1, d_model, H, W)
        x_conv_spectrum = x_conv_spectrum.squeeze(0).permute(1, 2, 0)  # (H, W, d_model)

        """
        空间特征提取层
        """
        # 3x3卷积：空间特征提取
        x_conv_spatial = x_norm.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        x_conv_spatial = self.conv3x3_spatial(x_conv_spatial)  # (1, d_model, H, W)
        # 3x3卷积：进一步空间特征提取
        x_conv_spatial = self.conv3x3_spatial_2(x_conv_spatial)  # (1, d_model, H, W)
        x_conv_spatial = x_conv_spatial.squeeze(0).permute(1, 2, 0)  # (H, W, d_model)

        """
        特征融合层
        """
        # 拼接光谱特征和空间特征
        x_concat = torch.cat(
            [x_conv_spectrum, x_conv_spatial], dim=2
        )  # (H, W, d_model * 2)
        # Reshape为序列用于Mamba: (H, W, d_model * 2) -> (H*W, d_model * 2)
        x_flat = x_concat.reshape(
            -1, self.spectral_dim + self.spatial_dim
        )  # (H*W, d_model * 2)
        x_seq = x_flat.unsqueeze(0)  # (1, H*W, spectral_dim + spatial_dim)
        # Mamba层
        x_mamba = self.mamba(x_seq)  # (1, H*W, spectral_dim + spatial_dim)
        x_mamba_flat = x_mamba.squeeze(0)  # (H*W, spectral_dim + spatial_dim)

        # 分类
        output_flat = self.classifier(x_mamba_flat)  # (H*W, 128)
        output = output_flat.reshape(h, w, -1)  # (H, W, 128)
        return output
