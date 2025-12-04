import torch
import torch.nn as nn

from mamba_ssm import Mamba2


class Mamba2HSIClassifier(nn.Module):
    def __init__(
        self,
        num_classes=17,
        d_model=128,
        bands=200,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.d_model = d_model
        # 改进的预处理层
        self.preprocess = nn.Sequential(
            nn.LayerNorm(bands),
            nn.Dropout(dropout_rate),
        )

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(bands, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 增强的卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(bands, 32, kernel_size=3, padding=2),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
            nn.AdaptiveAvgPool2d((145, 145)),  # 确保空间尺寸不变
        )

        # Mamba2模型输入投影
        new_in_features = 128 + 64  # 特征提取器输出 + 卷积层最终输出
        self.mamba_input_proj = nn.Sequential(
            nn.Linear(new_in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Mamba2块
        self.mamba_block = Mamba2(d_model=d_model)

        # 增强的分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        h, w, _ = x.shape
        # 预处理
        x_norm = self.preprocess(x)  # 预处理归一化 (145, 145, 200)

        # 特征提取
        x_features = self.feature_extractor(x_norm)  # 特征提取 (145, 145, 200)

        # 卷积
        x_conv_input = x_norm.permute(2, 0, 1)  # (200, 145, 145)
        conv_features = self.conv_layers(x_conv_input)  # 卷积 (32, 145, 145)
        conv_features = conv_features.permute(1, 2, 0)  # (145, 145, 32)

        # 特征融合
        x_combined = torch.cat([x_features, conv_features], dim=-1)

        # Mamba处理
        x_flat = x_combined.reshape(1, h * w, -1)
        x_proj = self.mamba_input_proj(x_flat)
        x_mamba = self.mamba_block(x_proj)
        x_restored = x_mamba.reshape(h, w, self.d_model)

        # 分类
        output = self.classifier(x_restored)
        return output
