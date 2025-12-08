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
        self.preprocess = nn.Sequential(
            nn.LayerNorm(bands),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        """
        光谱特征提取层
        """
        # 1x1卷积（简化结构以减少过拟合）
        self.conv1x1_spectrum = nn.Sequential(
            nn.Conv2d(bands, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        """
        空间特征提取层
        """
        # 3x3卷积
        self.conv3x3_spatial = nn.Sequential(
            nn.Conv2d(bands, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        """
        特征融合层
        """
        # 1x1卷积：特征融合
        self.conv1x1_fusion = nn.Sequential(
            nn.Conv2d(128 + 32, self.d_model, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        
        # Mamba正向层
        self.mamba_forward = Mamba2(d_model=self.d_model)
        # Mamba反向层
        self.mamba_backward = Mamba2(d_model=self.d_model)
        # Mamba归一化层
        self.mamba_norm = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        h, w, bands = x.shape
        # 预处理：归一化
        x_norm = self.preprocess(x)  # (H, W, bands)

        """
        光谱特征提取层
        """
        # 1x1卷积：光谱特征提取
        x_conv_spectrum = x_norm.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        x_conv_spectrum = self.conv1x1_spectrum(x_conv_spectrum)  # (1, 128, H, W)
        x_conv_spectrum = x_conv_spectrum.squeeze(0).permute(1, 2, 0)  # (H, W, 128)

        """
        空间特征提取层
        """
        # 3x3卷积：空间特征提取
        x_conv_spatial = x_norm.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        x_conv_spatial = self.conv3x3_spatial(x_conv_spatial)  # (1, 32, H, W)
        x_conv_spatial = x_conv_spatial.squeeze(0).permute(1, 2, 0)  # (H, W, 128)

        """
        特征融合层
        """
        # 拼接光谱特征和空间特征
        x_concat = torch.cat(
            [x_conv_spectrum, x_conv_spatial], dim=2
        )  # (H, W, 160)

        # 1x1卷积：特征融合（需要转换为Conv2d格式）
        x_concat_conv = x_concat.permute(2, 0, 1).unsqueeze(0)  # (1, 160, H, W)
        x_conv_fusion = self.conv1x1_fusion(x_concat_conv)  # (1, d_model, H, W)
        x_conv_fusion = x_conv_fusion.squeeze(0).permute(1, 2, 0)  # (H, W, d_model)

        # Reshape为序列用于Mamba: (H, W, d_model) -> (H*W, d_model)
        x_seq = x_conv_fusion.reshape(-1, self.d_model).unsqueeze(0)  # (1, H*W, d_model)
        
        # Mamba正向处理
        x_mamba_forward = self.mamba_forward(x_seq)  # (1, H*W, d_model)
        # Mamba反向处理（反转序列）
        x_mamba_backward = torch.flip(x_seq, dims=[1])  # (1, H*W, d_model) - 反转序列维度
        x_mamba_backward = self.mamba_backward(x_mamba_backward)  # (1, H*W, d_model)
        x_mamba_backward = torch.flip(x_mamba_backward, dims=[1])  # (1, H*W, d_model) - 反转回来保持原始顺序
        # 将正向和反向结果相加
        x_mamba = x_mamba_forward + x_mamba_backward  # (1, H*W, d_model)
        
        # 归一化
        x_mamba = self.mamba_norm(x_mamba)  # (1, H*W, d_model)
        x_mamba = x_mamba.squeeze(0)  # (H*W, d_model)

        # 分类
        output = self.classifier(x_mamba)  # (H*W, num_classes)
        output = output.reshape(h, w, -1)  # (H, W, num_classes)
        return output
