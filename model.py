import torch.nn as nn
from mamba_ssm import Mamba2


class Net(nn.Module):
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

        # 3x3卷积：空间特征提取，将bands维度投影到d_model维度
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(bands, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),
        )

        # 3x3卷积：进一步空间特征提取
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),
        )

        # 1x1卷积：特征变换
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),
        )

        # Mamba层
        self.mamba = Mamba2(d_model=self.d_model)
        self.mamba_norm = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        h, w, _ = x.shape
        # 预处理：归一化
        x_norm = self.preprocess(x)  # (H, W, bands)

        # 卷积：转换为Conv2d格式进行卷积
        x_conv = x_norm.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        # 3x3卷积：空间特征提取
        x_conv = self.conv3x3(x_conv)  # (1, d_model, H, W)
        # 3x3卷积：进一步空间特征提取
        x_conv = self.conv3x3_2(x_conv)  # (1, d_model, H, W)
        # 1x1卷积：特征变换
        x_conv = self.conv1x1(x_conv)  # (1, d_model, H, W)
        x_conv = x_conv.squeeze(0).permute(1, 2, 0)  # (H, W, d_model)

        # Reshape为序列用于Mamba: (H, W, d_model) -> (H*W, d_model)
        x_flat = x_conv.reshape(-1, self.d_model)  # (H*W, d_model)
        x_seq = x_flat.unsqueeze(0)  # (1, H*W, d_model)
        x_mamba = self.mamba(x_seq)  # (1, H*W, d_model)
        x_flat = x_mamba.squeeze(0)  # (H*W, d_model)

        # 分类
        output_flat = self.classifier(x_flat)  # (H*W, num_classes)
        output = output_flat.reshape(h, w, -1)  # (H, W, num_classes)
        return output
