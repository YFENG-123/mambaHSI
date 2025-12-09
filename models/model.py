import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from .attention import ChannelAttention, SpatialAttention
from .multi_scale_asymmetric_conv import MultiScaleAsymmetricDepthConv
from .depthwise_separable_aspp import DepthwiseSeparableASPP


class Net(nn.Module):
    """
    高光谱图像分类网络
    使用归一化 -> 通道注意力 -> 空间注意力 -> 双分支（深度可分离ASPP + 多尺度非对称深度可分离卷积） -> 双向Mamba -> 分类
    """
    def __init__(
        self,
        image_x=145,
        image_y=145,
        num_classes=17,
        bands=200,
        dropout_rate=0.5,
        d_model=256,
    ):
        super().__init__()
        self.bands = bands
        self.d_model = d_model

        # 预处理层：归一化
        self.preprocess = nn.LayerNorm(bands)

        # 通道注意力模块：对原始高光谱通道应用注意力
        self.channel_attention = ChannelAttention(channels=bands, reduction=16)
        
        # 空间注意力模块：对通道注意力后的高光谱数据应用空间注意力
        self.spatial_attention = SpatialAttention(kernel_size=7)

        """
        双分支特征提取
        """
        # 分支1：深度可分离ASPP
        self.branch1_aspp = DepthwiseSeparableASPP(
            in_channels=bands,
            out_channels=d_model // 2,
            dilations=[1, 6, 12, 18],
            dropout_rate=dropout_rate
        )
        
        # 分支2：多尺度非对称深度可分离卷积
        self.branch2_asymmetric = MultiScaleAsymmetricDepthConv(
            channels=bands,
            dropout_rate=dropout_rate
        )
        # 将非对称卷积输出投影到d_model//2维
        self.branch2_proj = nn.Sequential(
            nn.Conv2d(bands, d_model // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        """
        特征融合层
        """
        # 1x1卷积：将两个分支的特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(d_model // 2 + d_model // 2, self.d_model, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        """
        Mamba处理层
        """
        # Mamba正向层
        self.mamba_forward = Mamba2(d_model=self.d_model)
        # Mamba反向层
        self.mamba_backward = Mamba2(d_model=self.d_model)
        # Mamba融合归一化层
        self.mamba_norm_fusion = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        """
        分类器
        """
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
        """
        Args:
            x: 输入高光谱数据 (H, W, bands)
        Returns:
            分类结果 (H, W, num_classes)
        """
        h, w, bands = x.shape
        
        # 预处理：归一化
        x_norm = self.preprocess(x)  # (H, W, bands)

        """
        注意力机制
        """
        # 转换为Conv2d格式以应用注意力
        x_norm_conv = x_norm.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        
        # 通道注意力
        x_channel_attended = self.channel_attention(x_norm_conv)  # (1, bands, H, W)
        
        # 空间注意力
        x_attended = self.spatial_attention(x_channel_attended)  # (1, bands, H, W)

        """
        双分支特征提取
        """
        # 分支1：深度可分离ASPP
        x_branch1 = self.branch1_aspp(x_attended)  # (1, d_model//2, H, W)
        
        # 分支2：多尺度非对称深度可分离卷积
        x_branch2 = self.branch2_asymmetric(x_attended)  # (1, bands, H, W)
        x_branch2 = self.branch2_proj(x_branch2)  # (1, d_model//2, H, W)

        """
        特征融合
        """
        # 拼接两个分支的特征
        x_concat = torch.cat([x_branch1, x_branch2], dim=1)  # (1, d_model, H, W)
        
        # 1x1卷积融合
        x_fusion = self.fusion(x_concat)  # (1, d_model, H, W)
        x_fusion = x_fusion.squeeze(0).permute(1, 2, 0)  # (H, W, d_model)

        """
        Mamba处理
        """
        # Reshape为序列用于Mamba: (H, W, d_model) -> (H*W, d_model)
        x_seq = x_fusion.reshape(-1, self.d_model).unsqueeze(0)  # (1, H*W, d_model)
        
        # Mamba正向处理
        x_mamba_forward = self.mamba_forward(x_seq)  # (1, H*W, d_model)
        
        # Mamba反向处理（反转序列）
        x_mamba_backward = torch.flip(x_seq, dims=[1])  # (1, H*W, d_model)
        x_mamba_backward = self.mamba_backward(x_mamba_backward)  # (1, H*W, d_model)
        x_mamba_backward = torch.flip(x_mamba_backward, dims=[1])  # (1, H*W, d_model)

        # 将正向和反向结果相加
        x_mamba = x_mamba_forward + x_mamba_backward  # (1, H*W, d_model)
        x_mamba = self.mamba_norm_fusion(x_mamba)  # (1, H*W, d_model)
        x_mamba = x_mamba.squeeze(0).reshape(-1, self.d_model)  # (H*W, d_model)

        """
        分类
        """
        output = self.classifier(x_mamba)  # (H*W, num_classes)
        output = output.reshape(h, w, -1)  # (H, W, num_classes)
        
        return output

