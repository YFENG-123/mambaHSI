import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from .attention import MultiAttention
from .multi_scale_asymmetric_conv import MultiScaleAsymmetricDepthConv
from .depthwise_separable_aspp import DepthwiseSeparableASPP
from .depthwise_separable_square_conv import DepthwiseSeparableSquareConv


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
        d_model=128,
    ):
        super().__init__()
        self.bands = bands
        self.d_model = d_model

        # 预处理层：归一化
        self.preprocess = nn.LayerNorm(bands)

        # 多头注意力模块：组合通道注意力和空间注意力
        self.multi_attention = MultiAttention(channels=bands, reduction=16, spatial_kernel_size=7)

        """
        三分支特征提取
        """
        # 统一压缩维度：将每个分支的输出压缩到64通道
        branch_out_channels = 64
        
        # 分支1：深度可分离方形卷积（可配置列表，默认[3, 5, 7]）
        # 压缩层已集成在模块内部
        self.branch1_square = DepthwiseSeparableSquareConv(
            in_channels=bands,
            out_channels=branch_out_channels,
            kernel_sizes=[3, 5],  # 可配置列表
            dropout_rate=dropout_rate
        )

        # 分支2：深度可分离ASPP（不进行融合，保留所有膨胀卷积分支）
        # 压缩层已集成在模块内部
        self.branch2_aspp = DepthwiseSeparableASPP(
            in_channels=bands,
            out_channels=branch_out_channels,
            dilations=[9, 11, 13],
            dropout_rate=dropout_rate
        )
        
        # 分支3：多尺度非对称深度可分离卷积（不进行融合，保留所有非对称卷积对）
        # 压缩层已集成在模块内部
        self.branch3_asymmetric = MultiScaleAsymmetricDepthConv(
            in_channels=bands,
            out_channels=branch_out_channels,
            kernel_sizes=[15, 17, 19],  # 可配置列表
            dropout_rate=dropout_rate
        )
        
        # 计算融合层的输入通道数（压缩后，每个分支输出64通道）
        fusion_input_channels = branch_out_channels * 3  # 64 * 3 = 192

        """
        特征融合层
        """
        # 1x1卷积：将三个分支的特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_input_channels, self.d_model, kernel_size=1, stride=1, padding=0),
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
        
        # 多头注意力：组合通道注意力和空间注意力
        x_attended = self.multi_attention(x_norm_conv)  # (1, bands, H, W)

        """
        三分支特征提取
        """
        # 分支1：深度可分离方形卷积（压缩层已在模块内部）
        x_branch1 = self.branch1_square(x_attended)  # (1, 64, H, W)
        
        # 分支2：深度可分离ASPP（压缩层已在模块内部）
        x_branch2 = self.branch2_aspp(x_attended)  # (1, 64, H, W)
        
        # 分支3：多尺度非对称深度可分离卷积（压缩层已在模块内部）
        x_branch3 = self.branch3_asymmetric(x_attended)  # (1, 64, H, W)

        """
        特征融合
        """
        # 拼接三个分支的特征（每个分支64通道，共192通道）
        x_concat = torch.cat([x_branch1, x_branch2, x_branch3], dim=1)  # (1, 192, H, W)
        
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

