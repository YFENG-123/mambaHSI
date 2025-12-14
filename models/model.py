import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from .attention import MultiAttention
from .multi_scale_asymmetric_conv import MultiScaleAsymmetricDepthConv
from .depthwise_separable_aspp import DepthwiseSeparableASPP
from .depthwise_separable_square_conv import DepthwiseSeparableSquareConv


class Net(nn.Module):
    """
    高光谱图像分类网络（优化版本）
    
    架构流程：
    归一化 -> 通道注意力 + 空间注意力 -> 三分支局部特征提取 -> 特征融合 -> 单向Mamba全局建模 -> 分类
    
    三分支优化配置：
    - 分支1（方形卷积）：kernel_sizes=[3, 5]，输出48通道，提取局部细节特征
    - 分支2（ASPP膨胀卷积）：dilations=[6, 9, 12]，输出40通道，提取多尺度上下文
    - 分支3（条状卷积）：kernel_sizes=[7, 11, 15]，输出24通道，提取方向性结构特征
    
    总计：112通道融合后输入Mamba进行全局序列建模
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
        self.image_x = image_x
        self.image_y = image_y

        # 预处理层：归一化
        self.preprocess = nn.LayerNorm(bands)

        # 多头注意力模块：组合通道注意力和空间注意力
        self.multi_attention = MultiAttention(
            channels=bands, reduction=16, spatial_kernel_size=7
        )

        """
        三分支特征提取（优化配置）
        """

        # 分支1：深度可分离方形卷积（聚焦局部特征）
        # 优化：去掉7×7，更聚焦局部细节 [3, 5]
        # 压缩层已集成在模块内部
        branch1_out_channels = 48  # 局部特征：48通道

        self.branch1_square = DepthwiseSeparableSquareConv(
            in_channels=bands,
            out_channels=branch1_out_channels,
            kernel_sizes=[3, 5],  # 优化：去掉7，聚焦局部
            dropout_rate=dropout_rate,
        )

        # 分支2：深度可分离ASPP（多尺度上下文）
        # 优化：合理膨胀率 [6, 9, 12]，避免过度稀疏
        # 压缩层已集成在模块内部
        branch2_out_channels = 40  # 多尺度上下文：40通道
        self.branch2_aspp = DepthwiseSeparableASPP(
            in_channels=bands,
            out_channels=branch2_out_channels,
            dilations=[6, 9, 12],  # 优化：避免过度膨胀（原[9, 11, 13]）
            dropout_rate=dropout_rate,
        )

        # 分支3：多尺度非对称深度可分离卷积（方向性特征）
        # 优化：更实用的尺度 [7, 11, 15]，从较小的条状卷积开始
        # 压缩层已集成在模块内部
        branch3_out_channels = 24  # 方向性特征：24通道
        self.branch3_asymmetric = MultiScaleAsymmetricDepthConv(
            in_channels=bands,
            out_channels=branch3_out_channels,
            kernel_sizes=[7, 11, 15],  # 优化：更实用的尺度（原[15, 17, 19]）
            dropout_rate=dropout_rate,
        )

        # 计算融合层的输入通道数（总计：48 + 40 + 24 = 112通道）
        fusion_input_channels = (
            branch1_out_channels + branch2_out_channels + branch3_out_channels
        )

        """
        特征融合层
        """
        # 1x1卷积：将三个分支的特征融合，输出d_model-2维（添加坐标信息后为d_model维）
        self.fusion = nn.Sequential(
            nn.Conv2d(
                fusion_input_channels, self.d_model - 2, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm2d(self.d_model - 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        """
        Mamba处理层（单向）
        注意：输入维度为d_model（融合特征d_model-2 + 坐标信息2 = d_model）
        """
        # Mamba层（输入维度为d_model，已包含坐标信息）
        self.mamba = Mamba2(d_model=self.d_model)
        # Mamba后归一化层
        self.mamba_norm = nn.Sequential(
            nn.LayerNorm(self.d_model), nn.GELU(), nn.Dropout(dropout_rate)
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
        x_branch1 = self.branch1_square(x_attended)  # (1, 48, H, W)

        # 分支2：深度可分离ASPP（压缩层已在模块内部）
        x_branch2 = self.branch2_aspp(x_attended)  # (1, 40, H, W)

        # 分支3：多尺度非对称深度可分离卷积（压缩层已在模块内部）
        x_branch3 = self.branch3_asymmetric(x_attended)  # (1, 24, H, W)

        """
        特征融合
        """
        # 拼接三个分支的特征（48 + 40 + 24 = 112通道）
        x_concat = torch.cat([x_branch1, x_branch2, x_branch3], dim=1)  # (1, 112, H, W)

        # 1x1卷积融合
        x_fusion = self.fusion(x_concat)  # (1, d_model - 2, H, W)
        x_fusion = x_fusion.squeeze(0).permute(1, 2, 0)  # (H, W, d_model - 2)

        """
        添加坐标信息（x轴和y轴）
        """
        # 创建坐标网格并归一化到[-1, 1]
        # y坐标（行坐标）：从0到H-1，归一化到[-1, 1]
        y_coords = torch.arange(h, dtype=torch.float32, device=x_fusion.device)
        if h > 1:
            y_coords = (y_coords / (h - 1) * 2 - 1).unsqueeze(1).expand(h, w)  # (H, W)
        else:
            y_coords = torch.zeros(h, w, dtype=torch.float32, device=x_fusion.device)
        
        # x坐标（列坐标）：从0到W-1，归一化到[-1, 1]
        x_coords = torch.arange(w, dtype=torch.float32, device=x_fusion.device)
        if w > 1:
            x_coords = (x_coords / (w - 1) * 2 - 1).unsqueeze(0).expand(h, w)  # (H, W)
        else:
            x_coords = torch.zeros(h, w, dtype=torch.float32, device=x_fusion.device)
        
        # 将坐标信息作为额外的通道拼接
        x_coords = x_coords.unsqueeze(-1)  # (H, W, 1)
        y_coords = y_coords.unsqueeze(-1)  # (H, W, 1)
        x_fusion_with_coords = torch.cat([x_fusion, x_coords, y_coords], dim=-1)  # (H, W, d_model)

        """
        Mamba处理（单向）
        """
        # Reshape为序列用于Mamba: (H, W, d_model) -> (H*W, d_model)
        x_seq = x_fusion_with_coords.reshape(-1, self.d_model).unsqueeze(0)  # (1, H*W, d_model)

        # Mamba处理（输入包含坐标信息，维度为d_model）
        x_mamba = self.mamba(x_seq)  # (1, H*W, d_model)
        # Mamba后归一化
        x_mamba = self.mamba_norm(x_mamba)  # (1, H*W, d_model)
        x_mamba = x_mamba.squeeze(0)  # (H*W, d_model)

        """
        分类
        """
        output = self.classifier(x_mamba)  # (H*W, num_classes)
        output = output.reshape(h, w, -1)  # (H, W, num_classes)

        return output
