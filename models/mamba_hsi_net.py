import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from .attention import ChannelAttention, SpatialAttention
from .strip_conv import MultiScaleStripConvolution
from .depthwise_separable_aspp import DepthwiseSeparableASPP
from .depthwise_separable_square_conv import DepthwiseSeparableSquareConv


class ResidualBlock(nn.Module):
    """
    残差连接模块
    用于在特征提取过程中保留原始信息，防止梯度消失
    """
    def __init__(self, module):
        super(ResidualBlock, self).__init__()
        self.module = module
    
    def forward(self, x):
        return x + self.module(x)


class FeatureExtractionBranch(nn.Module):
    """
    特征提取分支
    每个分支内部包含残差连接，增强特征表达能力
    """
    def __init__(self, in_channels, out_channels, module, use_residual=True):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            module: 特征提取模块（条状卷积、膨胀卷积或深度可分离卷积）
            use_residual: 是否使用残差连接
        """
        super(FeatureExtractionBranch, self).__init__()
        self.use_residual = use_residual
        
        # 如果输入输出通道数不同，需要投影层
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.proj = nn.Identity()
        
        # 特征提取模块
        self.feature_extractor = module
        
        # 残差连接
        if use_residual:
            self.feature_extractor = ResidualBlock(self.feature_extractor)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, in_channels, H, W)
        Returns:
            提取的特征 (B, out_channels, H, W)
        """
        # 投影到目标通道数
        x_proj = self.proj(x)  # (B, out_channels, H, W)
        
        # 特征提取（可能包含残差连接）
        if self.use_residual:
            output = self.feature_extractor(x_proj)  # (B, out_channels, H, W)
        else:
            output = self.feature_extractor(x)  # (B, out_channels, H, W)
            # 如果通道数不同，需要投影
            if x.shape[1] != output.shape[1]:
                output = self.proj(output)
        
        return output


class MambaHSINet(nn.Module):
    """
    高光谱图像像素级分割网络
    
    架构设计理念：
    1. 通道注意力：在预处理阶段筛选重要的光谱通道，减少冗余信息
    2. 多分支特征提取（每个分支内部有残差连接）：
       - 条状卷积：提取长距离空间依赖（水平/垂直方向），捕获条状结构特征
       - 膨胀卷积：提取多尺度上下文信息，扩大感受野
       - 深度可分离卷积：提取局部特征，保持计算效率
    3. 空间注意力：融合多分支特征，突出重要的空间位置
    4. Mamba S6模块：建模长距离序列依赖，捕获全局上下文
    5. 跳跃连接：保留原始输入信息，防止信息丢失
    6. 分类器：最终像素级分类
    
    每个模块的作用：
    - 通道注意力：光谱维度特征选择，突出重要波段
    - 条状卷积：空间维度长距离依赖提取，捕获条状/线性结构
    - 膨胀卷积：多尺度上下文提取，扩大感受野而不增加参数
    - 深度可分离卷积：局部特征提取，保持计算效率
    - 残差连接：梯度流动，防止退化，保留原始信息
    - 空间注意力：空间维度特征选择，突出重要区域
    - Mamba S6：序列建模，捕获全局长距离依赖
    - 跳跃连接：信息融合，保留原始输入特征
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
        
        # ========== 阶段1：预处理 + 通道注意力 ==========
        # 作用：归一化输入，并通过通道注意力筛选重要的光谱通道
        # 角色：光谱维度特征选择，减少冗余信息
        self.preprocess = nn.LayerNorm(bands)
        self.channel_attention = ChannelAttention(channels=bands, reduction=16)
        
        # ========== 阶段2：多分支特征提取（每个分支内部有残差连接）==========
        # 分支1：条状卷积 - 提取长距离空间依赖
        # 作用：捕获水平/垂直方向的条状结构特征，适合高光谱图像中的线性地物
        # 角色：空间维度长距离依赖提取
        branch1_out_channels = 64
        self.branch1_strip = FeatureExtractionBranch(
            in_channels=bands,
            out_channels=branch1_out_channels,
            module=MultiScaleStripConvolution(
                in_channels=bands,
                out_channels=branch1_out_channels,
                kernel_sizes=[5, 7, 9],
                dropout_rate=dropout_rate,
            ),
            use_residual=True,
        )
        
        # 分支2：膨胀卷积 - 提取多尺度上下文
        # 作用：通过不同膨胀率扩大感受野，捕获多尺度上下文信息
        # 角色：多尺度上下文提取，扩大感受野
        branch2_out_channels = 64
        self.branch2_dilated = FeatureExtractionBranch(
            in_channels=bands,
            out_channels=branch2_out_channels,
            module=DepthwiseSeparableASPP(
                in_channels=bands,
                out_channels=branch2_out_channels,
                dilations=[3, 6, 9, 12],
                dropout_rate=dropout_rate,
            ),
            use_residual=True,
        )
        
        # 分支3：深度可分离卷积 - 提取局部特征
        # 作用：提取局部空间特征，保持计算效率
        # 角色：局部特征提取，保持计算效率
        branch3_out_channels = 64
        self.branch3_local = FeatureExtractionBranch(
            in_channels=bands,
            out_channels=branch3_out_channels,
            module=DepthwiseSeparableSquareConv(
                in_channels=bands,
                out_channels=branch3_out_channels,
                kernel_sizes=[3, 5, 7],
                dropout_rate=dropout_rate,
            ),
            use_residual=True,
        )
        
        # ========== 阶段3：特征融合 + 空间注意力 ==========
        # 作用：融合多分支特征，并通过空间注意力突出重要的空间位置
        # 角色：空间维度特征选择，多分支特征融合
        fusion_input_channels = branch1_out_channels + branch2_out_channels + branch3_out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_input_channels, self.d_model - 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.d_model - 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # ========== 阶段4：Mamba S6序列建模 ==========
        # 作用：建模长距离序列依赖，捕获全局上下文信息
        # 角色：序列建模，全局上下文捕获
        self.mamba = Mamba2(d_model=self.d_model)
        self.mamba_norm = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        
        # ========== 阶段5：跳跃连接 ==========
        # 作用：将原始输入信息与Mamba输出融合，保留原始特征
        # 角色：信息融合，防止信息丢失
        self.skip_connection = nn.Sequential(
            nn.Linear(bands, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
        )
        
        # ========== 阶段6：分类器 ==========
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
        
        # ========== 阶段1：预处理 + 通道注意力 ==========
        # 归一化
        x_norm = self.preprocess(x)  # (H, W, bands)
        
        # 转换为Conv2d格式
        x_conv = x_norm.permute(2, 0, 1).unsqueeze(0)  # (1, bands, H, W)
        
        # 通道注意力：筛选重要的光谱通道
        x_channel_att = self.channel_attention(x_conv)  # (1, bands, H, W)
        
        # ========== 阶段2：多分支特征提取（每个分支内部有残差连接）==========
        # 分支1：条状卷积 - 长距离空间依赖
        x_branch1 = self.branch1_strip(x_channel_att)  # (1, 64, H, W)
        
        # 分支2：膨胀卷积 - 多尺度上下文
        x_branch2 = self.branch2_dilated(x_channel_att)  # (1, 64, H, W)
        
        # 分支3：深度可分离卷积 - 局部特征
        x_branch3 = self.branch3_local(x_channel_att)  # (1, 64, H, W)
        
        # ========== 阶段3：特征融合 + 空间注意力 ==========
        # 拼接多分支特征
        x_concat = torch.cat([x_branch1, x_branch2, x_branch3], dim=1)  # (1, 192, H, W)
        
        # 融合
        x_fusion = self.fusion(x_concat)  # (1, d_model-2, H, W)
        
        # 空间注意力：突出重要的空间位置
        x_spatial_att = self.spatial_attention(x_fusion)  # (1, d_model-2, H, W)
        
        # 转换为序列格式
        x_fusion_seq = x_spatial_att.squeeze(0).permute(1, 2, 0)  # (H, W, d_model-2)
        
        # ========== 添加坐标信息 ==========
        # 创建坐标网格并归一化到[-1, 1]
        y_coords = torch.arange(h, dtype=torch.float32, device=x_fusion_seq.device)
        if h > 1:
            y_coords = (y_coords / (h - 1) * 2 - 1).unsqueeze(1).expand(h, w)  # (H, W)
        else:
            y_coords = torch.zeros(h, w, dtype=torch.float32, device=x_fusion_seq.device)
        
        x_coords = torch.arange(w, dtype=torch.float32, device=x_fusion_seq.device)
        if w > 1:
            x_coords = (x_coords / (w - 1) * 2 - 1).unsqueeze(0).expand(h, w)  # (H, W)
        else:
            x_coords = torch.zeros(h, w, dtype=torch.float32, device=x_fusion_seq.device)
        
        # 拼接坐标信息
        x_coords = x_coords.unsqueeze(-1)  # (H, W, 1)
        y_coords = y_coords.unsqueeze(-1)  # (H, W, 1)
        x_with_coords = torch.cat([x_fusion_seq, x_coords, y_coords], dim=-1)  # (H, W, d_model)
        
        # ========== 阶段4：Mamba S6序列建模 ==========
        # Reshape为序列
        x_seq = x_with_coords.reshape(-1, self.d_model).unsqueeze(0)  # (1, H*W, d_model)
        
        # Mamba处理
        x_mamba = self.mamba(x_seq)  # (1, H*W, d_model)
        x_mamba = self.mamba_norm(x_mamba)  # (1, H*W, d_model)
        x_mamba = x_mamba.squeeze(0)  # (H*W, d_model)
        
        # ========== 阶段5：跳跃连接 ==========
        # 将原始输入投影到d_model维度
        x_original = x_norm.reshape(-1, bands)  # (H*W, bands)
        x_skip = self.skip_connection(x_original)  # (H*W, d_model)
        
        # 融合Mamba输出和原始输入
        x_final = x_mamba + x_skip  # (H*W, d_model)
        
        # ========== 阶段6：分类 ==========
        output = self.classifier(x_final)  # (H*W, num_classes)
        output = output.reshape(h, w, -1)  # (H, W, num_classes)
        
        return output

