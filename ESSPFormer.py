import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# Residual Connection
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# Layer Normalization
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# SPPCF

class SPPCFModule(nn.Module):
    def __init__(self, input_dim: int):
        """
        SPPCF模块：按2、4、8、12不同梯度融合，适配通道数≥48（depth=4×input_dim≥48→input_dim≥12）
        参数：
            input_dim: ASBS筛选后波段数K（需满足K≥12），GPCF输出维度=4×input_dim（depth）
        文档依据：
            - 《PPCF模块优化策略》：自适应尺度池化（2/4/8/12）+ 跨融合
            - 《DPCFFormer完整优化策略V5》：无参数设计、维度协同（输入输出维度一致）
            - 《GPCF PPCF 模块优化策略》：注意力加权思想（按尺度重要性分配特征位）
        """
        super(SPPCFModule, self).__init__()
        # 1. 定义4个不同梯度（池化尺度），符合文档“多尺度覆盖”要求
        self.pool_scales: List[int] = [2, 4, 8, 12]  # 不同梯度，覆盖精细→全局特征
        self.input_dim = input_dim  # K（ASBS后波段数，≥12）
        self.depth = 4 * input_dim  # GPCF输出维度（4×K，≥48）
        # 2. 预验证输入合法性（避免通道数不足）
        assert self.depth >= 48, f"输入通道数需≥48，当前depth={self.depth}（K={input_dim}，需K≥12）"
        for s in self.pool_scales:
            assert self.depth % s == 0 or (self.depth // s) >= 4, f"尺度{s}不适配depth={self.depth}，池化后维度过小"

    def adaptive_pool(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        """
        修复版自适应尺度池化：用“重复+截断”替代反射填充，避免超维度错误
        文档依据：
        - 《PPCF 模块优化策略》：无参数操作，避免填充超维度
        - 《DPCFFormer 完整优化策略V5》：轻量化原则（无额外计算量）
        """
        batch_size, depth = x.shape
        # 1. 1D平均池化（保留全局统计特征，与原逻辑一致）
        pooled = F.avg_pool1d(
            x.unsqueeze(1),  # 扩展通道维度：(batch_size, 1, depth)
            kernel_size=scale,
            stride=scale
        ).squeeze(1)  # 压缩通道维度：(batch_size, pooled_dim)
        pooled_dim = pooled.shape[1]
        input_dim = self.input_dim  # 目标维度=ASBS后的K（如39）

        # 2. 动态适配维度：重复+截断（无参数，避免填充超维度）
        if pooled_dim < input_dim:
            # 计算重复次数：确保重复后长度≥input_dim（向上取整）
            repeat_times = (input_dim + pooled_dim - 1) // pooled_dim  # 例：39+19-1=57→57//19=3
            # 横向重复特征（维度1：波段特征维度，无参数操作）
            pooled_repeated = pooled.repeat(1, repeat_times)  # 例：(256,19)→(256,57)
            # 截断到目标维度（保留前input_dim个特征，符合《LKSE策略》关键特征优先）
            pooled = pooled_repeated[:, :input_dim]  # 例：(256,57)→(256,39)
        elif pooled_dim > input_dim:
            # 截断多余维度（保留前input_dim个关键特征，与原逻辑一致）
            pooled = pooled[:, :input_dim]

        return pooled

    def generate_fusion_slices(self) -> List[slice]:
        """
        生成融合切片：按不同梯度分配特征位，总长度=depth（4×K），符合《GPCF PPCF 模块优化策略》跨融合逻辑
        返回：
            slices: 4个切片，对应4个尺度的特征分配范围
        """
        # 按尺度重要性分配特征位数量（小尺度精细特征分配更多位，《注意力加权思想》）
        total_bits = self.depth
        # 分配比例：尺度2（30%）、尺度4（25%）、尺度8（25%）、尺度12（20%），总和100%
        bit_counts = [
            int(total_bits * 0.3),  # 尺度2：精细特征，最多位
            int(total_bits * 0.25),  # 尺度4：次精细特征
            int(total_bits * 0.25),  # 尺度8：全局特征
            total_bits - int(total_bits * 0.3) - int(total_bits * 0.25) - int(total_bits * 0.25)  # 尺度12：剩余位
        ]

        # 生成连续切片（避免特征碎片化，《PPCF模块优化策略》特征连续性要求）
        slices = []
        start = 0
        for bits in bit_counts:
            slices.append(slice(start, start + bits))
            start += bits
        return slices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：按2、4、8、12梯度池化→跨融合，适配通道数≥48的输入
        参数：
            x: 输入特征（GPCF输出），shape=(batch_size, depth)，depth=4×K≥48
        返回：
            fused_output: 融合后特征，shape=(batch_size, depth)，与输入维度一致（《CMRL策略》维度协同）
        """
        batch_size, input_depth = x.shape
        # 验证输入维度与预定义depth一致（《跨模块残差链接策略》维度协同）
        assert input_depth == self.depth, f"输入维度{input_depth}≠预定义depth{self.depth}（4×K）"

        # 1. 多梯度自适应池化（4个不同尺度，《PPCF模块优化策略》核心）
        pooled_features = []
        for scale in self.pool_scales:
            pooled = self.adaptive_pool(x, scale)
            pooled_features.append(pooled)  # 每个元素shape=(batch_size, input_dim)

        # 2. 生成融合切片（动态适配depth）
        fusion_slices = self.generate_fusion_slices()

        # 3. 跨融合（按梯度分配特征，《GPCF PPCF 模块优化策略》跨融合逻辑）
        fused_output = torch.empty((batch_size, self.depth), device=x.device)
        for i, (pooled, slice_range) in enumerate(zip(pooled_features, fusion_slices)):
            # 按切片分配特征，适配切片长度（动态计算，避免硬编码）
            slice_length = slice_range.stop - slice_range.start
            if pooled.shape[1] > slice_length:
                # 截断到切片长度
                fused_output[:, slice_range] = pooled[:, :slice_length]
            else:
                # 反射填充到切片长度（《PPCF模块优化策略》填充逻辑）
                pad_size = slice_length - pooled.shape[1]
                pooled_padded = F.pad(pooled, (0, pad_size), mode='reflect')
                fused_output[:, slice_range] = pooled_padded

        return fused_output


class SPPCF_encoder(nn.Module):
    def __init__(self, dim, num_blocks=3, mlp_dim=64, dropout=0., use_residual=True):
        """
        Transformer编码器：支持残差连接开关控制
        参数新增：
            use_residual: bool，是否启用残差连接（输出=当前Block计算+前一个Block输入）
                          （默认True，与《Manuscript 9.25.16.52.docx》Section II.C优化一致）
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_blocks = num_blocks
        self.use_residual = use_residual  # 保存残差控制参数
        # 原有Block构建逻辑不变（SPPCF+FeedForward，《Manuscript》Section II.C）
        for _ in range(num_blocks):
            self.layers.append(Residual(PreNorm(dim, SPPCFModule(input_dim=dim // 4))))
            self.layers.append(Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))))

    def forward(self, x):
        for block_idx in range(self.num_blocks):
            sppcf_layer = self.layers[2 * block_idx]
            ffn_layer = self.layers[2 * block_idx + 1]
            block_input = x  # 前一个Block的输出（当前Block输入，《Manuscript》残差设计核心）
            block_output = ffn_layer(sppcf_layer(block_input))  # 当前Block计算结果

            # 新增：按use_residual控制是否执行残差连接
            if self.use_residual:
                x = block_output + block_input  # 启用残差（优化后逻辑）
            else:
                x = block_output  # 禁用残差（原始Transformer逻辑，无加法）

        return x

# Group Pooling in the end
class Group_Pooling(nn.Module):
    def __init__(self, channels, num_classes, groupPoolScale):
        super().__init__()
        self.channels = channels
        self.gPS = groupPoolScale
        self.remainder = channels % groupPoolScale
        self.divisible_channels = channels - self.remainder
        if self.remainder == 0:
            self.fc = nn.Linear(channels // groupPoolScale, num_classes)
        else:
            self.fc = nn.Linear(channels // groupPoolScale + 1, num_classes)

    def groupPooling(self, x):
        if self.remainder == 0:
            x = x.view(x.shape[0], self.channels // self.gPS, 4 * self.gPS).mean(dim=2).cuda()
        else:
            xt = torch.empty((x.shape[0], self.channels // self.gPS + 1))
            xt[:, :-1] = x[:, :self.divisible_channels * 4].view(x.shape[0], self.channels // self.gPS, 4 * self.gPS)\
                .mean(dim=2)
            xt[:, -1] = x[:, self.divisible_channels * 4:].mean(dim=1)
            x = xt.cuda()
        return x

# Main Model
class ESSPFormerModel(nn.Module):
    def __init__(self, num_classes, encoder_n, input_dim, groupPoolScale=4, use_residual=True):
        """
        ESSPFormer主模型：传递残差控制参数到SPPCF_encoder
        参数新增：
            use_residual: bool，与SPPCF_encoder的use_residual对应，控制残差连接启用/禁用
        """
        super(ESSPFormerModel, self).__init__()
        self.encoder_n = encoder_n
        self.GPScale = groupPoolScale
        # 传递use_residual参数到SPPCF_encoder
        self.SPPCF_encoders = nn.ModuleList(
            [SPPCF_encoder(
                dim=4*input_dim,  # 输入维度=4×ASBS筛选后波段数K，《ASBS策略》Section III协同
                num_blocks=encoder_n,  # Block数量=原encoder_n，《Manuscript》Table III推荐3
                mlp_dim=32,
                dropout=0.,
                use_residual=use_residual  # 传递残差控制参数
            ) for _ in range(1)]
        )
        self.group_pooling = Group_Pooling(input_dim, num_classes, groupPoolScale)

    def forward(self, x):
        # 2. 前向传播：遍历编码器（仅1个），内部已实现Block间残差
        for encoder in self.SPPCF_encoders:
            x = encoder(x)  # 经过encoder_n个Block+残差后的特征

        # 3. 后续分组池化和分类（原逻辑不变，对齐论文Section II.D）
        x = self.group_pooling.groupPooling(x)
        x = self.group_pooling.fc(x)
        return x