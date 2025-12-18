import torch.nn as nn
from .s6_core import S6Core


class BidirectionalMamba(nn.Module):
    """
    双向Mamba模块：Pure Bidirectional SSM (V4.3)

    改进：
    响应用户反馈，**再次移除 Local Conv**。

    原因：
    用户指出：“双分支中空间分支就已经进行过卷积，提取空间信息”。
    这是一个非常精准的架构洞察。
    1. 前端的 Spatial Branch (V3.9 SK-Fusion) 已经使用了强大的多尺度卷积来提取丰富的局部空间特征。
    2. Fusion 模块已经将这些空间特征与光谱特征融合。
    3. 因此，Mamba 模块的任务应该是专注于利用 SSM 的长序列建模能力来捕获 **全局上下文 (Global Context)** 和 **长距离依赖**。
    4. 在此阶段再次添加 Local Conv 属于功能冗余，不仅增加参数，还可能因为过度平滑而模糊了 Spatial Branch 辛辛苦苦提取的精细特征。

    回归纯净的 SSM 结构，让各模块各司其职：
    - Spatial Branch -> Local Spatial Features
    - Spectral Branch -> Local Spectral Features
    - Mamba -> Global Context Integration
    """

    def __init__(
        self,
        d_model=64,
        dropout_rate=0.5,
    ):
        super().__init__()
        self.d_model = d_model

        # 移除 Local Conv
        # self.local_conv = ...

        # Global Context: Bidirectional SSM
        self.mamba_fwd = S6Core(d_model=d_model)
        self.mamba_bwd = S6Core(d_model=d_model)

        # Norm & Dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            x: 输入特征 (batch, seq_len, d_model)
        """
        # Residual connection
        residual = x

        # 1. 归一化
        x = self.norm(x)

        # 2. Local Conv (Removed)
        # x_t = x.transpose(1, 2)
        # x_t = self.local_conv(x_t)
        # x = x_t.transpose(1, 2)

        # 3. Bidirectional SSM
        x_fwd = self.mamba_fwd(x)
        x_bwd = self.mamba_bwd(x.flip(dims=[1])).flip(dims=[1])
        x_ssm = x_fwd + x_bwd

        # 4. Dropout & Residual
        out = self.dropout(x_ssm) + residual

        return out
