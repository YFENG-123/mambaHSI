import torch.nn as nn
from .s6_core import S6Core


class BidirectionalMamba(nn.Module):
    """
    双向Mamba模块：纯净版 (Pure SSM) - V4.4
    
    改进：
    在 V4.3 (Pure SSM) 的基础上，进一步精简。
    移除 `Residual` 连接。
    
    原因：
    - Mamba 内部是 `x + residual`，而 MambaHSINet 主干中调用 Mamba 时也是 `x = mamba(x) + x`。
    - 这导致了双重残差连接，不仅增加了梯度路径的复杂性，还可能导致信号强度在深层被过度放大，引起训练不稳定或收敛震荡。
    - 鉴于我们追求“精简模型”，去除模块内部冗余的残差连接，让主干网络的残差机制全权负责梯度流，有助于更平滑的训练。
    """
    
    def __init__(
        self,
        d_model=64,
        dropout_rate=0.5,
    ):
        super().__init__()
        self.d_model = d_model
        
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
        # 1. 归一化
        x_norm = self.norm(x)
        
        # 2. Bidirectional SSM
        x_fwd = self.mamba_fwd(x_norm)
        x_bwd = self.mamba_bwd(x_norm.flip(dims=[1])).flip(dims=[1])
        x_ssm = x_fwd + x_bwd
        
        # 3. Dropout
        out = self.dropout(x_ssm)
        
        # 移除内部 Residual，依赖外部主干网络的 Residual
        # out = out + x 
        
        return out
