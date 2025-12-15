import torch.nn as nn

class SpectralBranchModule(nn.Module):
    """
    光谱分支模块：增强的线性映射
    
    结构：Linear -> GELU -> Dropout
    目的：引入非线性和正则化，防止过拟合，平衡与空间分支的复杂度。
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        # 输入 (..., bands) -> 输出 (..., d_model)
        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: (H, W, bands) or (B, H, W, bands)
        x = self.linear(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

