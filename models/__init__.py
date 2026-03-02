"""
models 包初始化

该包导出本项目中定义的网络与基础模块，便于在外部通过 `from models import ...` 方式引用：

- `MambaHSINet`：高光谱图像分类主网络
- `S6Core`：序列建模核心（依赖实现）
- `MambaLayer`：双向 S6Core 的封装层
- `SpatialBranchModule`：多尺度空间卷积分支
- `SpectralBranchModule`：逐像素光谱分支

将常用组件通过 `__all__` 公开，便于导入自动完成和文档生成。
"""

from .model import MambaHSINet
from .s6_core import S6Core
from .spatial_branch import SpatialBranchModule
from .spectral_branch import SpectralBranchModule
from .mamba_layer import MambaLayer

__all__ = [
    "MambaHSINet",
    "S6Core",
    "MambaLayer",
    "SpatialBranchModule",
    "SpectralBranchModule",
    "Classifier",
    "FusionModule",
    "PreprocessModule",
    "MambaGlobalModule",
]
