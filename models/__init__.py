"""
模型模块
包含所有网络模型和基础模块
"""
try:
    from .model1 import Net1
except ImportError:
    Net1 = None

from .model import Net
from .mamba_hsi_net import MambaHSINet
from .attention import ChannelAttention, SpatialAttention, MultiAttention
from .multi_scale_asymmetric_conv import MultiScaleAsymmetricDepthConv
from .coordinate_conv import CoordinateConv, CoordinateConv2d
from .depthwise_separable_aspp import DepthwiseSeparableASPP
from .depthwise_separable_square_conv import DepthwiseSeparableSquareConv
from .strip_conv import StripConvolution, MultiScaleStripConvolution

__all__ = [
    'Net1',
    'Net',
    'MambaHSINet',
    'ChannelAttention',
    'SpatialAttention',
    'MultiAttention',
    'MultiScaleAsymmetricDepthConv',
    'CoordinateConv',
    'CoordinateConv2d',
    'DepthwiseSeparableASPP',
    'DepthwiseSeparableSquareConv',
    'StripConvolution',
    'MultiScaleStripConvolution',
]

