"""
S6核心模块 - 仅使用mamba-ssm中的selective_scan_fn（S6核心）
简化版本，移除Mamba的其他组件（in_proj, conv1d, out_proj等）
直接使用优化过的selective_scan_fn实现，速度快
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class S6Core(nn.Module):
    """
    S6核心模块 - 仅包含状态空间模型的核心更新机制
    使用mamba-ssm的selective_scan_fn，速度快且优化过
    """

    def __init__(
        self,
        d_model=64,
    ):
        super().__init__()
        self.d_model = d_model

        # S6参数在内部自动计算，不对外暴露
        # 状态维度，通常为d_model的1/4
        self.d_state = max(16, d_model // 4)
        # dt的秩，通常为d_model的1/16
        self.dt_rank = max(4, d_model // 16)
        # head维度
        self.headdim = 64
        self.ngroups = self.d_model // self.headdim

        # dt相关参数的默认值
        dt_min = 0.001
        dt_max = 0.1
        dt_scale = 1.0

        # S6核心参数 - 使用与Mamba相同的初始化方式
        # A: 状态矩阵 (d_model, d_state)，使用对数参数化
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).repeat(self.d_model, 1)  # (d_model, d_state)
        self.A_log = nn.Parameter(torch.log(A))

        # D: 跳跃连接参数 (d_model,)
        self.D = nn.Parameter(torch.ones(self.d_model))

        # dt相关参数 - 使用与Mamba相同的方式
        # dt_proj将dt_rank维度的输入投影到d_model维度
        self.dt_proj = nn.Linear(self.dt_rank, self.d_model, bias=True)

        # dt_bias初始化（使用随机初始化）
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        dt_init_std = dt_init_std * (dt_max - dt_min)
        dt_init_std = dt_init_std / (self.dt_rank**0.5)
        dt_bias_init = torch.rand(self.d_model) * dt_init_std + dt_min

        # 初始化dt_proj的bias
        with torch.no_grad():
            self.dt_proj.bias.data = dt_bias_init

        # 简化的投影层（仅用于生成B, C, dt参数）
        # 使用与Mamba类似的投影方式，但更简化
        # B和C需要投影到(d_state, d_model)维度
        self.proj_B = nn.Linear(d_model, self.d_state * self.d_model, bias=False)
        self.proj_C = nn.Linear(d_model, self.d_state * self.d_model, bias=False)
        self.proj_dt = nn.Linear(d_model, self.dt_rank, bias=False)

        # 输出投影（简化版）
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: 输入特征 (batch, seq_len, d_model)
        Returns:
            输出特征 (batch, seq_len, d_model)
        """
        batch, seq_len, d_inner = x.shape

        # selective_scan_fn期望的格式是 (batch, dim, seq_len)，需要转置
        # 将 (batch, seq_len, d_model) 转换为 (batch, d_model, seq_len)
        x_t = x.transpose(1, 2).contiguous()  # (batch, d_model, seq_len)

        # 生成B, C, dt参数
        B = self.proj_B(x)  # (batch, seq_len, d_state * d_model)
        C = self.proj_C(x)  # (batch, seq_len, d_state * d_model)
        dt = self.proj_dt(x)  # (batch, seq_len, dt_rank)

        # 通过dt_proj投影并添加bias得到delta
        delta = self.dt_proj(dt)  # (batch, seq_len, d_model)
        delta = F.softplus(delta)  # 使用softplus激活
        delta_t = delta.transpose(1, 2).contiguous()  # (batch, d_model, seq_len)

        # 重塑B, C为selective_scan_fn需要的形状
        # 根据Mamba的实现，B和C应该是 (batch, d_state, seq_len)
        # 但我们的投影输出是 (batch, seq_len, d_state * d_model)
        # 我们需要重新组织为 (batch, d_state, seq_len)
        # 为了简化，我们使用每个d_model维度共享相同的B和C
        B = B.view(batch, seq_len, self.d_state, self.d_model)
        C = C.view(batch, seq_len, self.d_state, self.d_model)
        # 取第一个d_model维度，然后转置为 (batch, d_state, seq_len)
        B = B[:, :, :, 0].transpose(1, 2).contiguous()  # (batch, d_state, seq_len)
        C = C[:, :, :, 0].transpose(1, 2).contiguous()  # (batch, d_state, seq_len)

        # 获取A矩阵
        A = -torch.exp(self.A_log.float())  # (d_model, d_state)

        # 使用selective_scan_fn（S6核心）
        # selective_scan_fn期望的输入格式：
        # u: (batch, dim, seq_len) - 输入
        # delta: (batch, dim, seq_len) - 时间步长
        # A: (dim, d_state) - 状态矩阵
        # B: (batch, d_state, seq_len) - 输入矩阵
        # C: (batch, d_state, seq_len) - 输出矩阵
        # D: (dim,) - 跳跃连接

        y = selective_scan_fn(
            u=x_t,  # (batch, d_model, seq_len)
            delta=delta_t,  # (batch, d_model, seq_len)
            A=A,  # (d_model, d_state)
            B=B,  # (batch, d_state, seq_len)
            C=C,  # (batch, d_state, seq_len)
            D=self.D,  # (d_model,)
        )

        # 转置回 (batch, seq_len, d_model)
        y = y.transpose(1, 2).contiguous()  # (batch, seq_len, d_model)

        # 输出投影
        y = self.out_proj(y)

        return y
