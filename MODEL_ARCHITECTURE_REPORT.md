# 模型架构检查报告

## 模型总览

- **模型名称**: MambaHSINet
- **总参数量**: 153,262
- **基础维度**: d_model = 64
- **输入**: 高光谱图像 (H, W, bands=200)
- **输出**: 分类结果 (H, W, num_classes=17)

## 架构流程

```
输入 (H, W, 200)
  ↓
预处理层 (LayerNorm)
  ↓
多头注意力 (MultiAttention: reduction=64, kernel=3)
  ↓
动态注意力卷积 (单尺度 kernel=3, 输出30通道)
  ↓
特征投影 (30 → 62通道)
  ↓
添加坐标信息 (62 → 64通道)
  ↓
S6核心模块 (selective_scan_fn)
  ↓
归一化 + Dropout
  ↓
残差连接 (可学习权重)
  ↓
特征平滑 (LayerNorm + Dropout)
  ↓
分类器 (64 → 16 → 17)
  ↓
输出 (H, W, 17)
```

## 组件详细分析

### 1. 预处理层 (preprocess)
- **类型**: LayerNorm
- **参数量**: 400 (200×2)
- **功能**: 对输入光谱维度进行归一化

### 2. 多头注意力 (multi_attention)
- **类型**: MultiAttention
- **参数量**: 1,626
- **配置**:
  - reduction: 64 (通道注意力降维比例)
  - spatial_kernel_size: 3 (空间注意力卷积核)
- **功能**: 通道注意力 + 空间注意力

### 3. 动态注意力卷积 (dynamic_attention_conv)
- **类型**: DynamicAttentionConv
- **参数量**: 10,740
- **配置**:
  - 输入通道: 200
  - 输出通道: 30 (reduced_out_channels = max(64-2-32, 16) = 30)
  - 卷积核: [3] (单尺度)
- **功能**: 多尺度特征提取 + 通道/空间注意力

### 4. 特征投影 (feature_projection)
- **类型**: Sequential (Linear + LayerNorm + Dropout)
- **参数量**: 2,046
- **配置**:
  - 输入: 30通道
  - 输出: 62通道 (d_model - 2)
- **功能**: 将特征投影到目标维度

### 5. S6核心模块 (mamba)
- **类型**: S6Core
- **参数量**: 136,832 (占总参数的89.3%)
- **配置**:
  - d_model: 64
  - d_state: 16
  - dt_rank: 4
  - headdim: 64
  - ngroups: 1
- **核心组件**:
  - A_log: (64, 16) - 状态矩阵参数
  - D: (64,) - 跳跃连接参数
  - dt_proj: (64, 4) - 时间步长投影
  - proj_B: (1024, 64) - B矩阵投影 (d_state×d_model = 16×64 = 1024)
  - proj_C: (1024, 64) - C矩阵投影
  - proj_dt: (4, 64) - dt投影
  - out_proj: (64, 64) - 输出投影
- **功能**: 使用mamba-ssm的selective_scan_fn进行状态空间模型更新

### 6. Mamba后归一化 (mamba_norm)
- **类型**: Sequential (LayerNorm + GELU + Dropout)
- **参数量**: 128
- **功能**: 增强正则化

### 7. 残差连接
- **类型**: 可学习标量权重
- **参数量**: 1
- **公式**: output = x_mamba * weight + x_residual * (1 - weight)

### 8. 特征平滑 (feature_smooth)
- **类型**: Sequential (LayerNorm + Dropout)
- **参数量**: 128
- **功能**: 分类器前的特征平滑

### 9. 分类器 (classifier)
- **类型**: Sequential
- **参数量**: 1,361
- **结构**: 
  - Linear(64 → 16)
  - LayerNorm(16)
  - GELU
  - Dropout
  - Linear(16 → 17)

## 参数量分布

| 组件 | 参数量 | 占比 |
|------|--------|------|
| S6核心模块 | 136,832 | 89.3% |
| 动态注意力卷积 | 10,740 | 7.0% |
| 特征投影 | 2,046 | 1.3% |
| 多头注意力 | 1,626 | 1.1% |
| 分类器 | 1,361 | 0.9% |
| 其他 | 657 | 0.4% |

## 优化特点

### 已实施的优化
1. ✅ **简化注意力**: reduction=64, kernel=3
2. ✅ **单尺度卷积**: 从多尺度减少到单尺度(kernel=3)
3. ✅ **减少输出通道**: 动态卷积输出30通道
4. ✅ **移除压缩-扩展层**: 直接使用d_model=64
5. ✅ **简化残差连接**: 使用标量权重而非门控网络
6. ✅ **简化分类器**: 64→16→17
7. ✅ **使用S6核心**: 直接使用mamba-ssm的selective_scan_fn

### 参数量对比
- **原始模型**: ~200K
- **当前模型**: 153,262
- **减少比例**: 23.4%

## 潜在优化方向

### 如果仍需减少参数量
1. **减少S6核心参数量** (当前占89.3%):
   - 减少d_state从16到8或12
   - 减少proj_B和proj_C的输出维度
   
2. **进一步简化动态注意力卷积**:
   - 减少输出通道从30到16或24
   - 简化内部注意力机制

3. **简化特征投影**:
   - 移除LayerNorm
   - 减少中间维度

### 如果速度仍慢
- S6核心已使用mamba-ssm的优化实现(selective_scan_fn)
- 确保在CUDA设备上运行以获得最佳性能

## 注意事项

1. **S6核心模块**: 使用mamba-ssm的selective_scan_fn，需要CUDA支持
2. **参数量**: S6核心模块占89.3%的参数量，是主要的参数来源
3. **维度要求**: d_model=64是Mamba2的最小要求，不能再减少

