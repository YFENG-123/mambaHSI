"""
分析Net模型的计算复杂度（FLOPs）
"""
import torch
import torch.nn as nn
from models import Net


def count_flops_by_module(model, input_shape=(145, 145, 200)):
    """计算每个模块的FLOPs"""
    model.eval()
    h, w, bands = input_shape
    
    # 创建输入
    x = torch.randn(h, w, bands).to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(x.device)
    
    print("=" * 100)
    print("Net模型计算复杂度（FLOPs）分析")
    print("=" * 100)
    print(f"\n输入尺寸: ({h}, {w}, {bands})")
    print(f"d_model: {model.d_model}")
    print(f"bands: {model.bands}")
    
    # 使用手动计算方法
    print("\n使用手动计算方法分析各模块FLOPs...")
    
    # 手动计算各模块FLOPs
    print("\n" + "=" * 100)
    print("各模块FLOPs详细分析")
    print("=" * 100)
    
    flops_breakdown = {}
    
    # 1. 预处理层
    preprocess_flops = h * w * bands * 2  # LayerNorm: mean + variance
    flops_breakdown['预处理层 (LayerNorm)'] = preprocess_flops
    print(f"\n1. 预处理层 (LayerNorm): {preprocess_flops / 1e6:.4f} MFLOPs")
    
    # 2. 注意力模块
    # 通道注意力
    # - 平均池化: H * W * bands
    # - 最大池化: H * W * bands
    # - MLP: 2 * (bands * bands/reduction + bands/reduction * bands) = 2 * bands^2 / reduction
    # - Sigmoid: bands
    channel_att_flops = h * w * bands * 2 + 2 * bands * bands // 16 + bands
    # 空间注意力
    # - 通道维度池化: H * W * bands * 2
    # - 卷积: H * W * 2 * 7 * 7 (kernel_size=7)
    # - Sigmoid: H * W
    spatial_att_flops = h * w * bands * 2 + h * w * 2 * 7 * 7 + h * w
    attention_flops = channel_att_flops + spatial_att_flops
    flops_breakdown['注意力模块'] = attention_flops
    print(f"\n2. 注意力模块: {attention_flops / 1e6:.4f} MFLOPs")
    print(f"   └─ 通道注意力: {channel_att_flops / 1e6:.4f} MFLOPs")
    print(f"   └─ 空间注意力: {spatial_att_flops / 1e6:.4f} MFLOPs")
    
    # 3. 分支1: ASPP
    # 每个膨胀卷积: H * W * bands * kernel_size^2 (深度可分离)
    # 深度卷积: H * W * bands * 3 * 3
    branch1_flops = 0
    for dilation in [9, 11, 13]:
        # 深度可分离卷积: 每个通道独立卷积
        conv_flops = h * w * bands * 3 * 3  # 3x3卷积
        branch1_flops += conv_flops
    flops_breakdown['分支1 (ASPP)'] = branch1_flops
    print(f"\n3. 分支1 (DepthwiseSeparableASPP): {branch1_flops / 1e6:.4f} MFLOPs")
    print(f"   └─ 3个膨胀卷积: {branch1_flops / 1e6:.4f} MFLOPs")
    
    # 4. 分支2: 非对称卷积
    # 每个非对称卷积对: 1xk和kx1
    branch2_flops = 0
    for kernel_size in [15, 17, 19]:
        # 1xk卷积: H * W * bands * kernel_size
        conv1k_flops = h * w * bands * kernel_size
        # kx1卷积: H * W * bands * kernel_size
        convk1_flops = h * w * bands * kernel_size
        branch2_flops += conv1k_flops + convk1_flops
    flops_breakdown['分支2 (非对称卷积)'] = branch2_flops
    print(f"\n4. 分支2 (MultiScaleAsymmetricDepthConv): {branch2_flops / 1e6:.4f} MFLOPs")
    print(f"   └─ 3个非对称卷积对: {branch2_flops / 1e6:.4f} MFLOPs")
    
    # 5. 分支3: 方形卷积
    branch3_flops = 0
    for kernel_size in [3, 5, 7]:
        # 深度可分离卷积: H * W * bands * kernel_size^2
        conv_flops = h * w * bands * kernel_size * kernel_size
        branch3_flops += conv_flops
    flops_breakdown['分支3 (方形卷积)'] = branch3_flops
    print(f"\n5. 分支3 (DepthwiseSeparableSquareConv): {branch3_flops / 1e6:.4f} MFLOPs")
    print(f"   └─ 3个方形卷积: {branch3_flops / 1e6:.4f} MFLOPs")
    
    # 6. 分支压缩层
    # 计算各分支的输出通道数
    branch1_output_channels = bands * len(model.branch1_aspp.dilated_convs)  # 600
    branch2_output_channels = bands * len(model.branch2_asymmetric.kernel_sizes) * 2  # 1200
    branch3_output_channels = bands * len(model.branch3_square.kernel_sizes)  # 600
    branch_compressed_channels = 64  # 压缩后的通道数
    
    # 分支1压缩层FLOPs
    branch1_compress_conv_flops = h * w * branch1_output_channels * branch_compressed_channels
    branch1_compress_bn_flops = h * w * branch_compressed_channels * 2
    branch1_compress_flops = branch1_compress_conv_flops + branch1_compress_bn_flops
    
    # 分支2压缩层FLOPs
    branch2_compress_conv_flops = h * w * branch2_output_channels * branch_compressed_channels
    branch2_compress_bn_flops = h * w * branch_compressed_channels * 2
    branch2_compress_flops = branch2_compress_conv_flops + branch2_compress_bn_flops
    
    # 分支3压缩层FLOPs
    branch3_compress_conv_flops = h * w * branch3_output_channels * branch_compressed_channels
    branch3_compress_bn_flops = h * w * branch_compressed_channels * 2
    branch3_compress_flops = branch3_compress_conv_flops + branch3_compress_bn_flops
    
    total_compress_flops = branch1_compress_flops + branch2_compress_flops + branch3_compress_flops
    flops_breakdown['分支压缩层'] = total_compress_flops
    print(f"\n6. 分支压缩层: {total_compress_flops / 1e6:.4f} MFLOPs")
    print(f"   └─ 分支1压缩({branch1_output_channels}→64): {branch1_compress_flops / 1e6:.4f} MFLOPs")
    print(f"   └─ 分支2压缩({branch2_output_channels}→64): {branch2_compress_flops / 1e6:.4f} MFLOPs")
    print(f"   └─ 分支3压缩({branch3_output_channels}→64): {branch3_compress_flops / 1e6:.4f} MFLOPs")
    
    # 7. 融合层（压缩后）
    # Conv2d: H * W * fusion_input_channels * d_model
    fusion_input_channels = branch_compressed_channels * 3  # 64 * 3 = 192
    fusion_conv_flops = h * w * fusion_input_channels * model.d_model
    # BatchNorm: H * W * d_model * 2 (mean + variance)
    fusion_bn_flops = h * w * model.d_model * 2
    fusion_flops = fusion_conv_flops + fusion_bn_flops
    flops_breakdown['融合层'] = fusion_flops
    print(f"\n7. 融合层: {fusion_flops / 1e6:.4f} MFLOPs")
    print(f"   └─ Conv2d({fusion_input_channels}→256): {fusion_conv_flops / 1e6:.4f} MFLOPs")
    print(f"   └─ BatchNorm2d: {fusion_bn_flops / 1e6:.4f} MFLOPs")
    
    # 8. Mamba层
    # Mamba的FLOPs计算较复杂
    # Mamba2的主要计算包括：
    # 1. 线性投影: seq_len * d_model * (d_model * 4)  # 输入投影到4倍维度
    # 2. 状态空间模型: seq_len * d_model * d_model  # SSM核心计算
    # 3. 输出投影: seq_len * d_model * d_model
    seq_len = h * w
    # Mamba2默认使用d_state=16, d_conv=4等参数
    # 简化计算：主要开销在投影和SSM计算
    # 输入投影: seq_len * d_model * (d_model * expand) ≈ seq_len * d_model^2 * 2
    # SSM计算: seq_len * d_model * d_state (d_state通常为16)
    # 输出投影: seq_len * d_model * d_model
    d_state = 16  # Mamba2默认状态维度
    expand = 2  # 扩展因子
    mamba_projection_flops = seq_len * model.d_model * model.d_model * expand * 2  # 输入和输出投影
    mamba_ssm_flops = seq_len * model.d_model * d_state  # SSM计算
    mamba_forward_flops = mamba_projection_flops + mamba_ssm_flops
    mamba_backward_flops = mamba_projection_flops + mamba_ssm_flops
    mamba_norm_flops = seq_len * model.d_model * 2  # LayerNorm
    mamba_total_flops = mamba_forward_flops + mamba_backward_flops + mamba_norm_flops
    flops_breakdown['Mamba层'] = mamba_total_flops
    print(f"\n8. Mamba层: {mamba_total_flops / 1e6:.4f} MFLOPs")
    print(f"   └─ Mamba正向: {mamba_forward_flops / 1e6:.4f} MFLOPs")
    print(f"   └─ Mamba反向: {mamba_backward_flops / 1e6:.4f} MFLOPs")
    print(f"   └─ 融合归一化: {mamba_norm_flops / 1e6:.4f} MFLOPs")
    
    # 9. 分类器
    classifier_flops = 0
    # Linear(256→128): seq_len * 256 * 128
    classifier_flops += seq_len * 256 * 128
    # LayerNorm(128): seq_len * 128 * 2
    classifier_flops += seq_len * 128 * 2
    # Linear(128→64): seq_len * 128 * 64
    classifier_flops += seq_len * 128 * 64
    # LayerNorm(64): seq_len * 64 * 2
    classifier_flops += seq_len * 64 * 2
    # Linear(64→num_classes): seq_len * 64 * 17
    classifier_flops += seq_len * 64 * 17
    flops_breakdown['分类器'] = classifier_flops
    print(f"\n9. 分类器: {classifier_flops / 1e6:.4f} MFLOPs")
    
    # 计算总FLOPs
    calculated_total = sum(flops_breakdown.values())
    print(f"\n计算得到的总FLOPs: {calculated_total / 1e9:.4f} GFLOPs")
    
    # 各模块占比
    print("\n" + "=" * 100)
    print("各模块FLOPs占比")
    print("=" * 100)
    
    for name, flops in sorted(flops_breakdown.items(), key=lambda x: x[1], reverse=True):
        percentage = (flops / calculated_total) * 100
        print(f"{name:50s}: {flops / 1e6:>12.4f} MFLOPs ({percentage:>5.2f}%)")
    
    # 精简建议
    print("\n" + "=" * 100)
    print("计算复杂度精简建议")
    print("=" * 100)
    
    suggestions = []
    
    # 检查分支压缩层
    compress_ratio = (total_compress_flops / calculated_total) * 100
    if compress_ratio > 5:
        suggestions.append(
            f"✓ 分支压缩层FLOPs占比 ({compress_ratio:.2f}%)，这是必要的开销以换取融合层的减少"
        )
    
    # 检查融合层
    fusion_ratio = (fusion_flops / calculated_total) * 100
    if fusion_ratio > 30:
        suggestions.append(
            f"⚠️  融合层FLOPs占比过高 ({fusion_ratio:.2f}%)，建议：\n"
            f"   - 减少融合层输入通道数（当前{fusion_input_channels}）\n"
            f"   - 使用分组卷积或深度可分离卷积替代1x1卷积"
        )
    else:
        suggestions.append(
            f"✓ 融合层FLOPs占比已优化 ({fusion_ratio:.2f}%)，通过分支压缩层显著减少"
        )
    
    # 检查Mamba层
    mamba_ratio = (mamba_total_flops / calculated_total) * 100
    if mamba_ratio > 30:
        suggestions.append(
            f"⚠️  Mamba层FLOPs占比过高 ({mamba_ratio:.2f}%)，建议：\n"
            f"   - 减小d_model（当前{model.d_model}）\n"
            f"   - 使用单方向Mamba（减少50%计算量）\n"
            f"   - 或对序列进行下采样后再处理"
        )
    
    # 检查分支2
    branch2_ratio = (branch2_flops / calculated_total) * 100
    if branch2_ratio > 10:
        suggestions.append(
            f"⚠️  分支2（非对称卷积）FLOPs占比较高 ({branch2_ratio:.2f}%)，建议：\n"
            f"   - 减少非对称卷积核数量（当前3个）\n"
            f"   - 减小卷积核大小（当前[15, 17, 19]）"
        )
    
    # 检查分类器
    classifier_ratio = (classifier_flops / calculated_total) * 100
    if classifier_ratio > 5:
        suggestions.append(
            f"⚠️  分类器FLOPs占比较高 ({classifier_ratio:.2f}%)，建议：\n"
            f"   - 减少中间层维度\n"
            f"   - 直接使用单层分类器"
        )
    
    # 检查分支3
    branch3_ratio = (branch3_flops / calculated_total) * 100
    if branch3_ratio > 5:
        suggestions.append(
            f"✓ 分支3（方形卷积）FLOPs占比 ({branch3_ratio:.2f}%)，可以考虑减少卷积核数量"
        )
    
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion}")
    else:
        print("\n模型计算复杂度较为合理，暂无明显的精简建议。")
    
    return {
        'total_flops': calculated_total,
        'flops_breakdown': flops_breakdown,
        'input_shape': (h, w, bands)
    }




if __name__ == "__main__":
    # 创建模型
    model = Net(
        image_x=145,
        image_y=145,
        num_classes=17,
        bands=200,
        dropout_rate=0.5,
        d_model=256
    )
    
    # 分析计算复杂度
    results = count_flops_by_module(model, input_shape=(145, 145, 200))
    
    print("\n" + "=" * 100)
    print("分析完成！")
    print("=" * 100)

