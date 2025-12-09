"""
分析Net模型的参数量和内存占用
"""
import torch
import torch.nn as nn
from models import Net

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_by_module(model, prefix=""):
    """递归计算每个模块的参数量"""
    results = {}
    total = 0
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        module_params = count_parameters(module)
        if module_params > 0:
            results[full_name] = module_params
            total += module_params
        
        # 递归处理子模块
        sub_results = count_parameters_by_module(module, full_name)
        results.update(sub_results)
        total += sum(sub_results.values())
    
    return results

def estimate_memory_usage(model, input_shape=(145, 145, 200)):
    """估算模型的内存占用（包括参数和激活值）"""
    model.eval()
    h, w, bands = input_shape
    
    # 创建输入（模型期望输入形状为 (H, W, bands)，不是 (B, H, W, bands)）
    x = torch.randn(h, w, bands).to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(x.device)
    
    # 参数内存（MB）
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    
    # 激活值内存（MB）- 通过hook记录
    activation_memory = {}
    
    def get_activation_size(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                size_mb = output.numel() * output.element_size() / (1024 ** 2)
                activation_memory[name] = size_mb
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            hooks.append(module.register_forward_hook(get_activation_size(name)))
    
    with torch.no_grad():
        _ = model(x)
    
    for hook in hooks:
        hook.remove()
    
    total_activation_memory = sum(activation_memory.values())
    
    return {
        'param_memory_mb': param_memory,
        'activation_memory_mb': total_activation_memory,
        'total_memory_mb': param_memory + total_activation_memory,
        'activation_details': activation_memory
    }

def analyze_model_components(model, input_shape=(145, 145, 200)):
    """详细分析模型各个组件的参数量和内存占用"""
    print("=" * 100)
    print("Net模型参数量和内存占用分析")
    print("=" * 100)
    
    # 假设输入尺寸
    h, w, bands = input_shape
    print(f"\n输入尺寸: ({h}, {w}, {bands})")
    print(f"d_model: {model.d_model}")
    print(f"bands: {model.bands}")
    
    # 1. 参数量分析
    print("\n" + "=" * 100)
    print("1. 参数量分析（按模块）")
    print("=" * 100)
    
    param_breakdown = {}
    
    # 预处理层
    preprocess_params = count_parameters(model.preprocess)
    param_breakdown['预处理层 (LayerNorm)'] = preprocess_params
    print(f"预处理层 (LayerNorm): {preprocess_params:,} 参数")
    
    # 注意力模块
    attention_params = count_parameters(model.multi_attention)
    param_breakdown['注意力模块 (MultiAttention)'] = attention_params
    
    # 通道注意力
    channel_att_params = count_parameters(model.multi_attention.channel_attention)
    param_breakdown['  └─ 通道注意力'] = channel_att_params
    print(f"  通道注意力: {channel_att_params:,} 参数")
    
    # 空间注意力
    spatial_att_params = count_parameters(model.multi_attention.spatial_attention)
    param_breakdown['  └─ 空间注意力'] = spatial_att_params
    print(f"  空间注意力: {spatial_att_params:,} 参数")
    
    print(f"注意力模块总计: {attention_params:,} 参数")
    
    # 分支1: ASPP
    branch1_params = count_parameters(model.branch1_aspp)
    param_breakdown['分支1 (DepthwiseSeparableASPP)'] = branch1_params
    print(f"\n分支1 (DepthwiseSeparableASPP): {branch1_params:,} 参数")
    dilations = model.branch1_aspp.dilated_convs
    for i, conv in enumerate(dilations):
        conv_params = count_parameters(conv)
        print(f"  └─ 膨胀卷积 {i+1} (dilation={model.branch1_aspp.dilated_convs[i].dilation[0]}): {conv_params:,} 参数")
    
    # 分支2: 非对称卷积
    branch2_params = count_parameters(model.branch2_asymmetric)
    param_breakdown['分支2 (MultiScaleAsymmetricDepthConv)'] = branch2_params
    print(f"\n分支2 (MultiScaleAsymmetricDepthConv): {branch2_params:,} 参数")
    kernel_sizes = model.branch2_asymmetric.kernel_sizes
    for i, (dconv1_k, dconvk_1) in enumerate(zip(
        model.branch2_asymmetric.dconv1_k_list,
        model.branch2_asymmetric.dconvk_1_list
    )):
        conv1_k_params = count_parameters(dconv1_k)
        convk_1_params = count_parameters(dconvk_1)
        print(f"  └─ 非对称卷积对 {i+1} (kernel={kernel_sizes[i]}): {conv1_k_params + convk_1_params:,} 参数")
        print(f"      ├─ 1x{kernel_sizes[i]}: {conv1_k_params:,} 参数")
        print(f"      └─ {kernel_sizes[i]}x1: {convk_1_params:,} 参数")
    
    # 分支3: 方形卷积
    branch3_params = count_parameters(model.branch3_square)
    param_breakdown['分支3 (DepthwiseSeparableSquareConv)'] = branch3_params
    print(f"\n分支3 (DepthwiseSeparableSquareConv): {branch3_params:,} 参数")
    kernel_sizes3 = model.branch3_square.kernel_sizes
    for i, conv in enumerate(model.branch3_square.square_convs):
        conv_params = count_parameters(conv)
        print(f"  └─ 方形卷积 {i+1} (kernel={kernel_sizes3[i]}): {conv_params:,} 参数")
    
    # 融合层
    fusion_params = count_parameters(model.fusion)
    param_breakdown['融合层'] = fusion_params
    print(f"\n融合层: {fusion_params:,} 参数")
    
    # 计算融合层输入通道数
    branch1_out = bands * len(dilations)
    branch2_out = bands * len(kernel_sizes) * 2
    branch3_out = bands * len(kernel_sizes3)
    fusion_input = branch1_out + branch2_out + branch3_out
    print(f"  融合层输入通道数: {fusion_input} = {branch1_out} + {branch2_out} + {branch3_out}")
    print(f"  融合层输出通道数: {model.d_model}")
    
    # 融合层详细参数
    for i, layer in enumerate(model.fusion):
        if isinstance(layer, nn.Conv2d):
            layer_params = count_parameters(layer)
            print(f"  └─ Conv2d {i+1}: {layer_params:,} 参数")
        elif isinstance(layer, nn.BatchNorm2d):
            layer_params = count_parameters(layer)
            print(f"  └─ BatchNorm2d {i+1}: {layer_params:,} 参数")
    
    # Mamba层
    mamba_forward_params = count_parameters(model.mamba_forward)
    mamba_backward_params = count_parameters(model.mamba_backward)
    mamba_norm_params = count_parameters(model.mamba_norm_fusion)
    mamba_total = mamba_forward_params + mamba_backward_params + mamba_norm_params
    param_breakdown['Mamba层'] = mamba_total
    print(f"\nMamba层: {mamba_total:,} 参数")
    print(f"  └─ Mamba正向: {mamba_forward_params:,} 参数")
    print(f"  └─ Mamba反向: {mamba_backward_params:,} 参数")
    print(f"  └─ Mamba融合归一化: {mamba_norm_params:,} 参数")
    
    # 分类器
    classifier_params = count_parameters(model.classifier)
    param_breakdown['分类器'] = classifier_params
    print(f"\n分类器: {classifier_params:,} 参数")
    for i, layer in enumerate(model.classifier):
        if isinstance(layer, nn.Linear):
            layer_params = count_parameters(layer)
            print(f"  └─ Linear {i+1}: {layer_params:,} 参数")
        elif isinstance(layer, nn.LayerNorm):
            layer_params = count_parameters(layer)
            print(f"  └─ LayerNorm {i+1}: {layer_params:,} 参数")
    
    # 总参数量
    total_params = count_parameters(model)
    print(f"\n总参数量: {total_params:,} 参数 ({total_params / 1e6:.2f}M)")
    
    # 2. 内存占用分析
    print("\n" + "=" * 100)
    print("2. 内存占用分析")
    print("=" * 100)
    
    memory_info = estimate_memory_usage(model, input_shape)
    print(f"参数内存: {memory_info['param_memory_mb']:.2f} MB")
    print(f"激活值内存: {memory_info['activation_memory_mb']:.2f} MB")
    print(f"总内存: {memory_info['total_memory_mb']:.2f} MB")
    
    # 3. 各模块参数量占比
    print("\n" + "=" * 100)
    print("3. 各模块参数量占比")
    print("=" * 100)
    
    for name, params in sorted(param_breakdown.items(), key=lambda x: x[1], reverse=True):
        percentage = (params / total_params) * 100
        print(f"{name:50s}: {params:>12,} ({percentage:>5.2f}%)")
    
    # 4. 精简建议
    print("\n" + "=" * 100)
    print("4. 精简建议")
    print("=" * 100)
    
    suggestions = []
    
    # 检查融合层输入通道数
    if fusion_input > 1000:
        suggestions.append(
            f"⚠️  融合层输入通道数过大 ({fusion_input})，建议：\n"
            f"   - 减少ASPP膨胀率数量（当前{len(dilations)}个）\n"
            f"   - 减少非对称卷积核数量（当前{len(kernel_sizes)}个）\n"
            f"   - 减少方形卷积核数量（当前{len(kernel_sizes3)}个）\n"
            f"   - 或在各分支内部先进行通道压缩"
        )
    
    # 检查Mamba参数量
    mamba_ratio = (mamba_total / total_params) * 100
    if mamba_ratio > 50:
        suggestions.append(
            f"⚠️  Mamba层参数量占比过高 ({mamba_ratio:.2f}%)，建议：\n"
            f"   - 减小d_model（当前{model.d_model}）\n"
            f"   - 或使用更轻量的序列模型"
        )
    
    # 检查分类器
    classifier_ratio = (classifier_params / total_params) * 100
    if classifier_ratio > 10:
        suggestions.append(
            f"⚠️  分类器参数量占比较高 ({classifier_ratio:.2f}%)，建议：\n"
            f"   - 减少中间层维度（当前128->64->num_classes）\n"
            f"   - 或直接使用单层分类器"
        )
    
    # 检查注意力模块
    attention_ratio = (attention_params / total_params) * 100
    if attention_ratio < 1:
        suggestions.append(
            f"✓ 注意力模块参数量占比很小 ({attention_ratio:.2f}%)，可以保持"
        )
    
    # 检查分支参数量
    branch_total = branch1_params + branch2_params + branch3_params
    branch_ratio = (branch_total / total_params) * 100
    if branch_ratio < 5:
        suggestions.append(
            f"✓ 三个分支参数量占比很小 ({branch_ratio:.2f}%)，主要开销在融合层和Mamba层"
        )
    
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion}")
    else:
        print("\n模型结构较为合理，暂无明显的精简建议。")
    
    return {
        'total_params': total_params,
        'param_breakdown': param_breakdown,
        'memory_info': memory_info
    }

if __name__ == "__main__":
    # 创建模型（使用Indian_pines的典型参数）
    model = Net(
        image_x=145,
        image_y=145,
        num_classes=17,
        bands=200,
        dropout_rate=0.5,
        d_model=256
    )
    
    # 分析模型
    results = analyze_model_components(model, input_shape=(145, 145, 200))
    
    print("\n" + "=" * 100)
    print("分析完成！")
    print("=" * 100)

