"""
测试MambaHSINet模型
"""
import torch
from models import MambaHSINet

def test_mamba_hsi_net():
    """测试MambaHSINet模型的前向传播"""
    # 设置参数
    image_x, image_y = 145, 145
    num_classes = 17
    bands = 200
    d_model = 128
    
    # 创建模型
    model = MambaHSINet(
        image_x=image_x,
        image_y=image_y,
        num_classes=num_classes,
        bands=bands,
        dropout_rate=0.5,
        d_model=d_model,
    )
    
    # 创建随机输入 (H, W, bands)
    x = torch.randn(image_x, image_y, bands)
    
    # 前向传播
    print("输入形状:", x.shape)
    output = model(x)
    print("输出形状:", output.shape)
    print(f"预期输出形状: ({image_x}, {image_y}, {num_classes})")
    
    # 验证输出形状
    assert output.shape == (image_x, image_y, num_classes), \
        f"输出形状不匹配！期望 ({image_x}, {image_y}, {num_classes})，得到 {output.shape}"
    
    print("\n✓ 模型测试通过！")
    print(f"✓ 输入: ({image_x}, {image_y}, {bands})")
    print(f"✓ 输出: ({image_x}, {image_y}, {num_classes})")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

if __name__ == "__main__":
    test_mamba_hsi_net()

