import torch
import torch.nn as nn


class Classifier(nn.Module):
    """
    分类器模块 - Enhanced Stable Version (V4.23)
    
    改进：
    - 保持单层隐藏层结构，避免过度复杂化
    - 添加可学习权重的残差连接：平衡残差连接的影响，提升整体稳定性
    - 使用更稳定的归一化：提升训练稳定性
    
    结构：
    - Linear(d_model -> classifier_hidden)
    - LayerNorm
    - GELU
    - Dropout
    - Linear(classifier_hidden -> num_classes)
    - 可学习权重的残差连接：output = output + alpha * input_projection
    """
    
    def __init__(
        self,
        d_model=64,
        num_classes=17,
        classifier_hidden=64,
        dropout_rate=0.5,
    ):
        """
        Args:
            d_model: 输入特征维度
            num_classes: 分类类别数
            classifier_hidden: 隐藏层维度
            dropout_rate: Dropout比率
        """
        super().__init__()
        self.d_model = d_model
        self.classifier_hidden = classifier_hidden
        
        # 精简分类器（移除残差连接以降低内存占用）
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_hidden, bias=True),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden, num_classes, bias=True),
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入特征 (..., d_model) 或 (H*W, d_model)
        Returns:
            分类结果 (..., num_classes) 或 (H*W, num_classes)
        """
        # 精简分类器（移除残差连接）
        output = self.classifier(x)
        
        return output

