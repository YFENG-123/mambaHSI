import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassAwareFeatureEnhancement(nn.Module):
    """
    类别感知的特征增强模块
    
    使用类别原型（class prototypes）来增强少数类别的特征表示：
    1. 维护可学习的类别原型向量
    2. 计算特征与类别原型的相似度
    3. 使用注意力机制融合原型信息来增强特征
    4. 特别关注少数类别的特征增强
    
    结构：
    - 类别原型嵌入 (num_classes, d_model)
    - 相似度计算（余弦相似度）
    - 注意力加权融合
    - 残差连接
    """
    
    def __init__(
        self,
        d_model=64,
        num_classes=17,
        dropout_rate=0.5,
        temperature=0.1,  # 温度参数，控制注意力分布的锐度
        enhancement_scale=0.5,  # 增强强度，控制原型增强的权重
    ):
        """
        Args:
            d_model: 特征维度
            num_classes: 类别数量
            dropout_rate: Dropout比率
            temperature: 温度参数，用于缩放相似度
            enhancement_scale: 增强强度，控制原型增强的权重
        """
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.temperature = temperature
        self.enhancement_scale = enhancement_scale
        
        # 类别原型：可学习的嵌入向量
        # 每个类别有一个原型向量，用于表示该类别的典型特征
        self.class_prototypes = nn.Parameter(
            torch.randn(num_classes, d_model)
        )
        # 初始化原型：使用Xavier初始化
        nn.init.xavier_uniform_(self.class_prototypes)
        
        # 特征投影层：将输入特征投影到原型空间
        self.feature_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 原型投影层：将原型投影到特征空间
        self.prototype_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 融合层：融合原始特征和增强特征
        self.fusion_norm = nn.LayerNorm(d_model)
        self.fusion_dropout = nn.Dropout(dropout_rate)
        
        # 门控机制：控制增强强度
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x, class_labels=None):
        """
        Args:
            x: 输入特征 (batch, seq_len, d_model) 或 (seq_len, d_model)
            class_labels: 可选的类别标签 (batch, seq_len) 或 (seq_len,)，用于监督原型学习
        Returns:
            增强后的特征，形状与输入相同
        """
        # 处理输入维度
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (seq_len, d_model) -> (1, seq_len, d_model)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # 1. 特征投影
        x_proj = self.feature_proj(x)  # (batch, seq_len, d_model)
        
        # 2. 计算特征与所有类别原型的相似度
        # 归一化特征和原型
        x_norm = F.normalize(x_proj, p=2, dim=-1)  # (batch, seq_len, d_model)
        prototypes_norm = F.normalize(self.class_prototypes, p=2, dim=-1)  # (num_classes, d_model)
        
        # 计算余弦相似度
        # (batch, seq_len, d_model) @ (num_classes, d_model).T -> (batch, seq_len, num_classes)
        similarity = torch.matmul(x_norm, prototypes_norm.transpose(0, 1))  # (batch, seq_len, num_classes)
        
        # 应用温度缩放
        similarity_scaled = similarity / self.temperature
        
        # 3. 计算注意力权重（softmax）
        attention_weights = F.softmax(similarity_scaled, dim=-1)  # (batch, seq_len, num_classes)
        
        # 4. 加权聚合类别原型
        # (batch, seq_len, num_classes) @ (num_classes, d_model) -> (batch, seq_len, d_model)
        prototype_proj = self.prototype_proj(self.class_prototypes)  # (num_classes, d_model)
        enhanced_features = torch.matmul(attention_weights, prototype_proj)  # (batch, seq_len, d_model)
        
        # 5. 门控机制：控制增强强度
        gate_weights = self.gate(x)  # (batch, seq_len, d_model)
        enhanced_features = enhanced_features * gate_weights * self.enhancement_scale
        
        # 6. 融合原始特征和增强特征
        x_enhanced = x + enhanced_features  # 残差连接
        
        # 7. 归一化和Dropout
        x_enhanced = self.fusion_norm(x_enhanced)
        x_enhanced = self.fusion_dropout(x_enhanced)
        
        # 恢复原始维度
        if squeeze_output:
            x_enhanced = x_enhanced.squeeze(0)  # (1, seq_len, d_model) -> (seq_len, d_model)
        
        return x_enhanced
    
    def get_prototypes(self):
        """
        获取当前类别原型
        
        Returns:
            类别原型 (num_classes, d_model)
        """
        return self.class_prototypes.detach()
    
    def update_prototypes_with_labels(self, x, class_labels, momentum=0.9):
        """
        使用类别标签更新原型（可选，用于显式原型更新）
        
        Args:
            x: 输入特征 (batch, seq_len, d_model)
            class_labels: 类别标签 (batch, seq_len)
            momentum: 动量系数，用于平滑更新
        """
        if class_labels is None:
            return
        
        with torch.no_grad():
            # 对每个类别，计算该类别的特征均值
            for class_id in range(self.num_classes):
                # 找到属于该类别的特征
                mask = (class_labels == class_id)  # (batch, seq_len)
                if mask.sum() > 0:
                    # 提取该类别的特征
                    class_features = x[mask]  # (n_samples, d_model)
                    # 计算均值
                    class_mean = class_features.mean(dim=0)  # (d_model,)
                    # 动量更新原型
                    self.class_prototypes.data[class_id] = (
                        momentum * self.class_prototypes.data[class_id] +
                        (1 - momentum) * class_mean
                    )

