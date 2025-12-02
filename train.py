import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from mamba_ssm import Mamba2
import time
import random


################################# 定义模型 ####################################
class Mamba2HSIClassifier(nn.Module):
    def __init__(
        self, num_classes=17, d_model=256, d_state=64, d_conv=4, expand=2, bands=200
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(bands, self.d_model)
        self.mamba2_block = Mamba2(
            d_model=self.d_model
        )  # 模型维度与输入特征维度（波段数）一致
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.mamba2_block(x)
        x = self.classifier(x)
        return x


################################# 设置超参数 #################################
splited = False
num_epochs = 500  # 训练轮数
num_classes = 17  # 根据实际类别数设置
bands = 200  # 高光谱波段数
model = Mamba2HSIClassifier(num_classes=num_classes, bands=bands).to(
    "cuda"
)  # 如果使用GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


################################# 加载数据 #################################
if splited:
    image_file = "./data/disjoin/Indian_pines.mat"  # (145,145,200)
    train_file = "./data/disjoin/IP_training.mat"  # (145,145)
    test_file = "./data/disjoin/IP_testing.mat"  # (145,145)
    image_file_dict = sio.loadmat(image_file)
    image = image_file_dict["indian_pines"]
    image = torch.from_numpy(image).float()  # 转为浮点类型的张量
    # image.shape : (145,145,200) ; type(image) : tensor ; torch.float32

    train_file_dict = sio.loadmat(train_file)
    train_label = train_file_dict["IP_training"]
    train_label = torch.from_numpy(train_label).long()  # 转为整型类型的张量
    # train_label.shape : (145,145) ; type(train_label) : tensor ; torch.int64

    test_file_dict = sio.loadmat(test_file)
    test_label = test_file_dict["IP_testing"]
    test_label = torch.from_numpy(test_label).long()  # 转为整型类型的张量
    # test_label.shape : (145,145) ; type(test_label) : tensor ; torch.int64

    image = image.flatten(start_dim=0, end_dim=1)  # (21025,200)
    train_label = train_label.flatten(start_dim=0, end_dim=1).unsqueeze(1)  # (21025,1)
    test_label = test_label.flatten(start_dim=0, end_dim=1).unsqueeze(1)  # (21025,1)

    train_dataset = TensorDataset(image, train_label)
    test_dataset = TensorDataset(image, test_label)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=1, shuffle=False, num_workers=0
    )

else:
    splite_rate_train = 0.1

    image_file = "./data/disjoin/Indian_pines.mat"  # (145,145,200)
    label_file = "./data/disjoin/IP_training.mat"  # (145,145)

    image_file_dict = sio.loadmat(image_file)
    image = image_file_dict["indian_pines"]
    image = torch.from_numpy(image).float()
    # image.shape -> (145,145,200) ; type(image) -> tensor ; torch.float32
    image_flatten = image.flatten(start_dim=0, end_dim=1)

    label_file_dict = sio.loadmat(label_file)
    label = label_file_dict["IP_training"]
    label = torch.from_numpy(label).long()
    # label.shape -> (145,145) ; type(label) -> tensor ; torch.int64

    weight, height, channels = image.shape
    max_label = int(torch.max(label))  # 寻找label中的最大值
    label_flatten = label.flatten(start_dim=0, end_dim=1)  # 将label展平 (21025,)
    index_list = [
        torch.where(label_flatten == i)[0].tolist() for i in range(max_label + 1)
    ]
    train_pixel_list = []
    test_pixel_list = []
    for i in range(max_label + 1):
        random.shuffle(index_list[i])
        splite_index = int(len(index_list[i]) * splite_rate_train)
        train_pixel_list.extend(index_list[i][:splite_index])
        test_pixel_list.extend(index_list[i][splite_index:])

    pixel_train = image_flatten[train_pixel_list].unsqueeze(1)
    label_train = label_flatten[train_pixel_list]#.unsqueeze(1)
    '''label_train = (
        torch.tensor(
            [
                [1 if j == i else 0 for j in range(max_label + 1)]
                for i in label_flatten[train_pixel_list]
            ]
        ).float()
        .unsqueeze(1)
    )'''
    pixel_test = image_flatten[test_pixel_list].unsqueeze(1)
    label_test = label_flatten[test_pixel_list]#.unsqueeze(1)
    '''label_test = (
        torch.tensor(
            [
                [1 if j == i else 0 for j in range(max_label + 1)]
                for i in label_flatten[test_pixel_list]
            ]
        ).float()
        .unsqueeze(1)
    )'''

    train_dataset = TensorDataset(pixel_train, label_train)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=1, shuffle=True, num_workers=0
    )
    test_dataset = TensorDataset(pixel_test, label_test)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=True, num_workers=0
    )


################################# 训练模型 ##################################
start_time = time.time()
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式
    total_loss = 0.0
    correct = 0
    total = 0
    for i, (data, label) in enumerate(train_loader):
        data = data.to("cuda")
        label = label.to("cuda")
        optimizer.zero_grad()  # 清除梯度

        outputs = model(data).squeeze(1)
        loss = criterion(outputs, label)
        loss.backward()  # 反向传播和优化

        optimizer.step()  # 优化参数
        total_loss += loss.item()  # 统计损失

        _, predicted = outputs.max(1)  # 统计预测正确的数量
        correct += predicted.eq(label).sum().item()  # 统计预测正确的数量
        total += label.size(0)  # 统计总样本数
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total  # 计算准确率

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%"
    )

end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")


###################################### 测试模型 ####################################
start_time = time.time()
with torch.no_grad():  # 关闭梯度计算
    model.eval()
    all_predictions = []
    for data, labels in test_loader:  # 如果测试集有标签，这里也能获取到
        data = data.to("cuda")
        # labels = labels.to('cuda')  # 如果有标签

        outputs = model(data)
        outputs = outputs.reshape(1, 25, 21025)  # 保持和训练时一致的维度调整
        # 获取预测类别（在类别维度上取最大值的索引）
        _, predicted = torch.max(outputs, 1)  # predicted的形状应为[1, 21025]

        # 将预测结果移回CPU并转换为NumPy数组，便于后续分析
        all_predictions.append(predicted.cpu().numpy())

end_time = time.time()
print(f"Testing time: {end_time - start_time:.2f} seconds")
all_predictions = np.concatenate(all_predictions, axis=0)
prediction_map = all_predictions.reshape(145, 145)


# 保存为PNG图像
plt.figure(figsize=(10, 10))
plt.imshow(prediction_map, cmap="tab20")  # 使用tab20色彩映射，支持多类别
plt.colorbar()
plt.axis("off")
plt.savefig("prediction_result.png", bbox_inches="tight", dpi=300, transparent=True)
plt.close()
