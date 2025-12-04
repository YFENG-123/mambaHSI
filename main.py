import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import random
from util import load_data
from model import Mamba2HSIClassifier

################################# 固定种子 #################################
random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

################################# 设置超参数 #################################
num_epochs = 500  # 训练轮数
learning_rate = 0.001
dataset_splited_mode = "None"
image_path = "data/Indian_pines.mat"
gt_path = "data/Indian_pines_gt.mat"
val_split_rate = 0
test_split_rate = 0.9

################################# 加载数据 #################################
train_loader, val_loader, test_loader, image_x, image_y, bands, num_classes = load_data(
    image_path=image_path,
    gt_path=gt_path,
    val_split_rate=val_split_rate,
    test_split_rate=test_split_rate,
)
################################# 初始化模型 ################################
model = Mamba2HSIClassifier(num_classes=num_classes, bands=bands).to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

################################# TensorBoard 日志 ##################################
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("runs", "mamba_hsi", timestamp)
writer = SummaryWriter(log_dir=log_dir)
print(
    f"TensorBoard 日志目录: {log_dir} （运行 `tensorboard --logdir runs` 查看实时曲线）"
)

start_time = time.time()
# 记录最佳验证损失，用于保存最佳模型
best_val_loss = float("inf")

for epoch in range(num_epochs):
    # =================== 训练阶段 ====================
    model.train()
    total_loss = 0.0
    correct_label = 0
    total_label = 0
    for i, (image, label, mask) in enumerate(train_loader):
        optimizer.zero_grad()  # 清除梯度

        # 移动数据到 GPU
        image = image.squeeze(0).to("cuda")
        mask = mask.squeeze(0).to("cuda")
        label = label.squeeze(0).to("cuda")

        # 获取模型输出
        outputs = model(image)

        # 拉平数据
        mask_flatten = mask.flatten(start_dim=0, end_dim=1)
        label_flatten = label.flatten(start_dim=0, end_dim=1)
        outputs_flatten = outputs.flatten(start_dim=0, end_dim=1)

        # 获取掩码内的数据
        outputs_flatten_masked = outputs_flatten[mask_flatten]
        label_flatten_masked = label_flatten[mask_flatten]

        # 调整模型
        loss = criterion(outputs_flatten_masked, label_flatten_masked)  # 计算损失
        loss.backward()  # 反向传播和优化
        optimizer.step()  # 优化参数

        # 统计准确率
        total_loss += loss.item()  # 统计损失
        _, predicted = outputs_flatten.max(1)  # 统计预测正确的数量
        total_label += label_flatten.size(0)  # 统计总样本数
        correct_label += predicted.eq(label_flatten).sum().item()  # 统计预测正确的数量

    # 计算打印记录结果
    avg_train_loss = total_loss / len(train_loader)
    train_acc = 100.0 * correct_label / total_label  # 计算准确率
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"  Train -> Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    writer.add_scalars(
        "Loss",
        {
            "Train": avg_train_loss,
        },
        epoch + 1,
    )
    writer.add_scalars(
        "Accuracy",
        {
            "Train": train_acc,
        },
        epoch + 1,
    )

    # ==================== 验证阶段 ====================
    if val_split_rate > 0:
        model.eval()
        val_loss = 0.0
        val_correct_label = 0
        val_total_label = 0
        with torch.no_grad():
            for i, (val_image, val_label, val_mask) in enumerate(val_loader):
                # 移动数据到 GPU
                val_image = val_image.squeeze(0).to("cuda")
                val_label = val_label.squeeze(0).to("cuda")
                val_mask = val_mask.squeeze(0).to("cuda")

                # 获取模型输出
                val_outputs = model(val_image)

                # 拉平数据
                val_outputs_flatten = val_outputs.flatten(start_dim=0, end_dim=1)
                val_label_flatten = val_label.flatten(start_dim=0, end_dim=1)
                val_mask_flatten = val_mask.flatten(start_dim=0, end_dim=1)

                # 获取掩码内的数据
                val_outputs_flatten_masked = val_outputs_flatten[val_mask_flatten]
                val_label_flatten_masked = val_label_flatten[val_mask_flatten]

                # 计算验证集损失
                loss_val = criterion(val_outputs_flatten, val_label_flatten)
                val_loss += loss_val.item()

                # 计算验证集准确率
                _, val_predicted = val_outputs_flatten.max(1)
                val_total_label += val_label_flatten.size(0)
                val_correct_label += val_predicted.eq(val_label_flatten).sum().item()

        # 计算打印记录结果
        avg_val_loss = val_loss / len(val_loader)  # 计算验证集损失
        val_acc = 100.0 * val_correct_label / val_total_label  # 计算准确率
        print(f"  Val   -> Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        writer.add_scalars(
            "Loss",
            {
                "Val": avg_val_loss,
            },
            epoch + 1,
        )
        writer.add_scalars(
            "Accuracy",
            {
                "Val": val_acc,
            },
            epoch + 1,
        )

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  Best model saved with Val Loss: {best_val_loss:.4f}")

# 若未开启验证模式，保存最终模型权重
if val_split_rate <= 0:
    torch.save(model.state_dict(), "best_model.pth")


end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")
################################# 最终测试评估 ##################################
print("\n" + "=" * 50 + "\n开始最终测试评估\n" + "=" * 50)

# 加载最佳模型进行测试
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
all_predictions = []
all_labels = []
with torch.no_grad():
    for i, (test_image, test_label, test_mask) in enumerate(test_loader):
        # 移动数据到 GPU
        test_image = test_image.squeeze(0).to("cuda")
        test_label = test_label.squeeze(0).to("cuda")
        test_mask = test_mask.squeeze(0).to("cuda")

        # 获取模型输出
        test_outputs = model(test_image)

        # 拉平数据
        test_outputs_flatten = test_outputs.flatten(start_dim=0, end_dim=1)
        test_mask_flatten = test_mask.flatten(start_dim=0, end_dim=1)
        test_label_flatten = test_label.flatten(start_dim=0, end_dim=1)

        # 获取掩码内的数据
        test_outputs_masked = test_outputs_flatten[test_mask_flatten]
        test_label_masked = test_label_flatten[test_mask_flatten]

        # 计算测试集损失
        loss_test = criterion(test_outputs_masked, test_label_masked)
        test_loss += loss_test.item()

        # 计算测试集准确率
        _, test_predicted = test_outputs_masked.max(1)
        test_correct += test_predicted.eq(test_label_masked).sum().item()
        test_total += test_label_masked.size(0)

        # 保存预测结果用于后续分析
        all_predictions.append(test_predicted.cpu().numpy())
        all_labels.append(test_label_masked.cpu().numpy())

# 计算打印记录整体测试性能
test_accuracy = 100.0 * test_correct / test_total if test_total > 0 else 0
avg_test_loss = test_loss / len(test_loader)
print("最终测试结果:")
print(f"测试集 Loss: {avg_test_loss:.4f}")
print(f"测试集 Accuracy: {test_accuracy:.2f}%")
writer.add_scalars(
    "Loss",
    {"Test": avg_test_loss},
    num_epochs + 1,
)
writer.add_scalars(
    "Accuracy",
    {"Test": test_accuracy},
    num_epochs + 1,
)

writer.flush()
writer.close()


################################# 生成整个图像的预测图 ##################################
print("\n生成整个测试图像的预测图...")

# 获取测试数据
test_image, test_label, test_mask = next(iter(test_loader))
test_image = test_image.squeeze(0).to("cuda")

with torch.no_grad():
    # 获取整个图像的预测结果
    test_outputs = model(test_image)  # 形状: (145, 145, num_classes)

    # 获取每个像素的预测类别（在最后一个维度上取最大值）
    _, prediction_map = torch.max(test_outputs, dim=2)  # 形状: (145, 145)

    # 转换为CPU和numpy格式用于可视化
    prediction_map_np = prediction_map.cpu().numpy()
    true_label_map_np = test_label.squeeze(0).cpu().numpy()
    test_mask_np = test_mask.squeeze(0).cpu().numpy()

# 创建可视化图像
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 真实标签图（只显示有标签的区域）
true_label_masked = np.ma.masked_where(true_label_map_np == 0, true_label_map_np)
im1 = axes[0, 0].imshow(true_label_masked, cmap="tab20", vmin=0, vmax=num_classes - 1)
axes[0, 0].set_title("真实标签图 (True Labels)")
axes[0, 0].axis("off")
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

# 2. 预测结果图（整个图像）
im2 = axes[0, 1].imshow(prediction_map_np, cmap="tab20", vmin=0, vmax=num_classes - 1)
axes[0, 1].set_title("预测结果图 (Prediction Map)")
axes[0, 1].axis("off")
plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

# 3. 差异图（只显示有标签且预测错误的区域）
error_mask = (prediction_map_np != true_label_map_np) & (true_label_map_np != 0)
error_map = np.ma.masked_where(~error_mask, prediction_map_np)
im3 = axes[1, 0].imshow(error_map, cmap="tab20", vmin=0, vmax=num_classes - 1)
axes[1, 0].set_title("错误预测区域 (Misclassified Pixels)")
axes[1, 0].axis("off")
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

# 4. 置信度图（显示最大预测概率）
confidence_map = torch.softmax(test_outputs, dim=2).max(dim=2)[0].cpu().numpy()
confidence_masked = np.ma.masked_where(true_label_map_np == 0, confidence_map)
im4 = axes[1, 1].imshow(confidence_masked, cmap="viridis", vmin=0, vmax=1)
axes[1, 1].set_title("预测置信度 (Prediction Confidence)")
axes[1, 1].axis("off")
plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

plt.tight_layout()
plt.savefig("final_prediction_results.png", dpi=300, bbox_inches="tight")
plt.close()

print("预测结果图已保存为 'final_prediction_results.png'")

# 计算并显示各类别准确率
print("\n各类别准确率分析:")
with torch.no_grad():
    test_image = test_image.cpu()
    test_label = test_label.cpu()
    test_mask = test_mask.cpu()

    # 只分析有标签的类别（从1开始）
    for class_id in range(1, num_classes):
        class_mask = true_label_map_np == class_id
        if np.sum(class_mask) > 0:  # 确保该类别存在样本
            class_predictions = prediction_map_np[class_mask]
            class_accuracy = np.mean(class_predictions == class_id)
            num_samples = np.sum(class_mask)
            print(
                f"类别 {class_id}: 准确率 {class_accuracy * 100:.2f}% ({num_samples} 个样本)"
            )
