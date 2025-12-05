import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import random
from util import load_data, calculate_result, run_step, generate_picture
from model import Mamba2HSIClassifier


################################# 固定种子（确保完全可复现）#################################
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 设置环境变量以确保完全确定性
os.environ["PYTHONHASHSEED"] = str(seed)
print(f"随机种子已固定为: {seed}")

################################# 初始化 TensorBoard ##################################
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("runs", "mamba_hsi", timestamp)
writer = SummaryWriter(log_dir=log_dir)
print(
    f"TensorBoard 日志目录: {log_dir} （运行 `tensorboard --logdir runs` 查看实时曲线）"
)

################################# 设置超参数 #################################
num_epochs = 1000  # 训练轮数
learning_rate = 0.0001
dropout_rate = 0.3
image_path = "data/Indian_pines.mat"
gt_path = "data/Indian_pines_gt.mat"
val_split_rate = 0.45
test_split_rate = 0.45

################################# 加载数据 #################################
(
    train_loader,
    val_loader,
    test_loader,
    image_x,
    image_y,
    bands,
    num_classes,
    image,
    gt,
) = load_data(
    image_path=image_path,
    gt_path=gt_path,
    val_split_rate=val_split_rate,
    test_split_rate=test_split_rate,
)
################################# 初始化模型 ################################
model = Mamba2HSIClassifier(
    image_x=image_x,
    image_y=image_y,
    num_classes=num_classes,
    bands=bands,
    dropout_rate=dropout_rate,
).to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


################################# 训练模型 ##################################
start_time = time.time()

best_loss = float("inf")  # 记录最佳验证损失，用于保存最佳模型
for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")

    # ================================ 训练阶段 =====================================
    (
        avg_train_loss,
        train_acc,
        all_predictions,
        all_label_masked,
        full_prediction_map,
        full_test_label,
    ) = run_step(model, train_loader, criterion, optimizer, "train")  # 运行训练集
    print(f"Train -> Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    writer.add_scalars(f"Loss{timestamp}", {"Train": avg_train_loss}, epoch + 1)  # 写入TensorBoard
    writer.add_scalars(f"Accuracy{timestamp}", {"Train": train_acc}, epoch + 1)  # 写入TensorBoard

    # ================================== 验证阶段 =================================
    if val_split_rate > 0:
        (
            avg_val_loss,
            val_acc,
            all_predictions,
            all_label_masked,
            full_prediction_map,
            full_test_label,
        ) = run_step(model, val_loader, criterion, optimizer, "val")  # 运行验证集
        print(f"Val   -> Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        writer.add_scalars(f"Loss{timestamp}", {"Val": avg_val_loss}, epoch + 1)  # 写入TensorBoard
        writer.add_scalars(f"Accuracy{timestamp}", {"Val": val_acc}, epoch + 1)  # 写入TensorBoard

        if avg_val_loss < best_loss:  # 保存最佳模型
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  Best model saved with Val Loss: {best_loss:.4f}")
    else:
        if avg_train_loss < best_loss:  # 保存最佳模型
            best_loss = avg_train_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  Best model saved with Train Loss: {best_loss:.4f}")

end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

################################# 测试评估模型 ##################################
print("=" * 50 + "\n开始测试评估模型\n" + "=" * 50)
model.load_state_dict(torch.load("best_model.pth"))  # 加载最佳模型
(
    avg_test_loss,
    test_accuracy,
    all_test_predictions,
    all_test_label_masked,
    full_prediction_map,
    full_test_label,
) = run_step(model, test_loader, criterion, optimizer, "test")
oa, aa, kappa, confusion_matrix = calculate_result(
    writer,
    avg_test_loss,
    test_accuracy,
    all_test_predictions,
    all_test_label_masked,
    num_classes,
    timestamp,
)
writer.add_scalars(f"result{timestamp}", {"OA": oa}, num_classes + 2)
writer.add_scalars(f"result{timestamp}", {"AA": aa}, num_classes + 1)
writer.add_scalars(f"result{timestamp}", {"Kappa": kappa}, num_classes + 3)
print("最终测试结果:")
print(f"测试集 Loss: {avg_test_loss:.4f}")
print(f"测试集 OA: {oa:.2f}%")
print(f"测试集 AA: {aa:.2f}%")
print(f"测试集 Kappa: {kappa:.2f}")

generate_picture(
    confusion_matrix, num_classes, full_prediction_map, full_test_label, gt
)

writer.flush()
writer.close()
