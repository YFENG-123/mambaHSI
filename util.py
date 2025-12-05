import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import random
import matplotlib.pyplot as plt


def load_data(
    image_path,
    gt_path,
    val_split_rate=0.1,
    test_split_rate=0.1,
):
    # 加载image
    image_dict = sio.loadmat(image_path)
    image_np = (
        next(  # 图像是三维矩阵:  (H, W, C)，找第一个ndim为3的 numpy.ndarray 作为image
            m for m in image_dict.values() if isinstance(m, np.ndarray) and m.ndim == 3
        )
    )
    image = torch.from_numpy(image_np).float()
    image = image.unsqueeze(0)  # (1, H, W, C)

    # 提取image信息
    image_x, image_y, channel = image_np.shape

    # 加载gt
    gt_dict = sio.loadmat(gt_path)
    gt_np = next(  # gt 是二维矩阵: (H, W)，找第一个ndim为2的 numpy.ndarray 作为gt
        m for m in gt_dict.values() if isinstance(m, np.ndarray) and m.ndim == 2
    )
    gt = torch.from_numpy(gt_np).long()

    # 提取gt信息
    max_label = int(torch.max(gt))
    gt_flatten = gt.flatten(start_dim=0, end_dim=1)
    label_index_list = [
        torch.where(gt_flatten == i)[0].tolist() for i in range(max_label + 1)
    ]

    # 分割gt
    train_label_index_list = []
    test_label_index_list = []
    val_label_index_list = []
    for i in range(1, max_label + 1):
        random.shuffle(label_index_list[i])
        test_split_index = int(len(label_index_list[i]) * test_split_rate)
        val_split_index = int(
            len(label_index_list[i]) * (test_split_rate + val_split_rate)
        )
        test_label_index_list.extend(label_index_list[i][:test_split_index])
        val_label_index_list.extend(
            label_index_list[i][test_split_index:val_split_index]
        )
        train_label_index_list.extend(label_index_list[i][val_split_index:])

    # 生成掩码，并生成数据集
    train_mask = np.zeros(gt_flatten.shape)
    train_mask[train_label_index_list] = 1
    train_mask = train_mask.reshape(gt.shape)
    train_mask = torch.from_numpy(train_mask)
    train_mask = train_mask.bool()
    train_mask = train_mask.unsqueeze(0)
    train_label = gt.unsqueeze(0)
    train_dataset = TensorDataset(image, train_label, train_mask)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=1,
        shuffle=False,  # 确保数据顺序固定
        num_workers=0,  # 设置为0以避免多进程随机性
        generator=torch.Generator().manual_seed(42) if torch.cuda.is_available() else None
    )

    val_mask = np.zeros(gt_flatten.shape)
    val_mask[val_label_index_list] = 1
    val_mask = val_mask.reshape(gt.shape)
    val_mask = torch.from_numpy(val_mask)
    val_mask = val_mask.bool()
    val_mask = val_mask.unsqueeze(0)
    val_label = gt.unsqueeze(0)
    val_dataset = TensorDataset(image, val_label, val_mask)
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(42) if torch.cuda.is_available() else None
    )

    test_mask = np.zeros(gt_flatten.shape)
    test_mask[test_label_index_list] = 1
    test_mask = test_mask.reshape(gt.shape)
    test_mask = torch.from_numpy(test_mask)
    test_mask = test_mask.bool()
    test_mask = test_mask.unsqueeze(0)
    test_label = gt.unsqueeze(0)
    test_dataset = TensorDataset(image, test_label, test_mask)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(42) if torch.cuda.is_available() else None
    )

    # 注意：Indian Pines 等数据集的标签通常是 0 表示背景，1~max_label 表示类别，
    # 因此真实的类别总数应为 max_label + 1（包含背景类 0）。
    # CrossEntropyLoss 要求 target ∈ [0, num_classes-1]，如果 num_classes == max_label，
    # 而标签中存在值为 max_label 的像素，就会触发 CUDA device-side assert。
    # 这里返回 max_label + 1，保证标签上界 < num_classes。
    return (
        train_loader,
        val_loader,
        test_loader,
        image_x,
        image_y,
        channel,
        max_label + 1,
        image,
        gt,
    )


# 使用一个函数完成上面三个功能
def run_step(model, loader, criterion, optimizer, mode):
    if mode == "train":
        model.train()
        return step(model, loader, criterion, optimizer, mode)
    elif mode == "val":
        model.eval()
        with torch.no_grad():
            return step(model, loader, criterion, optimizer, mode)
    elif mode == "test":
        model.eval()
        with torch.no_grad():
            return step(model, loader, criterion, optimizer, mode)
    raise ValueError(f"Invalid mode: {mode}")


def step(model, loader, criterion, optimizer, mode):
    total_loss = 0.0
    correct_label = 0
    total_label = 0
    all_predictions = []
    all_label_masked = []
    # 用于保存整个图像的预测结果（仅在测试模式下）
    full_prediction_map = None
    full_test_label = None
    for i, (image, label, mask) in enumerate(loader):
        # 移动数据到 GPU
        image = image.squeeze(0).to("cuda")
        label = label.squeeze(0).to("cuda")
        mask = mask.squeeze(0).to("cuda")

        # 获取模型输出
        outputs = model(image)

        # 拉平数据
        outputs_flatten = outputs.flatten(start_dim=0, end_dim=1)
        mask_flatten = mask.flatten(start_dim=0, end_dim=1)
        label_flatten = label.flatten(start_dim=0, end_dim=1)

        # 获取掩码内的数据
        outputs_masked = outputs_flatten[mask_flatten]
        label_masked = label_flatten[mask_flatten]

        # 计算损失
        loss = criterion(outputs_masked, label_masked)
        total_loss += loss.item()

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 计算准确率
        _, predicted = outputs_masked.max(1)
        correct_label += predicted.eq(label_masked).sum().item()
        total_label += label_masked.size(0)

        if mode == "test":
            # 保存预测结果用于后续分析
            all_predictions.append(predicted.cpu().numpy())
            all_label_masked.append(label_masked.cpu().numpy())
            # 保存整个图像的预测结果和测试标签
            _, full_pred = torch.max(outputs, dim=2)
            full_prediction_map = full_pred.cpu().numpy()
            full_test_label = label.cpu().numpy()

    avg_loss = total_loss / len(loader)
    acc = 100.0 * correct_label / total_label

    return (
        avg_loss,
        acc,
        all_predictions,
        all_label_masked,
        full_prediction_map,
        full_test_label,
    )


def calculate_result(
    writer,
    avg_test_loss,
    test_accuracy,
    all_test_predictions,
    all_test_label_masked,
    num_classes,
):
    # 1)计算打印记录（OA）
    oa = test_accuracy

    # 2) AA：各类别精度的平均（忽略背景类 0），并输出每个类别精度
    class_accuracies = []
    all_predictions_np = np.concatenate(all_test_predictions, axis=0)
    all_labels_np = np.concatenate(all_test_label_masked, axis=0)
    print("各类别精度（per-class accuracy）:")
    for class_id in range(1, num_classes):
        classified_label_mask = all_labels_np == class_id
        acc_c = 100.0 * (all_predictions_np[classified_label_mask] == class_id).mean()
        class_accuracies.append(acc_c)
        print(
            f"  类别 {class_id}: 精度 {acc_c:.2f}% ({classified_label_mask.sum()} 个样本)"
        )
        # 在循环中直接写入 TensorBoard
        writer.add_scalars("result", {f"class_{class_id}": acc_c}, class_id)
    aa = float(np.mean(class_accuracies))

    # 3) Kappa：基于混淆矩阵（包含背景类 0）
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels_np, all_predictions_np):
        cm[t, p] += 1
    total_samples = cm.sum()
    po = np.trace(cm) / total_samples
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (total_samples**2)
    kappa = float((po - pe) / (1 - pe) * 100.0)

    return oa, aa, kappa, cm


def generate_picture(
    confusion_matrix, num_classes, prediction_map=None, test_label=None, gt=None
):
    plt.rcParams["font.sans-serif"] = [
        "WenQuanYi Zen Hei"
    ]  #  Linux 系统推荐使用文泉驿微米黑
    plt.rcParams["axes.unicode_minus"] = (
        False  # 解决负号（'-'）显示为方块的问题 [1,2,3](@ref)
    )

    # 生成六张图：左上ground truth，中上test_label，右上错误区域，左下完整预测结果，中下预测结果，右下隐藏
    if prediction_map is not None and test_label is not None and gt is not None:
        print("\n生成分类结果可视化图片...")
        # 使用已计算好的预测结果，不需要重新运行模型
        prediction_map_np = prediction_map
        test_label_np = test_label.squeeze(0) if test_label.ndim == 3 else test_label

        # 计算错误预测区域
        error_mask = (prediction_map_np != test_label_np) & (test_label_np != 0)
        error_map = prediction_map_np.copy()
        error_map[~error_mask] = 0  # 只保留错误预测的区域

        # 创建2x3布局可视化（六张图）
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))

        # 左上：Ground Truth（整个图像的标签）
        gt_masked = np.ma.masked_where(gt == 0, gt)
        im1 = axes[0, 0].imshow(gt_masked, cmap="tab20", vmin=0, vmax=num_classes - 1)
        axes[0, 0].set_title("Ground Truth (完整标签)", fontsize=14, fontweight="bold")
        axes[0, 0].axis("off")
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

        # 中上：Test Label（测试集标签）
        test_label_masked = np.ma.masked_where(test_label_np == 0, test_label_np)
        im2 = axes[0, 1].imshow(
            test_label_masked, cmap="tab20", vmin=0, vmax=num_classes - 1
        )
        axes[0, 1].set_title("Test Label (测试集标签)", fontsize=14, fontweight="bold")
        axes[0, 1].axis("off")
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

        # 右上：分类错误区域
        error_map_masked = np.ma.masked_where(error_map == 0, error_map)
        im3 = axes[0, 2].imshow(
            error_map_masked, cmap="tab20", vmin=0, vmax=num_classes - 1
        )
        axes[0, 2].set_title(
            "Misclassified Pixels (错误预测区域)", fontsize=14, fontweight="bold"
        )
        axes[0, 2].axis("off")
        plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

        # 左下：完整预测结果（不去除背景区域）
        im4 = axes[1, 0].imshow(
            prediction_map_np, cmap="tab20", vmin=0, vmax=num_classes - 1
        )
        axes[1, 0].set_title(
            "Full Prediction (完整预测结果)", fontsize=14, fontweight="bold"
        )
        axes[1, 0].axis("off")
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

        # 中下：预测结果（去除背景区域）
        prediction_masked = np.ma.masked_where(gt == 0, prediction_map_np)
        im5 = axes[1, 1].imshow(
            prediction_masked, cmap="tab20", vmin=0, vmax=num_classes - 1
        )
        axes[1, 1].set_title(
            "Prediction Map (预测结果，去除背景)", fontsize=14, fontweight="bold"
        )
        axes[1, 1].axis("off")
        plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

        # 右下：隐藏第六个子图
        axes[1, 2].axis("off")

        plt.tight_layout()
        prediction_result_path = "prediction_results.png"
        plt.savefig(prediction_result_path, dpi=300, bbox_inches="tight")
        print(f"分类结果可视化已保存为: {prediction_result_path}")
        plt.close()

    # 生成并保存混淆矩阵图片
    print("\n生成混淆矩阵图片...")
    # 归一化混淆矩阵（按行归一化，显示每个真实类别的预测分布）
    confusion_matrix_float = confusion_matrix.astype(np.float32)
    row_sums = confusion_matrix_float.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    cm_normalized = confusion_matrix_float / row_sums

    # 创建混淆矩阵可视化
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 左图：原始计数混淆矩阵
    im1 = axes[0].imshow(confusion_matrix, cmap="Blues", interpolation="nearest")
    axes[0].set_title("Confusion Matrix - Counts", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("预测类别 (Predicted Class)", fontsize=12)
    axes[0].set_ylabel("真实类别 (True Class)", fontsize=12)
    axes[0].set_xticks(range(num_classes))
    axes[0].set_yticks(range(num_classes))
    axes[0].set_xticklabels(range(num_classes))
    axes[0].set_yticklabels(range(num_classes))
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # 在原始矩阵上添加数值标注
    thresh1 = confusion_matrix.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            axes[0].text(
                j,
                i,
                format(confusion_matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if confusion_matrix[i, j] > thresh1 else "black",
                fontsize=8,
            )

    # 右图：归一化混淆矩阵（百分比）
    im2 = axes[1].imshow(
        cm_normalized, cmap="Blues", interpolation="nearest", vmin=0, vmax=1
    )
    axes[1].set_title("Confusion Matrix - Normalized", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("预测类别 (Predicted Class)", fontsize=12)
    axes[1].set_ylabel("真实类别 (True Class)", fontsize=12)
    axes[1].set_xticks(range(num_classes))
    axes[1].set_yticks(range(num_classes))
    axes[1].set_xticklabels(range(num_classes))
    axes[1].set_yticklabels(range(num_classes))
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # 在归一化矩阵上添加百分比标注
    thresh2 = 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            axes[1].text(
                j,
                i,
                format(cm_normalized[i, j], ".2f"),
                ha="center",
                va="center",
                color="white" if cm_normalized[i, j] > thresh2 else "black",
                fontsize=8,
            )

    plt.tight_layout()
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
    print(f"混淆矩阵已保存为: {confusion_matrix_path}")
    plt.close()
