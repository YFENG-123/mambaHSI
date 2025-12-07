import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import Mamba2HSIClassifier
from util import (
    load_data,
    calculate_result,
    run_model,
    generate_picture,
    set_seed,
    write_results_to_txt,
)

################################# 设置超参数 #################################
num_epochs = 1000  # 训练轮数
learning_rate = 0.0005
dropout_rate = 0.50
seeds = [21, 22, 80, 443, 445, 554, 3306, 5900, 8080, 25565]
image_paths = [
    "data/Botswana.mat",
    "data/Indian_pines.mat",
    "data/KSC.mat",
    "data/PaviaU.mat",
    "data/Salinas.mat",
    "data/SalinasA.mat",
]
gt_paths = [
    "data/Botswana_gt.mat",
    "data/Indian_pines_gt.mat",
    "data/KSC_gt.mat",
    "data/PaviaU_gt.mat",
    "data/Salinas_gt.mat",
    "data/SalinasA_gt.mat",
]
val_split_rate = 0.00
test_split_rate = 0.90
print(f"训练轮数:{num_epochs}\t\t学习率:{learning_rate}\t\tDropout率:{dropout_rate}")
print(f"验证集比例:{val_split_rate}\t\t测试集比例:{test_split_rate}")

################################# 记录程序开始时间 ##################################
program_start_time = time.time()

################################# 初始化TensorBoard ##################################
timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("logs", timestamp)
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard 日志目录: {log_dir}, 使用 tensorboard --logdir logs 查看日志")

################################# 创建权重文件目录 ##################################
weights_dir = os.path.join("weights", timestamp)
print(f"权重文件目录: {weights_dir}")

################################# 创建结果记录txt文件 ##################################
results_txt_path = f"results_{timestamp}.txt"
print(f"结果记录文件: {results_txt_path}")
with open(results_txt_path, "w", encoding="utf-8") as results_file:
    results_file.write("=" * 120 + "\n")
    results_file.write(f"实验结果记录 - {timestamp}\n")
    results_file.write("=" * 120 + "\n")
    results_file.write("超参数设置:\n")
    results_file.write(f"  训练轮数: {num_epochs}\n")
    results_file.write(f"  学习率: {learning_rate}\n")
    results_file.write(f"  Dropout率: {dropout_rate}\n")
    results_file.write(f"  验证集比例: {val_split_rate}\n")
    results_file.write(f"  测试集比例: {test_split_rate}\n")
    results_file.write(f"  随机种子: {seeds}\n")
    results_file.write("=" * 120 + "\n\n")

################################# 训练模型 ##################################
for image_path, gt_path in zip(image_paths, gt_paths):  # 遍历所有数据集
    data_name = image_path.split("/")[-1].split(".")[0]  # 获取数据集名称
    experiment_results = []  # 存储每次实验的详细结果
    for seed_idx, seed in enumerate(seeds):  # 遍历所有随机种子
        set_seed(seed)  # 设置随机种子
        print(f"数据集:{data_name}\t第{seed_idx + 1}次实验\t\t随机种子:{seed}")

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

        ################################# 开始一次实验 ##################################

        start_time = time.time()
        best_train_loss = float("inf")
        best_train_acc = 0.0
        best_train_epoch = 0
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_val_epoch = 0
        avg_val_loss = float("inf")
        val_acc = 0.0
        for epoch in range(num_epochs):
            print(f"{data_name}_{seed_idx + 1}_Epoch[{epoch + 1}/{num_epochs}]")

            ############################### 训练阶段 ##################################
            train_start_time = time.time()
            (
                avg_train_loss,
                train_acc,
                all_predictions,
                all_label_masked,
                full_prediction_map,
                full_test_label,
            ) = run_model(  # 运行训练集
                model, train_loader, criterion, optimizer, "train"
            )
            train_end_time = time.time()
            train_time = train_end_time - train_start_time
            ############################### 打印并写入训练结果 ##################################
            print(  # 打印训练集结果
                f"    Train -> Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%, Time: {train_time:.2f}s"
            )
            writer.add_scalars(  # TensorBoard 写入 Loss
                f"Loss_{timestamp}_{data_name}",
                {f"Train_{seed_idx + 1}": avg_train_loss},
                epoch + 1,
            )
            writer.add_scalars(  # TensorBoard 写入 Accuracy
                f"Accuracy_{timestamp}_{data_name}",
                {f"Train_{seed_idx + 1}": train_acc},
                epoch + 1,
            )
            writer.add_scalars(  # TensorBoard 写入训练时间
                f"Time_{timestamp}_{data_name}",
                {f"Train_{seed_idx + 1}": train_time},
                epoch + 1,
            )

            ############################### 验证阶段（如果存在验证集） ##################################
            if val_split_rate > 0:  # 如果存在验证集
                val_start_time = time.time()
                (
                    avg_val_loss,
                    val_acc,
                    all_predictions,
                    all_label_masked,
                    full_prediction_map,
                    full_test_label,
                ) = run_model(  # 运行验证集
                    model, val_loader, criterion, optimizer, "val"
                )
                val_end_time = time.time()
                val_time = val_end_time - val_start_time

                ############################### 打印并写入验证结果（如果存在验证集） ##################################
                print(  # 打印验证集结果
                    f"    Val   -> Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%, Time: {val_time:.2f}s"
                )
                writer.add_scalars(  # TensorBoard 写入 Loss
                    f"Loss_{timestamp}_{data_name}",
                    {f"Val_{seed_idx + 1}": avg_val_loss},
                    epoch + 1,
                )
                writer.add_scalars(  # TensorBoard 写入 Accuracy
                    f"Accuracy_{timestamp}_{data_name}",
                    {f"Val_{seed_idx + 1}": val_acc},
                    epoch + 1,
                )
                writer.add_scalars(  # TensorBoard 写入验证时间
                    f"Time_{timestamp}_{data_name}",
                    {f"Val_{seed_idx + 1}": val_time},
                    epoch + 1,
                )

            ############################### 保存当前最佳模型 ##################################
            if val_split_rate > 0:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = val_acc
                    best_val_epoch = epoch + 1
                    os.makedirs(os.path.join(weights_dir, data_name), exist_ok=True)
                    torch.save(
                        model.state_dict(),
                        os.path.join(weights_dir, data_name, f"{seed_idx + 1}.pth"),
                    )
                    print(
                        f"Best model saved at Epoch {best_val_epoch} with Val Loss: {best_val_loss:.4f} Val Accuracy: {best_val_acc:.2f}%"
                    )
            else:
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss
                    best_train_acc = train_acc
                    best_train_epoch = epoch + 1
                    os.makedirs(os.path.join(weights_dir, data_name), exist_ok=True)
                    torch.save(
                        model.state_dict(),
                        os.path.join(weights_dir, data_name, f"{seed_idx + 1}.pth"),
                    )
                    print(
                        f"Best model saved at Epoch {best_train_epoch} with Train Loss: {best_train_loss:.4f} Train Accuracy: {best_train_acc:.2f}%"
                    )
        end_time = time.time()
        total_training_time = end_time - start_time
        print(f"Training time: {total_training_time:.2f} seconds")

        ################################# 记录该次实验的最佳模型信息 ##################################
        # 确定保存模型时的loss和acc
        if val_split_rate > 0:
            saved_loss = best_val_loss
            saved_acc = best_val_acc
            saved_epoch = best_val_epoch
            saved_type = "Val"
        else:
            saved_loss = best_train_loss
            saved_acc = best_train_acc
            saved_epoch = best_train_epoch
            saved_type = "Train"

        ################################# 写入TensorBoard：最佳模型时的loss和acc ##################################
        # 合并所有数据集到单个表格，每条曲线代表一个数据集
        writer.add_scalars(
            f"Best_Model_Loss_{timestamp}",
            {data_name: saved_loss},
            seed_idx + 1,  # x轴：实验序号
        )
        writer.add_scalars(
            f"Best_Model_Accuracy_{timestamp}",
            {data_name: saved_acc},
            seed_idx + 1,  # x轴：实验序号
        )

        ################################# 开始该次实验的测试评估 ##################################
        model.load_state_dict(  # 加载最佳模型
            torch.load(os.path.join(weights_dir, data_name, f"{seed_idx + 1}.pth"))
        )
        (
            avg_test_loss,
            test_accuracy,
            all_test_predictions,
            all_test_label_masked,
            full_prediction_map,
            full_test_label,
        ) = run_model(model, test_loader, criterion, optimizer, "test")  # 运行测试集
        oa, aa, kappa, confusion_matrix, class_accuracies = (  # 计算测试集结果
            calculate_result(
                avg_test_loss,
                test_accuracy,
                all_test_predictions,
                all_test_label_masked,
                num_classes,
            )
        )

        ################################# 记录该次实验测试评估的结果 ##################################
        # 统一收集本次实验的所有信息，用于TensorBoard和txt记录
        experiment_results.append(
            {
                "seed": seed,
                "seed_idx": seed_idx + 1,
                "total_training_time": total_training_time,
                "test_loss": avg_test_loss,
                "oa": oa,
                "aa": aa,
                "kappa": kappa,
                "performance": (oa + aa + kappa) / 3.0,  # 性能指标
                "class_accuracies": class_accuracies,
                "best_model_loss": saved_loss,
                "best_model_acc": saved_acc,
                "best_model_epoch": saved_epoch,
                "best_model_type": saved_type,
            }
        )

        ################################# 打印该次实验测试评估的结果 ##################################
        print(f"数据集: {data_name}，第{seed_idx + 1}次实验，测试集结果:")
        print(f"测试集 Loss: {avg_test_loss:.4f}")
        print(f"测试集 OA: {oa:.2f}%")
        print(f"测试集 AA: {aa:.2f}%")
        print(f"测试集 Kappa: {kappa:.2f}")

        ################################# 写入TensorBoard：该次实验的 OA、AA、Kappa、Performance、训练时间 ##################################
        # 合并所有数据集到单个表格，每条曲线代表一个数据集
        writer.add_scalars(
            f"OA_{timestamp}",
            {data_name: oa},
            seed_idx + 1,  # x轴：实验序号
        )
        writer.add_scalars(
            f"AA_{timestamp}",
            {data_name: aa},
            seed_idx + 1,  # x轴：实验序号
        )
        writer.add_scalars(
            f"Kappa_{timestamp}",
            {data_name: kappa},
            seed_idx + 1,  # x轴：实验序号
        )
        writer.add_scalars(
            f"Performance_{timestamp}",
            {data_name: (oa + aa + kappa) / 3.0},
            seed_idx + 1,  # x轴：实验序号
        )
        writer.add_scalars(
            f"Training_Time_{timestamp}",
            {data_name: total_training_time},
            seed_idx + 1,  # x轴：实验序号
        )
        # 写入每个类别的精度到TensorBoard（合并为一个表格）
        # x轴：类别序号，y轴：准确率，每条曲线代表一次实验
        for i, acc in enumerate(class_accuracies):
            writer.add_scalars(
                f"Class_Accuracy_{timestamp}_{data_name}",
                {f"Experiment_{seed_idx + 1}": acc},
                i + 1,  # x轴：class序号
            )

        ################################# 可视化该次实验测试评估的结果 ##################################

        # 创建images文件夹（根据timestamp创建子目录）
        images_dir = os.path.join("images", timestamp, data_name)
        os.makedirs(images_dir, exist_ok=True)
        # 生成图片
        generate_picture(
            confusion_matrix,
            num_classes,
            full_prediction_map,
            full_test_label,
            gt,
            data_name,
            seed_idx,
            images_dir=images_dir,
        )

    ################################# 计算该数据集所有实验的平均结果 ##################################
    # 从experiment_results中提取数据计算平均值和标准差
    oa_values = [exp["oa"] for exp in experiment_results]
    aa_values = [exp["aa"] for exp in experiment_results]
    kappa_values = [exp["kappa"] for exp in experiment_results]
    performance_values = [exp["performance"] for exp in experiment_results]
    training_time_values = [exp["total_training_time"] for exp in experiment_results]

    average_oa = np.mean(oa_values)
    average_aa = np.mean(aa_values)
    average_kappa = np.mean(kappa_values)
    average_performance = np.mean(performance_values)
    average_training_time = np.mean(training_time_values)
    std_oa = np.std(oa_values)
    std_aa = np.std(aa_values)
    std_kappa = np.std(kappa_values)
    std_performance = np.std(performance_values)
    std_training_time = np.std(training_time_values)

    ################################# 打印该数据集所有实验的平均结果 ##################################
    print(f"数据集: {data_name}，平均OA: {average_oa:.2f}%")
    print(f"数据集: {data_name}，平均AA: {average_aa:.2f}%")
    print(f"数据集: {data_name}，平均Kappa: {average_kappa:.2f}")
    print(f"数据集: {data_name}，平均性能: {average_performance:.2f}")
    print(f"数据集: {data_name}，平均训练时间: {average_training_time:.2f}秒")

    ################################# 写入TensorBoard：该数据集所有实验的平均结果 ##################################
    # 将平均结果添加到原始值表格的最后一个点后面第5个点位置
    avg_position = len(seeds) + 5
    writer.add_scalars(
        f"OA_{timestamp}",
        {f"{data_name}_Average": average_oa},
        avg_position,
    )
    writer.add_scalars(
        f"AA_{timestamp}",
        {f"{data_name}_Average": average_aa},
        avg_position,
    )
    writer.add_scalars(
        f"Kappa_{timestamp}",
        {f"{data_name}_Average": average_kappa},
        avg_position,
    )
    writer.add_scalars(
        f"Performance_{timestamp}",
        {f"{data_name}_Average": average_performance},
        avg_position,
    )
    writer.add_scalars(
        f"Training_Time_{timestamp}",
        {f"{data_name}_Average": average_training_time},
        avg_position,
    )

    ################################# 统一写入txt文件：该数据集所有实验的平均结果（表格形式） ##################################
    write_results_to_txt(
        results_txt_path=results_txt_path,
        data_name=data_name,
        experiment_results=experiment_results,
        average_oa=average_oa,
        average_aa=average_aa,
        average_kappa=average_kappa,
        average_performance=average_performance,
        average_training_time=average_training_time,
        std_oa=std_oa,
        std_aa=std_aa,
        std_kappa=std_kappa,
        std_performance=std_performance,
        std_training_time=std_training_time,
        num_experiments=len(seeds),
    )

writer.flush()
writer.close()

################################# 计算并打印程序总运行时间 ##################################
program_end_time = time.time()
total_program_time = program_end_time - program_start_time
hours = int(total_program_time // 3600)
minutes = int((total_program_time % 3600) // 60)
seconds = total_program_time % 60

# 将总运行时间写入txt文件
with open(results_txt_path, "a", encoding="utf-8") as results_file:
    results_file.write(f"\n{'=' * 120}\n")
    results_file.write("程序总运行时间:\n")
    results_file.write(
        f"  {hours}小时 {minutes}分钟 {seconds:.2f}秒 (总计: {total_program_time:.2f}秒)\n"
    )
    results_file.write(f"{'=' * 120}\n")

print(f"\n所有结果已保存到: {results_txt_path}")
print(f"\n{'=' * 60}")
print(
    f"程序总运行时间: {hours}小时 {minutes}分钟 {seconds:.2f}秒 ({total_program_time:.2f}秒)"
)
print(f"{'=' * 60}")
