import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
from util import load_data, calculate_result, run_model, generate_picture, set_seed
from model import Mamba2HSIClassifier

################################# 设置超参数 #################################
num_epochs = 1000  # 训练轮数
learning_rate = 0.0005
dropout_rate = 0.50

seeds = [21, 22, 80, 443, 445, 554, 3306, 5900, 8080, 25565]
image_paths = [
    "data/Indian_pines.mat",
    "data/Botswana.mat",
    "data/KSC.mat",
    "data/Pavia.mat",
    "data/PaviaU.mat",
    "data/Salinas.mat",
    "data/SalinasA.mat",
]
gt_paths = [
    "data/Indian_pines_gt.mat",
    "data/Botswana_gt.mat",
    "data/KSC_gt.mat",
    "data/Pavia_gt.mat",
    "data/PaviaU_gt.mat",
    "data/Salinas_gt.mat",
    "data/SalinasA_gt.mat",
]
val_split_rate = 0.00
test_split_rate = 0.90
print(f"训练轮数:{num_epochs}\t\t学习率:{learning_rate}\t\tDropout率:{dropout_rate}")
print(f"验证集比例:{val_split_rate}\t\t测试集比例:{test_split_rate}")

timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("logs", timestamp)
writer = SummaryWriter(log_dir=log_dir)
print(f"TensorBoard 日志目录: {log_dir}, 使用 tensorboard --logdir logs 查看日志")

weights_dir = os.path.join("weights", timestamp)
print(f"权重文件目录: {weights_dir}")

# 创建结果记录txt文件
results_txt_path = f"results_{timestamp}.txt"
results_file = open(results_txt_path, "w", encoding="utf-8")
results_file.write("=" * 80 + "\n")
results_file.write(f"实验结果记录 - {timestamp}\n")
results_file.write("=" * 80 + "\n")
results_file.write("超参数设置:\n")
results_file.write(f"  训练轮数: {num_epochs}\n")
results_file.write(f"  学习率: {learning_rate}\n")
results_file.write(f"  Dropout率: {dropout_rate}\n")
results_file.write(f"  验证集比例: {val_split_rate}\n")
results_file.write(f"  测试集比例: {test_split_rate}\n")
results_file.write(f"  随机种子: {seeds}\n")
results_file.write("=" * 80 + "\n\n")
print(f"结果记录文件: {results_txt_path}")

for image_path, gt_path in zip(image_paths, gt_paths):
    data_name = image_path.split("/")[-1].split(".")[0]
    oa_list = []
    aa_list = []
    kappa_list = []
    performance_list = []  # oa + aa + kappa
    training_time_list = []  # 训练时间列表
    for seed_idx, seed in enumerate(seeds):
        set_seed(seed)
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

        ################################# 训练一次实验的模型 ##################################

        start_time = time.time()
        best_train_loss = float("inf")
        best_train_acc = 0.0
        best_val_loss = float("inf")
        best_val_acc = 0.0
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

            ############################### 验证阶段 ##################################
            avg_val_loss = float("inf")
            val_acc = 0.0
            if val_split_rate > 0:
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

                ############################### 打印并写入验证结果 ##################################
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

            ############################### 保存最佳模型 ##################################
            if val_split_rate > 0:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = val_acc
                    os.makedirs(os.path.join(weights_dir, data_name), exist_ok=True)
                    torch.save(
                        model.state_dict(),
                        os.path.join(weights_dir, data_name, f"{seed_idx + 1}.pth"),
                    )
                    print(
                        f"Best model saved with Val Loss: {best_val_loss:.4f} Val Accuracy: {best_val_acc:.2f}%"
                    )
            else:
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss
                    best_train_acc = train_acc
                    os.makedirs(os.path.join(weights_dir, data_name), exist_ok=True)
                    torch.save(
                        model.state_dict(),
                        os.path.join(weights_dir, data_name, f"{seed_idx + 1}.pth"),
                    )
                    print(
                        f"Best model saved with Train Loss: {best_train_loss:.4f} Train Accuracy: {best_train_acc:.2f}%"
                    )
        end_time = time.time()
        total_training_time = end_time - start_time
        print(f"Training time: {total_training_time:.2f} seconds")

        ################################# 记录保存模型时的loss和acc ##################################
        # 确定保存模型时的loss和acc
        if val_split_rate > 0:
            saved_loss = best_val_loss
            saved_acc = best_val_acc
            saved_type = "Val"
        else:
            saved_loss = best_train_loss
            saved_acc = best_train_acc
            saved_type = "Train"

        # 记录到TensorBoard的最后一个epoch点
        writer.add_scalar(
            f"Best_Model_Loss_{timestamp}_{data_name}",
            saved_loss,
            seed_idx + 1,  # x轴：seed_idx
        )
        writer.add_scalar(
            f"Best_Model_Accuracy_{timestamp}_{data_name}",
            saved_acc,
            seed_idx + 1,  # x轴：seed_idx
        )

        # 记录到txt文件
        results_file.write(
            f"\n数据集: {data_name} - 第{seed_idx + 1}次实验保存的最佳模型 ({saved_type})\n"
        )
        results_file.write(f"  最佳 {saved_type} Loss: {saved_loss:.4f}\n")
        results_file.write(f"  最佳 {saved_type} Accuracy: {saved_acc:.2f}%\n")

        ################################# 测试评估该次实验的模型 ##################################
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
        ################################# 打印该次实验的测试集结果 ##################################
        print(f"数据集: {data_name}，第{seed_idx + 1}次实验，测试集结果:")
        print(f"测试集 Loss: {avg_test_loss:.4f}")
        print(f"测试集 OA: {oa:.2f}%")
        print(f"测试集 AA: {aa:.2f}%")
        print(f"测试集 Kappa: {kappa:.2f}")

        ################################# 写入txt文件：该次实验的每个类别的精度 ##################################
        results_file.write(
            f"\n数据集: {data_name} - 第{seed_idx + 1}次实验 (种子: {seed})\n"
        )
        results_file.write("-" * 80 + "\n")
        results_file.write(f"训练时间: {total_training_time:.2f} 秒\n")
        results_file.write(f"测试集 Loss: {avg_test_loss:.4f}\n")
        results_file.write(f"测试集 OA: {oa:.2f}%\n")
        results_file.write(f"测试集 AA: {aa:.2f}%\n")
        results_file.write(f"测试集 Kappa: {kappa:.2f}\n")
        results_file.write("\n各类别精度:\n")
        for i, acc in enumerate(class_accuracies):  # 写入每类精度
            # 使用add_scalar，x轴表示class序号
            writer.add_scalar(
                f"Class_Accuracy_{timestamp}_{data_name}_{seed_idx + 1}",
                acc,
                i + 1,  # x轴：class序号
            )
            results_file.write(f"  类别 {i + 1}: {acc:.2f}%\n")
        results_file.write("\n")

        ################################# 写入TensorBoard：该次实验的 OA、AA、Kappa、训练时间 ##################################
        writer.add_scalar(
            f"OA_{timestamp}_{data_name}",
            oa,
            seed_idx + 1,  # x轴：seed_idx
        )
        writer.add_scalar(
            f"AA_{timestamp}_{data_name}",
            aa,
            seed_idx + 1,  # x轴：seed_idx
        )
        writer.add_scalar(
            f"Kappa_{timestamp}_{data_name}",
            kappa,
            seed_idx + 1,  # x轴：seed_idx
        )
        writer.add_scalar(
            f"Training_Time_{timestamp}_{data_name}",
            total_training_time,
            seed_idx + 1,  # x轴：seed_idx
        )
        ################################# 记录该次实验的结果 ##################################
        oa_list.append(oa)
        aa_list.append(aa)
        kappa_list.append(kappa)
        performance_list.append((oa + aa + kappa) / 3.0)
        training_time_list.append(total_training_time)

        ################################# 可视化该次实验的结果 ##################################

        # 创建images文件夹
        images_dir = os.path.join("images", data_name)
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
        )

    ################################# 计算平均结果 ##################################
    average_oa = np.mean(oa_list)
    average_aa = np.mean(aa_list)
    average_kappa = np.mean(kappa_list)
    average_performance = np.mean(performance_list)
    average_training_time = np.mean(training_time_list)
    std_oa = np.std(oa_list)
    std_aa = np.std(aa_list)
    std_kappa = np.std(kappa_list)
    std_performance = np.std(performance_list)
    std_training_time = np.std(training_time_list)

    print(f"数据集: {data_name}，平均OA: {average_oa:.2f}%")
    print(f"数据集: {data_name}，平均AA: {average_aa:.2f}%")
    print(f"数据集: {data_name}，平均Kappa: {average_kappa:.2f}")
    print(f"数据集: {data_name}，平均性能: {average_performance:.2f}")
    print(f"数据集: {data_name}，平均训练时间: {average_training_time:.2f}秒")

    # 写入txt文件：平均结果
    results_file.write(f"\n数据集: {data_name} - 平均结果 (共{len(seeds)}次实验)\n")
    results_file.write("=" * 80 + "\n")
    results_file.write(f"平均 OA: {average_oa:.2f}% ± {std_oa:.2f}%\n")
    results_file.write(f"平均 AA: {average_aa:.2f}% ± {std_aa:.2f}%\n")
    results_file.write(f"平均 Kappa: {average_kappa:.2f} ± {std_kappa:.2f}\n")
    results_file.write(
        f"平均性能 (OA+AA+Kappa)/3: {average_performance:.2f} ± {std_performance:.2f}\n"
    )
    results_file.write(
        f"平均训练时间: {average_training_time:.2f}秒 ± {std_training_time:.2f}秒\n"
    )
    results_file.write("\n详细结果:\n")
    for i, seed in enumerate(seeds):
        results_file.write(
            f"  实验 {i + 1} (种子 {seed}): OA={oa_list[i]:.2f}%, AA={aa_list[i]:.2f}%, Kappa={kappa_list[i]:.2f}, 训练时间={training_time_list[i]:.2f}秒\n"
        )
    results_file.write("\n" + "=" * 80 + "\n\n")

    # 写入TensorBoard：平均结果
    writer.add_scalars(
        f"Result_{timestamp}_{data_name}",
        {
            "OA": average_oa,
            "AA": average_aa,
            "Kappa": average_kappa,
            "Performance": average_performance,
            "Training_Time": average_training_time,
        },
        len(seeds) + 1,
    )


writer.flush()
writer.close()

# 关闭结果记录文件
results_file.close()
print(f"\n所有结果已保存到: {results_txt_path}")
