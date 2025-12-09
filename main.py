import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models import Net
from util import (
    load_data,
    calculate_seed_result,
    calculate_dataset_result,
    run_model,
    generate_picture,
    set_seed,
    write_results_to_txt,
)
from log import Log

################################# 记录实验开始时间 ##################################
program_start_time = time.time()

################################# 设置超参数 #################################
num_epochs = 1000  # 训练轮数
learning_rate =  1e-1
dropout_rate = 0.5
################################# 优化器参数 ##################################
optimizer_type = "SGD"  # 优化器类型: "SGD", "Adam", "AdamW", "RMSprop", "Adagrad"
# SGD 参数（随机梯度下降）
momentum = 0.9  # 动量因子
weight_decay = 1e-4  # 权重衰减（L2正则化）
nesterov = False  # 是否使用Nesterov动量
# Adam/AdamW 参数
adam_beta1 = 0.9  # Adam的beta1参数
adam_beta2 = 0.999  # Adam的beta2参数
adam_eps = 1e-8  # Adam的epsilon参数

################################# 学习率调度器参数 ##################################
use_scheduler = True  # 是否使用学习率调度器
scheduler_type = "CosineAnnealingWarmRestarts"  # 调度器类型: "ReduceLROnPlateau", "StepLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "ExponentialLR", "MultiStepLR"
# ReduceLROnPlateau 参数（根据验证损失自动调整）
scheduler_patience = 100  # 验证损失不下降的等待轮数
scheduler_factor = 0.8  # 学习率衰减因子
scheduler_min_lr = 1e-4  # 最小学习率
# StepLR 参数（每隔固定轮数降低学习率）
step_size = 100  # 每多少轮降低一次学习率
gamma = 0.5  # 学习率衰减因子
# CosineAnnealingLR 参数（余弦退火）
T_max = num_epochs  # 余弦退火的周期
# CosineAnnealingWarmRestarts 参数（带热重启的余弦退火）
T_0 = 250  # 第一次重启的周期
T_mult = 1  # 重启后周期的倍数（1表示每次重启周期相同，2表示每次重启周期翻倍）
# ExponentialLR 参数（指数衰减）
exp_gamma = 0.95  # 每个epoch的衰减因子
# MultiStepLR 参数（在指定里程碑降低学习率）
milestones = [300, 600, 800]  # 降低学习率的epoch列表
seeds = [21, 22, 80, 443, 445, 554, 3306, 5900, 8080, 25565]
image_paths = [
    # "data/HuaiLai.mat",
    # "data/Botswana.mat",
    "data/Indian_pines.mat",
    # "data/KSC.mat",
    # "data/Pavia.mat",
    # "data/PaviaU.mat",
    # "data/Salinas.mat",
    # "data/SalinasA.mat",
]
gt_paths = [
    # "data/HuaiLai.mat",
    # "data/Botswana_gt.mat",
    "data/Indian_pines_gt.mat",
    # "data/KSC_gt.mat",
    # "data/Pavia_gt.mat",
    # "data/PaviaU_gt.mat",
    # "data/Salinas_gt.mat",
    # "data/SalinasA_gt.mat",
]
val_split_rate = 0.45
test_split_rate = 0.45
print(f"训练轮数:{num_epochs}\t\t学习率:{learning_rate}\t\tDropout率:{dropout_rate}")
print(f"优化器类型:{optimizer_type}\t\t验证集比例:{val_split_rate}\t\t测试集比例:{test_split_rate}")
if use_scheduler:
    print(f"学习率调度器:{scheduler_type}\t\t启用动态学习率")
    
################################# 创建结果目录结构 ##################################
timestamp = time.strftime("%Y%m%d-%H%M%S")
results_base_dir = os.path.join("results", timestamp)
print(f"结果目录: {results_base_dir}")

weights_base_dir = os.path.join(results_base_dir, "weights")
os.makedirs(weights_base_dir, exist_ok=True)
print(f"权重文件目录: {weights_base_dir}")

images_base_dir = os.path.join(results_base_dir, "images")
os.makedirs(images_base_dir, exist_ok=True)
print(f"图像文件目录: {images_base_dir}")

results_txt_path = os.path.join(results_base_dir, f"results_{timestamp}.txt")
with open(results_txt_path, "w", encoding="utf-8") as results_file:
    results_file.write("=" * 120 + "\n")
    results_file.write(f"实验结果记录 - {timestamp}\n")
    results_file.write("=" * 120 + "\n")
    results_file.write("超参数设置:\n")
    results_file.write(f"  训练轮数: {num_epochs}\n")
    results_file.write(f"  学习率: {learning_rate}\n")
    results_file.write(f"  Dropout率: {dropout_rate}\n")
    results_file.write(f"  优化器类型: {optimizer_type}\n")
    # 写入优化器参数
    if optimizer_type == "SGD":
        results_file.write(f"    动量: {momentum}\n")
        results_file.write(f"    权重衰减: {weight_decay}\n")
        results_file.write(f"    Nesterov: {nesterov}\n")
    elif optimizer_type in ["Adam", "AdamW"]:
        results_file.write(f"    Beta1: {adam_beta1}\n")
        results_file.write(f"    Beta2: {adam_beta2}\n")
        results_file.write(f"    Epsilon: {adam_eps}\n")
        results_file.write(f"    权重衰减: {weight_decay}\n")
    elif optimizer_type == "RMSprop":
        results_file.write(f"    动量: {momentum}\n")
        results_file.write(f"    权重衰减: {weight_decay}\n")
    elif optimizer_type == "Adagrad":
        results_file.write(f"    权重衰减: {weight_decay}\n")
        results_file.write(f"    Epsilon: {adam_eps}\n")
    results_file.write(f"  验证集比例: {val_split_rate}\n")
    results_file.write(f"  测试集比例: {test_split_rate}\n")
    results_file.write(f"  随机种子: {seeds}\n")
    # 写入学习率调度器参数
    if use_scheduler:
        results_file.write(f"  学习率调度器: {scheduler_type} (PyTorch自带)\n")
        if scheduler_type == "ReduceLROnPlateau":
            results_file.write(f"    耐心值: {scheduler_patience}\n")
            results_file.write(f"    衰减因子: {scheduler_factor}\n")
            results_file.write(f"    最小学习率: {scheduler_min_lr}\n")
        elif scheduler_type == "StepLR":
            results_file.write(f"    步长: {step_size}\n")
            results_file.write(f"    衰减因子: {gamma}\n")
        elif scheduler_type == "CosineAnnealingLR":
            results_file.write(f"    周期: {T_max}\n")
            results_file.write(f"    最小学习率: {scheduler_min_lr}\n")
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            results_file.write(f"    初始周期: {T_0}\n")
            results_file.write(f"    周期倍数: {T_mult}\n")
            results_file.write(f"    最小学习率: {scheduler_min_lr}\n")
        elif scheduler_type == "ExponentialLR":
            results_file.write(f"    衰减因子: {exp_gamma}\n")
        elif scheduler_type == "MultiStepLR":
            results_file.write(f"    里程碑: {milestones}\n")
            results_file.write(f"    衰减因子: {gamma}\n")
    results_file.write("=" * 120 + "\n\n")
print(f"结果记录文件: {results_txt_path}")

log_base_dir = os.path.join(results_base_dir, "logs")
os.makedirs(log_base_dir, exist_ok=True)
logger = Log(log_base_dir, timestamp)
print(
    f"TensorBoard 日志目录: {log_base_dir}, 查看日志使用: tensorboard --logdir {log_base_dir}"
)

################################# 遍历所有数据集和种子 ##################################
for image_path, gt_path in zip(image_paths, gt_paths):
    data_name = image_path.split("/")[-1].split(".")[0]  # 获取数据集名称
    experiment_results = []  # 存储每个种子的详细结果
    weights_dir = os.path.join(weights_base_dir, data_name)
    images_dir = os.path.join(images_base_dir, data_name)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    for seed_idx, seed in enumerate(seeds):
        set_seed(seed)  # 设置种子
        print(f"数据集:{data_name}\t第{seed_idx + 1}个种子\t\t随机种子:{seed}")
        ############################### 每个种子重新随机初始化数据集分割 ##################################
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
        ############################### 每个种子重新随机初始化模型参数 ##################################
        model = Net(
            image_x=image_x,
            image_y=image_y,
            num_classes=num_classes,
            bands=bands,
            dropout_rate=dropout_rate,
        ).to("cuda")
        criterion = nn.CrossEntropyLoss()
        ################################# 创建优化器 ##################################
        if optimizer_type == "SGD":
            # SGD: 随机梯度下降优化器
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov
            )
        elif optimizer_type == "Adam":
            # Adam: 自适应矩估计优化器
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                betas=(adam_beta1, adam_beta2),
                eps=adam_eps,
                weight_decay=weight_decay
            )
        elif optimizer_type == "AdamW":
            # AdamW: 带权重衰减的Adam优化器
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(adam_beta1, adam_beta2),
                eps=adam_eps,
                weight_decay=weight_decay
            )
        elif optimizer_type == "RMSprop":
            # RMSprop: 均方根传播优化器
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_type == "Adagrad":
            # Adagrad: 自适应梯度优化器
            optimizer = optim.Adagrad(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                eps=adam_eps
            )
        else:
            # 默认使用SGD
            print(f"警告: 未知的优化器类型 {optimizer_type}，使用默认SGD")
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        ################################# 创建学习率调度器（使用PyTorch自带的lr_scheduler）##################################
        scheduler = None
        if use_scheduler:
            if scheduler_type == "ReduceLROnPlateau" and val_split_rate > 0:
                # ReduceLROnPlateau: 根据验证损失自动调整学习率（PyTorch自带）
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=scheduler_factor,
                    patience=scheduler_patience,
                    min_lr=scheduler_min_lr,
                )
            elif scheduler_type == "StepLR":
                # StepLR: 每隔固定轮数降低学习率（PyTorch自带）
                scheduler = lr_scheduler.StepLR(
                    optimizer, step_size=step_size, gamma=gamma
                )
            elif scheduler_type == "CosineAnnealingLR":
                # CosineAnnealingLR: 余弦退火学习率（PyTorch自带）
                scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=T_max, eta_min=scheduler_min_lr
                )
            elif scheduler_type == "CosineAnnealingWarmRestarts":
                # CosineAnnealingWarmRestarts: 带热重启的余弦退火学习率（PyTorch自带）
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=T_0, T_mult=T_mult, eta_min=scheduler_min_lr
                )
            elif scheduler_type == "ExponentialLR":
                # ExponentialLR: 指数衰减学习率（PyTorch自带）
                scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=exp_gamma)
            elif scheduler_type == "MultiStepLR":
                # MultiStepLR: 在指定里程碑降低学习率（PyTorch自带）
                scheduler = lr_scheduler.MultiStepLR(
                    optimizer, milestones=milestones, gamma=gamma
                )
            else:
                print(
                    f"警告: 未知的调度器类型 {scheduler_type}，请使用: ReduceLROnPlateau, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, MultiStepLR"
                )
        ################################# 开始一个种子的训练和验证 ##################################
        best_train_loss = float("inf")
        best_train_acc = 0.0
        best_train_epoch = 0
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_val_epoch = 0
        avg_val_loss = float("inf")
        val_acc = 0.0
        start_time = time.time()  # 记录每个种子开始时间
        for epoch in range(num_epochs):
            print(f"{data_name}_{seed_idx + 1}_Epoch[{epoch + 1}/{num_epochs}]")
            ############################### 每个epoch的训练阶段 ##################################
            (
                avg_train_loss,
                train_acc,
                all_predictions,
                all_label_masked,
                full_prediction_map,
                full_test_label,
                train_time,
            ) = run_model(model, train_loader, criterion, optimizer, "train")
            logger.each_train(  # 记录每次训练的结果到TensorBoard
                data_name,
                seed_idx + 1,
                epoch + 1,
                avg_train_loss,
                train_acc,
                train_time,
            )
            print(  # 打印训练集结果
                f"    Train -> Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%, Time: {train_time:.2f}s"
            )
            ############################### 每个epoch的验证阶段 ##################################
            if val_split_rate <= 0:
                # 如果没有验证集，在训练后更新学习率（适用于StepLR和CosineAnnealingLR）
                if scheduler is not None and scheduler_type != "ReduceLROnPlateau":
                    scheduler.step()
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(f"    当前学习率: {current_lr:.6f}")
                continue
            (
                avg_val_loss,
                val_acc,
                all_predictions,
                all_label_masked,
                full_prediction_map,
                full_test_label,
                val_time,
            ) = run_model(model, val_loader, criterion, optimizer, "val")
            logger.each_val(  # 记录每次验证的结果到TensorBoard
                data_name,
                seed_idx + 1,
                epoch + 1,
                avg_val_loss,
                val_acc,
                val_time,
            )
            print(  # 打印验证集结果
                f"    Val   -> Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%, Time: {val_time:.2f}s"
            )
            ############################### 更新学习率 ##################################
            if scheduler is not None:
                if scheduler_type == "ReduceLROnPlateau" and val_split_rate > 0:
                    scheduler.step(int(avg_val_loss))  # 根据验证损失更新学习率
                else:
                    scheduler.step()  # 根据epoch更新学习率
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"    当前学习率: {current_lr:.6f}")
            ############################### 每个epoch保存当前最佳模型 ##################################
            if val_split_rate > 0:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = val_acc
                    best_val_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(weights_dir, f"{seed_idx + 1}.pth"),
                    )
                    print(
                        f"Best model saved at Epoch {best_val_epoch} with Val Loss: {best_val_loss:.4f} Val Accuracy: {best_val_acc:.2f}%"
                    )
            else:
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss
                    best_train_acc = train_acc
                    best_train_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(weights_dir, f"{seed_idx + 1}.pth"),
                    )
                    print(
                        f"Best model saved at Epoch {best_train_epoch} with Train Loss: {best_train_loss:.4f} Train Accuracy: {best_train_acc:.2f}%"
                    )
        end_time = time.time()  # 记录每个种子结束时间
        total_training_time = end_time - start_time  # 计算每个种子训练时间
        print(f"Training time: {total_training_time:.2f} seconds")
        ################################# 记录一个种子的最佳模型信息 ##################################
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
        logger.each_seed(  # 记录一个种子的最佳模型信息到TensorBoard
            data_name,
            seed_idx + 1,
            saved_loss,
            saved_acc,
            saved_epoch,
            saved_type,
        )
        ################################# 完成一个种子训练和验证后的测试评估 ##################################
        model.load_state_dict(  # 加载一个种子的最佳模型
            torch.load(os.path.join(weights_dir, f"{seed_idx + 1}.pth"))
        )
        (
            avg_test_loss,
            test_accuracy,
            all_test_predictions,
            all_test_label_masked,
            full_prediction_map,
            full_test_label,
            test_time,
        ) = run_model(model, test_loader, criterion, optimizer, "test")
        oa, aa, kappa, confusion_matrix, class_accuracies = (  # 计算测试集结果
            calculate_seed_result(
                avg_test_loss,
                test_accuracy,
                all_test_predictions,
                all_test_label_masked,
                num_classes,
            )
        )
        ################################# 记录并打印一个种子的训练和验证后的测试评估的结果 ##################################
        generate_picture(  # 生成一个种子的图片
            confusion_matrix,
            num_classes,
            full_prediction_map,
            full_test_label,
            gt,
            data_name,
            seed_idx,
            images_dir=images_dir,
        )
        print(f"数据集: {data_name}，第{seed_idx + 1}个种子，测试集结果:")
        print(f"测试集 Loss: {avg_test_loss:.4f}")
        print(f"测试集 OA: {oa:.2f}%")
        print(f"测试集 AA: {aa:.2f}%")
        print(f"测试集 Kappa: {kappa:.2f}")
        logger.each_test(  # 记录一个种子的测试结果到TensorBoard
            data_name,
            seed_idx + 1,
            oa,
            aa,
            kappa,
            (oa + aa + kappa) / 3.0,
            total_training_time,
            class_accuracies,
        )
        experiment_results.append(  # 统一收集一个种子的所有信息，用于一个数据集所有种子的平均结果计算
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

    ################################# 计算一个数据集所有种子的平均结果 ##################################
    (
        average_oa,
        average_aa,
        average_kappa,
        average_performance,
        average_training_time,
        average_test_loss,
        average_best_model_loss,
        average_best_model_acc,
        average_best_model_epoch,
        std_oa,
        std_aa,
        std_kappa,
        std_performance,
        std_training_time,
        std_test_loss,
        std_best_model_loss,
        std_best_model_acc,
        std_best_model_epoch,
    ) = calculate_dataset_result(experiment_results)
    ################################# 打印一个数据集所有种子的平均结果 ##################################
    print(f"数据集: {data_name}，平均OA: {average_oa:.2f}%")
    print(f"数据集: {data_name}，平均AA: {average_aa:.2f}%")
    print(f"数据集: {data_name}，平均Kappa: {average_kappa:.2f}")
    print(f"数据集: {data_name}，平均性能: {average_performance:.2f}")
    print(f"数据集: {data_name}，平均训练时间: {average_training_time:.2f}秒")
    logger.each_dataset_average(
        data_name,
        len(seeds),
        average_oa,
        average_aa,
        average_kappa,
        average_performance,
        average_training_time,
    )
    write_results_to_txt(
        results_txt_path=results_txt_path,
        data_name=data_name,
        experiment_results=experiment_results,
        average_oa=average_oa,
        average_aa=average_aa,
        average_kappa=average_kappa,
        average_performance=average_performance,
        average_training_time=average_training_time,
        average_test_loss=average_test_loss,
        average_best_model_loss=average_best_model_loss,
        average_best_model_acc=average_best_model_acc,
        average_best_model_epoch=average_best_model_epoch,
        std_oa=std_oa,
        std_aa=std_aa,
        std_kappa=std_kappa,
        std_performance=std_performance,
        std_training_time=std_training_time,
        std_test_loss=std_test_loss,
        std_best_model_loss=std_best_model_loss,
        std_best_model_acc=std_best_model_acc,
        std_best_model_epoch=std_best_model_epoch,
        num_experiments=len(seeds),
    )
logger.flush()
logger.close()

################################# 计算并打印实验的总运行时间 ##################################
program_end_time = time.time()

total_seconds = program_end_time - program_start_time
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = total_seconds % 60

# 将所有数据集的总运行时间写入txt文件
with open(results_txt_path, "a", encoding="utf-8") as results_file:
    results_file.write(f"\n{'=' * 120}\n")
    results_file.write("所有数据集的总运行时间:\n")
    results_file.write(
        f"  {hours}小时 {minutes}分钟 {seconds:.2f}秒 (总计: {total_seconds:.2f}秒)\n"
    )
    results_file.write(f"{'=' * 120}\n")

print(f"\n所有数据集的结果已保存到: {results_txt_path}")
print(
    f"所有数据集的总运行时间: {hours}小时 {minutes}分钟 {seconds:.2f}秒 ({total_seconds:.2f}秒)"
)
