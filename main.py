import os
import time
import torch
from models import MambaHSINet
from util import (
    load_data,
    calculate_seed_result,
    calculate_dataset_result,
    run_model,
    generate_picture,
    set_seed,
    write_results_to_txt,
    init_results_file_header,
    create_criterion,
    create_optimizer,
    create_scheduler,
    print_experiment_config,
)
from log import Log

################################# 记录实验开始时间 ##################################
program_start_time = time.time()

################################# 设置超参数 #################################
num_epochs = 1000  # 训练轮数
learning_rate = 1e-3  # 适配 Pre-Norm Residual 结构
dropout_rate = 0.2
# 仅在最后 50 个 epoch 开始保存模型；在倒数第50轮（即开始的那一轮）固定保存一次快照
save_start_epoch = max(1, num_epochs - 50 + 1)
################################# 优化器参数 ##################################
optimizer_type = "Adam"  # 优化器类型: "SGD", "Adam", "AdamW", "RMSprop", "Adagrad"
# SGD 参数（随机梯度下降）
momentum = 0.9  # 动量因子
weight_decay = 1e-3  # 权重衰减（L2正则化）
nesterov = False  # 是否使用Nesterov动量
# Adam/AdamW 参数
adam_beta1 = 0.9  # Adam的beta1参数
adam_beta2 = 0.999  # Adam的beta2参数
adam_eps = 1e-8  # Adam的epsilon参数

################################# 损失函数参数 ##################################
# 损失函数类型选择，支持以下选项：
# - "CrossEntropy": 标准交叉熵损失（会根据类别权重自动选择是否使用加权版本）
# - "Focal": Focal Loss，用于处理类别不平衡问题
# - "WeightedCrossEntropy": 强制使用加权交叉熵损失（需要手动提供类别权重）
loss_type = "WeightedCrossEntropy"  

# Focal Loss 参数（仅当 loss_type == "Focal" 时生效）
focal_alpha = 1.0  # Focal Loss的alpha参数：平衡因子，用于调整正负样本权重
focal_gamma = 2.0  # Focal Loss的gamma参数：聚焦参数，用于降低易分类样本权重

################################# 学习率调度器参数 ##################################
# 将是否使用调度器合并到 scheduler_type 中，指定 "None" 表示不使用调度器
scheduler_type = "None"  # 调度器类型: "None", "StepLR", "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "ExponentialLR", "MultiStepLR"
scheduler_patience = 50  # 验证损失不下降的等待轮数（降低patience，更早调整学习率）
scheduler_factor = 0.5  # 学习率衰减因子（更激进的衰减）
scheduler_min_lr = 1e-6  # 最小学习率（更小的最小学习率，允许更精细的调整）
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

################################# 数据集和种子 ##################################
seeds = [
    21,
    22,
    80,
    443,
    445,
    #554,
    #3306,
    #5900,
    #8080,
    #25565,
]
image_paths = [
    # "data/HuaiLai.mat",
    # "data/Botswana.mat",
    # "data/Indian_pines.mat",
    # "data/KSC.mat",
    "data/Pavia.mat",
    # "data/PaviaU.mat",
    # "data/Salinas.mat",
    # "data/SalinasA.mat",
]
gt_paths = [
    # "data/HuaiLai_gt.mat",
    # "data/Botswana_gt.mat",
    # "data/Indian_pines_gt.mat",
    # "data/KSC_gt.mat",
    "data/Pavia_gt.mat",
    # "data/PaviaU_gt.mat",
    # "data/Salinas_gt.mat",
    # "data/SalinasA_gt.mat",
]
val_split_rate = 0.01  # 验证集比例固定严禁修改！！！
test_split_rate = 0.98  # 测试集比例固定严禁修改！！！

################################# 打印超参数 ##################################
print_experiment_config(
    num_epochs,
    learning_rate,
    dropout_rate,
    optimizer_type,
    momentum,
    weight_decay,
    nesterov,
    adam_beta1,
    adam_beta2,
    adam_eps,
    val_split_rate,
    test_split_rate,
    loss_type,
    focal_alpha,
    focal_gamma,
    seeds,
    scheduler_type,
    scheduler_patience,
    scheduler_factor,
    scheduler_min_lr,
    step_size,
    gamma,
    T_max,
    T_0,
    T_mult,
    exp_gamma,
    milestones,
    image_paths=image_paths,
    gt_paths=gt_paths,
)


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
init_results_file_header(
    results_txt_path,
    timestamp,
    num_epochs,
    learning_rate,
    dropout_rate,
    optimizer_type,
    momentum,
    weight_decay,
    nesterov,
    adam_beta1,
    adam_beta2,
    adam_eps,
    val_split_rate,
    test_split_rate,
    loss_type,
    focal_alpha,
    focal_gamma,
    seeds,
    scheduler_type,
    scheduler_patience,
    scheduler_factor,
    scheduler_min_lr,
    step_size,
    gamma,
    T_max,
    T_0,
    T_mult,
    exp_gamma,
    milestones,
)
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
            class_weights,
        ) = load_data(
            image_path=image_path,
            gt_path=gt_path,
            val_split_rate=val_split_rate,
            test_split_rate=test_split_rate,
        )
        ############################### 每个种子重新随机初始化模型参数 ##################################
        model = MambaHSINet(
            image_x=image_x,
            image_y=image_y,
            num_classes=num_classes,
            bands=bands,
            dropout_rate=dropout_rate,
        ).to("cuda")
        sample_input = image.squeeze(0).to("cuda")
        logger.log_model_graph(model, sample_input)
        print("已将网络结构写入 TensorBoard")
        ################################# 创建损失函数 ##################################
        criterion = create_criterion(loss_type, focal_alpha, focal_gamma, class_weights)
        ################################# 创建优化器 ##################################
        optimizer = create_optimizer(
            model.parameters(),
            optimizer_type,
            learning_rate,
            momentum,
            weight_decay,
            nesterov,
            adam_beta1,
            adam_beta2,
            adam_eps,
        )
        ################################# 创建学习率调度器（使用PyTorch自带的lr_scheduler）##################################
        scheduler = create_scheduler(
            scheduler_type,
            optimizer,
            val_split_rate,
            scheduler_patience,
            scheduler_factor,
            scheduler_min_lr,
            step_size,
            gamma,
            T_max,
            T_0,
            T_mult,
            exp_gamma,
            milestones,
        )
        ################################# 开始一个种子的训练和验证 ##################################
        # 使用统一变量跟踪最佳模型（训练/验证共用），简化逻辑并提高可读性
        best_loss = float("inf")
        best_acc = 0.0
        best_epoch = 0
        best_type = None
        avg_val_loss = float("inf")
        val_acc = 0.0
        start_time = time.time()  # 记录每个种子开始时间
        for epoch in range(num_epochs):
            ############################### 该轮训练和验证开始 ##################################
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
                current_lr,
            ) = run_model(model, train_loader, criterion, optimizer, "train", scheduler)
            logger.each_train(  # 记录每次训练的结果到TensorBoard
                data_name,
                seed_idx + 1,
                epoch + 1,
                avg_train_loss,
                train_acc,
                train_time,
                current_lr,
            )
            print(  # 打印训练集结果
                f"    Train -> Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.2f}%, Time: {train_time:.2f}s"
            )
            # 当前学习率在 run_model 中已更新（如提供 scheduler），并作为 current_lr 返回
            print(f"    当前学习率: {current_lr:.6f}")
            ############################### 如果没有验证集，每个epoch的训练阶段结束后保存最佳模型 ##################################
            if val_split_rate <= 0:
                current_epoch = epoch + 1
                # 仅在最后 save_start_epoch 之后才开始保存最佳模型
                if current_epoch >= save_start_epoch:
                    # 在倒数第50轮（起始轮）固定保存一次快照
                    if current_epoch == save_start_epoch:
                        fixed_save_path = os.path.join(
                            weights_dir,
                            f"{seed_idx + 1}_fixed_epoch{current_epoch}.pth",
                        )
                        torch.save(model.state_dict(), fixed_save_path)
                        print(f"    固定快照已保存为: {fixed_save_path}")

                    if avg_train_loss < best_loss:
                        best_loss = avg_train_loss
                        best_acc = train_acc
                        best_epoch = current_epoch
                        best_type = "Train"
                        save_path = os.path.join(weights_dir, f"{seed_idx + 1}.pth")
                        torch.save(model.state_dict(), save_path)
                        print(f"    模型已在训练阶段结束后保存为: {save_path}")
                continue
            ############################### 如果有验证集，每个epoch的验证阶段 ##################################
            (
                avg_val_loss,
                val_acc,
                all_predictions,
                all_label_masked,
                full_prediction_map,
                full_test_label,
                val_time,
                current_lr,
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
            ############################### 每个epoch保存当前最佳模型 ##################################
            current_epoch = epoch + 1
            # 仅在最后 save_start_epoch 之后才进行最佳模型保存
            if current_epoch >= save_start_epoch:
                # 在倒数第50轮（起始轮）固定保存一次快照
                if current_epoch == save_start_epoch:
                    fixed_save_path = os.path.join(
                        weights_dir, f"{seed_idx + 1}_fixed_epoch{current_epoch}.pth"
                    )
                    torch.save(model.state_dict(), fixed_save_path)
                    print(f"    固定快照已保存为: {fixed_save_path}")

                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    best_acc = val_acc
                    best_epoch = current_epoch
                    best_type = "Val"
                    torch.save(
                        model.state_dict(),
                        os.path.join(weights_dir, f"{seed_idx + 1}.pth"),
                    )
                    print(
                        f"Best model saved at Epoch {best_epoch} with Loss: {best_loss:.4f} Accuracy: {best_acc:.2f}%"
                    )
            ############################### 该轮训练和验证结束 ##################################
        end_time = time.time()  # 记录每个种子结束时间
        total_training_time = end_time - start_time  # 计算每个种子训练时间
        print(f"Training time: {total_training_time:.2f} seconds")
        ################################# 记录一个种子的最佳模型信息 ##################################
        # 确定保存模型时的loss和acc（使用统一变量）
        saved_loss = best_loss
        saved_acc = best_acc
        saved_epoch = best_epoch
        logger.each_seed(  # 记录一个种子的最佳模型信息到TensorBoard
            data_name,
            seed_idx + 1,
            saved_loss,
            saved_acc,
            saved_epoch,
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
            current_lr,
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
