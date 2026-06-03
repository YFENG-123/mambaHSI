from torch.utils.tensorboard import SummaryWriter


class Log:
    """
    TensorBoard日志记录类
    用于统一管理所有训练、验证、测试和最佳模型的日志记录
    """

    def __init__(self, log_base_dir, timestamp):
        """
        初始化Log类

        参数:
            log_base_dir: TensorBoard日志目录路径（已在main.py中创建）
            timestamp: 时间戳字符串
        """
        self.timestamp = timestamp
        self.writer = SummaryWriter(log_dir=log_base_dir)

    def flush(self):
        """刷新TensorBoard writer缓存"""
        self.writer.flush()

    def close(self):
        """关闭TensorBoard writer"""
        self.writer.close()

    def log_model_graph(self, model, sample_input, verbose=False):
        """
        将模型结构写入 TensorBoard（尽可能兼容不同 PyTorch 版本的 add_graph 调用）。

        参数:
            model: nn.Module 实例
            sample_input: 传入模型的示例输入，通常是一个 Tensor 或者 tuple/list of Tensors
            verbose: 是否使用 verbose 模式（传递给 add_graph）
        """
        # 直接记录模型结构到 TensorBoard（保持原始简洁风格）
        self.writer.add_graph(model, (sample_input,), verbose=verbose)

    def each_train(
        self,
        data_name,
        seed_idx,
        epoch,
        avg_train_loss,
        train_acc,
        train_time,
        current_lr,
        allocated_memory,
        cached_memory,
    ):
        """
        记录每次训练的结果到TensorBoard

        参数:
            data_name: 数据集名称
            seed_idx: 种子序号（从1开始）
            epoch: 当前epoch
            avg_train_loss: 平均训练损失
            train_acc: 训练准确率
            train_time: 训练时间
        """
        # 使用 log copy 的风格：Loss 与 LR 合并到一张图，Accuracy、Time 各自一张图，tag 包含 timestamp 与 data_name
        self.writer.add_scalars(
            f"Loss_LR_{self.timestamp}_{data_name}",
            {
                f"Train_Loss_{seed_idx}": avg_train_loss,
                **({f"LR_{seed_idx}": current_lr} if current_lr is not None else {}),
            },
            epoch,
        )

        self.writer.add_scalars(
            f"Accuracy_{self.timestamp}_{data_name}",
            {f"Train_Acc_{seed_idx}": train_acc},
            epoch,
        )

        self.writer.add_scalars(
            f"Time_{self.timestamp}_{data_name}",
            {f"Train_{seed_idx}": train_time},
            epoch,
        )

        self.writer.add_scalars(
            f"GPU_Memory_{self.timestamp}_{data_name}",
            {f"Train_Alloc_MB_{seed_idx}": allocated_memory, f"Train_Cached_MB_{seed_idx}": cached_memory},
            epoch,
        )

    def each_val(self, data_name, seed_idx, epoch, avg_val_loss, val_acc, val_time, allocated_memory, cached_memory):
        """
        记录每次验证的结果到TensorBoard

        参数:
            data_name: 数据集名称
            seed_idx: 种子序号（从1开始）
            epoch: 当前epoch
            avg_val_loss: 平均验证损失
            val_acc: 验证准确率
            val_time: 验证时间
            allocated_memory: 已分配显存
            cached_memory: 缓存显存
        """
        # 与 train 保持一致的 tag 风格，便于在 TensorBoard 中对比 Train/Val
        self.writer.add_scalars(
            f"Loss_LR_{self.timestamp}_{data_name}",
            {f"Val_Loss_{seed_idx}": avg_val_loss},
            epoch,
        )

        self.writer.add_scalars(
            f"Accuracy_{self.timestamp}_{data_name}",
            {f"Val_Acc_{seed_idx}": val_acc},
            epoch,
        )
        self.writer.add_scalars(
            f"GPU_Memory_{self.timestamp}_{data_name}",
            {f"Val_Alloc_MB_{seed_idx}": allocated_memory, f"Val_Cached_MB_{seed_idx}": cached_memory},
            epoch,
        )
        self.writer.add_scalars(
            f"Time_{self.timestamp}_{data_name}",
            {f"Val_{seed_idx}": val_time},
            epoch,
        )

    def each_test(
        self,
        data_name,
        seed_idx,
        oa,
        aa,
        kappa,
        performance,
        total_training_time,
        class_accuracies,
        allocated_memory,
        cached_memory,
    ):
        """
        记录每次测试的结果到TensorBoard

        参数:
            data_name: 数据集名称
            seed_idx: 种子序号（从1开始）
            oa: Overall Accuracy
            aa: Average Accuracy
            kappa: Kappa系数
            performance: 性能指标 (oa + aa + kappa) / 3.0
            total_training_time: 总训练时间
            class_accuracies: 各类别精度列表
            allocated_memory: 已分配显存
            cached_memory: 缓存显存
        """
        # 使用 log copy 的风格：将 OA/AA/Kappa/Performance 合并到一张图（便于对比），并记录训练时间单独图
        self.writer.add_scalars(
            f"Test_Metrics_AKOP_{self.timestamp}",
            {
                f"AA_{data_name}": aa,
                f"Kappa_{data_name}": kappa,
                f"OA_{data_name}": oa,
                f"Performance_{data_name}": performance,
            },
            seed_idx,  # x轴：种子序号
        )
        self.writer.add_scalars(
            f"Training_Time_{self.timestamp}",
            {data_name: total_training_time},
            seed_idx,  # x轴：种子序号
        )
        self.writer.add_scalars(
            f"GPU_Memory_{self.timestamp}_{data_name}",
            {f"Test_Alloc_MB_{seed_idx}": allocated_memory, f"Test_Cached_MB_{seed_idx}": cached_memory},
            seed_idx,
        )
        # 每个类别的精度，tag 包含 data_name，series 为 Seed_{seed_idx}
        for i, acc in enumerate(class_accuracies):
            self.writer.add_scalars(
                f"Class_Accuracy_{self.timestamp}_{data_name}",
                {f"Seed_{seed_idx}": acc},
                i + 1,  # x轴：class序号
            )

    def each_seed(self, data_name, seed_idx, saved_loss, saved_acc, saved_epoch):
        """
        记录每个种子的最佳模型信息到TensorBoard

        参数:
            data_name: 数据集名称
            seed_idx: 种子序号（从1开始）
            saved_loss: 最佳模型的损失
            saved_acc: 最佳模型的准确率
            saved_epoch: 最佳模型产生的epoch
        """
        # 合并 Best Loss/Acc/Epoch 到一张图，series 使用 data_name 便于按数据集查看
        self.writer.add_scalars(
            f"Best_Model_Metrics_{self.timestamp}",
            {
                f"Best_Loss_{data_name}": saved_loss,
                f"Best_Acc_{data_name}": saved_acc,
                f"Best_Epoch_{data_name}": saved_epoch,
            },
            seed_idx,  # x轴：种子序号
        )

    def each_dataset_average(
        self,
        data_name,
        len_seeds,
        average_oa,
        average_aa,
        average_kappa,
        average_performance,
        average_training_time,
    ):
        """
        记录每个数据集所有种子的平均结果到TensorBoard

        参数:
            data_name: 数据集名称
            len_seeds: 种子数量（用于计算avg_position = len_seeds + 5）
            average_oa: 平均OA
            average_aa: 平均AA
            average_kappa: 平均Kappa
            average_performance: 平均性能指标
            average_training_time: 平均训练时间
        """
        # 将平均结果写入：与 each_test 保持同样的 AKOP 合并图，便于对比
        avg_position = len_seeds + 5

        self.writer.add_scalars(
            f"Test_Metrics_AKOP_{self.timestamp}",
            {
                f"Average_AA_{data_name}": average_aa,
                f"Average_Kappa_{data_name}": average_kappa,
                f"Average_OA_{data_name}": average_oa,
                f"Average_Performance_{data_name}": average_performance,
            },
            avg_position,
        )
        self.writer.add_scalars(
            f"Training_Time_{self.timestamp}",
            {f"Average_{data_name}": average_training_time},
            avg_position,
        )