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

    def each_train(self, data_name, seed_idx, epoch, avg_train_loss, train_acc, train_time):
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
        self.writer.add_scalars(  # TensorBoard 写入 Loss
            f"Loss_{self.timestamp}_{data_name}",
            {f"Train_{seed_idx}": avg_train_loss},
            epoch,
        )
        self.writer.add_scalars(  # TensorBoard 写入 Accuracy
            f"Accuracy_{self.timestamp}_{data_name}",
            {f"Train_{seed_idx}": train_acc},
            epoch,
        )
        self.writer.add_scalars(  # TensorBoard 写入训练时间
            f"Time_{self.timestamp}_{data_name}",
            {f"Train_{seed_idx}": train_time},
            epoch,
        )

    def each_val(self, data_name, seed_idx, epoch, avg_val_loss, val_acc, val_time):
        """
        记录每次验证的结果到TensorBoard
        
        参数:
            data_name: 数据集名称
            seed_idx: 种子序号（从1开始）
            epoch: 当前epoch
            avg_val_loss: 平均验证损失
            val_acc: 验证准确率
            val_time: 验证时间
        """
        self.writer.add_scalars(  # TensorBoard 写入 Loss
            f"Loss_{self.timestamp}_{data_name}",
            {f"Val_{seed_idx}": avg_val_loss},
            epoch,
        )
        self.writer.add_scalars(  # TensorBoard 写入 Accuracy
            f"Accuracy_{self.timestamp}_{data_name}",
            {f"Val_{seed_idx}": val_acc},
            epoch,
        )
        self.writer.add_scalars(  # TensorBoard 写入验证时间
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
        """
        # 写入测试结果指标
        self.writer.add_scalars(
            f"OA_{self.timestamp}",
            {data_name: oa},
            seed_idx,  # x轴：种子序号
        )
        self.writer.add_scalars(
            f"AA_{self.timestamp}",
            {data_name: aa},
            seed_idx,  # x轴：种子序号
        )
        self.writer.add_scalars(
            f"Kappa_{self.timestamp}",
            {data_name: kappa},
            seed_idx,  # x轴：种子序号
        )
        self.writer.add_scalars(
            f"Performance_{self.timestamp}",
            {data_name: performance},
            seed_idx,  # x轴：种子序号
        )
        self.writer.add_scalars(
            f"Training_Time_{self.timestamp}",
            {data_name: total_training_time},
            seed_idx,  # x轴：种子序号
        )

        # 写入每个类别的精度到TensorBoard（合并为一个表格）
        # x轴：类别序号，y轴：准确率，每条曲线代表一个种子
        for i, acc in enumerate(class_accuracies):
            self.writer.add_scalars(
                f"Class_Accuracy_{self.timestamp}_{data_name}",
                {f"Seed_{seed_idx}": acc},
                i + 1,  # x轴：class序号
            )

    def each_seed(
        self,
        data_name,
        seed_idx,
        saved_loss,
        saved_acc,
        saved_epoch,
        saved_type,
    ):
        """
        记录每个种子的最佳模型信息到TensorBoard
        
        参数:
            data_name: 数据集名称
            seed_idx: 种子序号（从1开始）
            saved_loss: 最佳模型的损失
            saved_acc: 最佳模型的准确率
            saved_epoch: 最佳模型产生的epoch
            saved_type: 最佳模型类型（"Train"或"Val"）
        """
        # 写入最佳模型信息
        self.writer.add_scalars(
            f"Best_Model_Loss_{self.timestamp}",
            {data_name: saved_loss},
            seed_idx,  # x轴：种子序号
        )
        self.writer.add_scalars(
            f"Best_Model_Accuracy_{self.timestamp}",
            {data_name: saved_acc},
            seed_idx,  # x轴：种子序号
        )
        self.writer.add_scalars(
            f"Best_Model_Epoch_{self.timestamp}",
            {data_name: saved_epoch},
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
        # 将平均结果添加到原始值表格的最后一个点后面第5个点位置
        avg_position = len_seeds + 5
        
        self.writer.add_scalars(
            f"OA_{self.timestamp}",
            {f"{data_name}_Average": average_oa},
            avg_position,
        )
        self.writer.add_scalars(
            f"AA_{self.timestamp}",
            {f"{data_name}_Average": average_aa},
            avg_position,
        )
        self.writer.add_scalars(
            f"Kappa_{self.timestamp}",
            {f"{data_name}_Average": average_kappa},
            avg_position,
        )
        self.writer.add_scalars(
            f"Performance_{self.timestamp}",
            {f"{data_name}_Average": average_performance},
            avg_position,
        )
        self.writer.add_scalars(
            f"Training_Time_{self.timestamp}",
            {f"{data_name}_Average": average_training_time},
            avg_position,
        )
