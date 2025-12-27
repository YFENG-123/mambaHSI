import torch
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


    def add_model_graph(self, model, input_sample, data_name, seed_idx):
        """
        将模型架构添加到TensorBoard
        
        参数:
            model: PyTorch模型
            input_sample: 示例输入张量，形状为 (H, W, bands)
            data_name: 数据集名称
            seed_idx: 种子序号（从1开始）
        """
        try:
            # 将模型设置为评估模式以记录架构
            model.eval()
            with torch.no_grad():
                self.writer.add_graph(
                    model, 
                    input_sample,
                    verbose=False
                )
            print(f"    模型架构已添加到TensorBoard: {data_name}_Seed_{seed_idx}")
        except Exception as e:
            print(f"    警告: 添加模型架构到TensorBoard时出错: {e}")

    def flush(self):
        """刷新TensorBoard writer缓存"""
        self.writer.flush()

    def close(self):
        """关闭TensorBoard writer"""
        self.writer.close()

    def each_train(self, data_name, seed_idx, epoch, avg_train_loss, train_acc, train_time, current_lr):
        """
        记录每次训练的结果到TensorBoard
        
        参数:
            data_name: 数据集名称
            seed_idx: 种子序号（从1开始）
            epoch: 当前epoch
            avg_train_loss: 平均训练损失
            train_acc: 训练准确率
            train_time: 训练时间
            current_lr: 当前学习率
        """
        # 合并 Loss 和 Learning Rate 到一张图
        self.writer.add_scalars(
            f"Loss_LR_{self.timestamp}_{data_name}",
            {
                f"Train_Loss_{seed_idx}": avg_train_loss,
                f"LR_{seed_idx}": current_lr
            },
            epoch,
        )

        # Acc 单独一张图
        self.writer.add_scalars(
            f"Accuracy_{self.timestamp}_{data_name}",
            {f"Train_Acc_{seed_idx}": train_acc},
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
        # Loss 单独绘制 (或保留与 LR 相同的 Tag 以便对比，虽然 val 阶段没有 LR 变化)
        # 为了与 Train Loss 在同一图表中，我们使用相同的 Tag: Loss_LR_{...}
        self.writer.add_scalars(
            f"Loss_LR_{self.timestamp}_{data_name}",
            {
                f"Val_Loss_{seed_idx}": avg_val_loss,
            },
            epoch,
        )

        # Acc 单独一张图 (与 Train Acc 同一个图表 Tag)
        self.writer.add_scalars(
            f"Accuracy_{self.timestamp}_{data_name}",
            {f"Val_Acc_{seed_idx}": val_acc},
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
        # 合并 AA, Kappa, OA, Performance 到一张图
        self.writer.add_scalars(
            f"Test_Metrics_AKOP_{self.timestamp}",
            {
                f"AA_{data_name}": aa,
                f"Kappa_{data_name}": kappa,
                f"OA_{data_name}": oa,
                f"Performance_{data_name}": performance
            },
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
        # 合并 Best Acc, Best Epoch, Best Loss 到一张图
        self.writer.add_scalars(
            f"Best_Model_Metrics_{self.timestamp}",
            {
                f"Best_Loss_{data_name}": saved_loss,
                f"Best_Acc_{data_name}": saved_acc,
                f"Best_Epoch_{data_name}": saved_epoch
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
        # 将平均结果添加到原始值表格的最后一个点后面第5个点位置
        avg_position = len_seeds + 5
        
        # 合并 AA, Kappa, OA, Performance 到一张图 (Average)
        self.writer.add_scalars(
            f"Test_Metrics_AKOP_{self.timestamp}",
            {
                f"Average_AA_{data_name}": average_aa,
                f"Average_Kappa_{data_name}": average_kappa,
                f"Average_OA_{data_name}": average_oa,
                f"Average_Performance_{data_name}": average_performance
            },
            avg_position,
        )
        self.writer.add_scalars(
            f"Training_Time_{self.timestamp}",
            {f"Average_{data_name}": average_training_time},
            avg_position,
        )
