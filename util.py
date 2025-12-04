import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import random


def load_data(
    image_path,
    gt_path,
    val_split_rate=0.1,
    test_split_rate=0.1,
    dataset_splited_mode="None",
):
    if dataset_splited_mode == "half":
        splite_rate_train = 0.9

        image_file = "./data/disjoin/Indian_pines.mat"  # (145,145,200)
        train_val_file = "./data/disjoin/IP_training.mat"  # (145,145)
        test_file = "./data/disjoin/IP_testing.mat"  # (145,145)

        """
        加载图像
        """
        image_file_dict = sio.loadmat(image_file)
        image = torch.from_numpy(image_file_dict["indian_pines"]).float()
        image = image.unsqueeze(0)

        """
        加载训练集和验证集标签，分别生成训练集和验证集掩码，并分别生训练数据集和验证数据集加载器
        """
        train_val_dict = sio.loadmat(train_val_file)
        train_val_label = torch.from_numpy(train_val_dict["IP_training"]).long()

        max_label = int(torch.max(train_val_label))
        train_val_label_flatten = train_val_label.flatten(start_dim=0, end_dim=1)
        train_val_label_index_list = [
            torch.where(train_val_label_flatten == i)[0].tolist()
            for i in range(max_label + 1)
        ]
        train_label_index_list = []
        val_label_index_list = []
        for i in range(1, max_label + 1):
            random.shuffle(train_val_label_index_list[i])
            splite_index = int(len(train_val_label_index_list[i]) * splite_rate_train)
            train_label_index_list.extend(train_val_label_index_list[i][:splite_index])
            val_label_index_list.extend(train_val_label_index_list[i][splite_index:])

        train_mask = np.zeros(train_val_label_flatten.shape)
        train_mask[train_label_index_list] = 1
        train_mask = train_mask.reshape(train_val_label.shape)
        train_mask = torch.from_numpy(train_mask)
        train_mask = train_mask.bool()
        train_mask = train_mask.unsqueeze(0)
        train_label = train_val_label.unsqueeze(0)
        train_dataset = TensorDataset(image, train_label, train_mask)
        train_loader = DataLoader(dataset=train_dataset, batch_size=1)

        val_mask = np.zeros(train_val_label_flatten.shape)
        val_mask[val_label_index_list] = 1
        val_mask = val_mask.reshape(train_val_label.shape)
        val_mask = torch.from_numpy(val_mask)
        val_mask = val_mask.bool()
        val_mask = val_mask.unsqueeze(0)
        val_label = train_val_label.unsqueeze(0)
        val_dataset = TensorDataset(image, val_label, val_mask)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)

        """
        加载测试集标签，生成测试集掩码，并生成测试数据集加载器
        """
        test_file_dict = sio.loadmat(test_file)
        test_label = torch.from_numpy(test_file_dict["IP_testing"]).long()
        test_mask = test_label != 0
        test_mask = test_label.bool()
        test_mask = test_mask.unsqueeze(0)
        test_label = test_label.unsqueeze(0)
        test_dataset = TensorDataset(image, test_label, test_mask)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    elif dataset_splited_mode == "None":
        # 加载image
        image_dict = sio.loadmat(image_path)
        image_np = next(  # 图像是三维矩阵:  (H, W, C)，找第一个ndim为3的 numpy.ndarray 作为image
            m for m in image_dict.values() if isinstance(m, np.ndarray) and m.ndim == 3
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
        train_loader = DataLoader(dataset=train_dataset, batch_size=1)

        val_mask = np.zeros(gt_flatten.shape)
        val_mask[val_label_index_list] = 1
        val_mask = val_mask.reshape(gt.shape)
        val_mask = torch.from_numpy(val_mask)
        val_mask = val_mask.bool()
        val_mask = val_mask.unsqueeze(0)
        val_label = gt.unsqueeze(0)
        val_dataset = TensorDataset(image, val_label, val_mask)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)

        test_mask = np.zeros(gt_flatten.shape)
        test_mask[test_label_index_list] = 1
        test_mask = test_mask.reshape(gt.shape)
        test_mask = torch.from_numpy(test_mask)
        test_mask = test_mask.bool()
        test_mask = test_mask.unsqueeze(0)
        test_label = gt.unsqueeze(0)
        test_dataset = TensorDataset(image, test_label, test_mask)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)

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
    )
