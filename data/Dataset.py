import torch
import random
import numpy as np
import torch.nn as nn
from scipy.io import loadmat
from sklearn.decomposition import PCA
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset


def min_max(x):
    min = np.min(x)
    max = np.max(x)
    return (x - min) / (max - min)

# 设置随机数种子
def set_random_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# PCA 降维
# def applyPCA(data, n_components):
#
#     h, w, b = data.shape
#     pca = PCA(n_components=n_components)
#     data = np.reshape(pca.fit_transform(np.reshape(data, (-1, b))), (h, w, -1))
#     return data


def applyPCA(X, numComponents):
    """
    apply PCA to the image to reduce dimensionality
  """
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

# 创建 Dataset
class HLDataset(Dataset):

    def __init__(self, hsi, lidar, pos, windowSize, gt=None, transform=None):
        self.pad = (windowSize - 1) // 2
        self.windowSize = windowSize
        self.hsi = np.pad(hsi, ((self.pad, self.pad),
                          (self.pad, self.pad), (0, 0)), mode='reflect')

        self.lidar = np.pad(lidar, ((self.pad, self.pad),
                                    (self.pad, self.pad)), mode='reflect')

        self.pos = pos
        self.gt = None
        if gt is not None:
            self.gt = gt
        if transform:
            self.transform = transform

    def __getitem__(self, index):
        h, w = self.pos[index, :]
        hsi = self.hsi[h: h + self.windowSize, w: w + self.windowSize]
        lidar = self.lidar[h: h + self.windowSize, w: w + self.windowSize]
        if self.transform:
            hsi = self.transform(hsi).float()
            lidar = self.transform(lidar).float()
        if self.gt is not None:
            gt = torch.tensor(self.gt[h, w] - 1).long()
            # if gt == -1:
            #     print("warning!")
            return hsi.unsqueeze(0), lidar, gt
        return hsi.unsqueeze(0), lidar, h, w

    def __len__(self):
        return self.pos.shape[0]


# 根据 index 获取数据
def getData(hsi_path, lidar_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers):
    '''
    hsi_path: 高光谱数据集路径
    lidar_path: Lidar数据集路径
    gt_path: 真实标签数据集路径
    index_path: 索引数据集路径
    keys: mat 文件的 key
    channels: 降维后的通道数
    windowSize: 每张图片切割后的尺寸
    batch_size: 每个 batch 中的图片数量
    num_workers: 使用几个工作进程进行 Dataloader 的加载
    '''

    # 加载图片数据和坐标位置
    '''
    hsi: 高光谱图像数据
    lidar: Lidar 图像数据
    gt: 真实标签, 0 代表未标注
    train_index: 用于训练的数据索引
    test_index: 用于测试的数据索引
    trntst_index: 用于训练和测试的数据索引，用于对有标签的数据进行可视化
    all_index: 所有数据的索引，包含未标注数据，用于对所有数据进行可视化
    '''
    hsi = loadmat(hsi_path)[keys[0]]
    lidar = loadmat(lidar_path)[keys[1]]
    if hsi_path == "data/Houston2013/houston_hsi.mat":
        lidar = min_max(lidar)
    gt = loadmat(gt_path)[keys[2]]
    train_index = loadmat(index_path)[keys[3]]
    test_index = loadmat(index_path)[keys[4]]
    trntst_index = np.concatenate((train_index, test_index), axis=0)
    all_index = loadmat(index_path)[keys[5]]

    # 使用 PCA 对 HSI 进行降维
    hsi = applyPCA(hsi, channels)

    # 创建 Dataset, 用于生成对应的 Dataloader
    HLtrainset = HLDataset(hsi, lidar, train_index,
                           windowSize, gt, transform=ToTensor())
    HLtestset = HLDataset(hsi, lidar, test_index,
                          windowSize, gt, transform=ToTensor())
    HLtrntstset = HLDataset(hsi, lidar, trntst_index,
                            windowSize, transform=ToTensor())
    HLallset = HLDataset(hsi, lidar, all_index,
                         windowSize, transform=ToTensor())

    # 创建 Dataloader
    '''
    train_loader: 训练集
    test_loader: 测试集 
    trntst_loader: 用于画图，底色为白色，如 Trento 可视化图
    all_loader: 用于画图，底色为非白色，如 Houston 可视化图
    '''

    train_loader = DataLoader(
        HLtrainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(
        HLtestset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    trntst_loader = DataLoader(
        HLtrntstset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_loader = DataLoader(
        HLallset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Success!")
    return train_loader, test_loader, trntst_loader, all_loader


# 获取 Houston 数据集
def getHoustonData(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    '''
    hsi_path: 高光谱数据集路径
    lidar_path: Lidar数据集路径
    gt_path: 真实标签数据集路径
    index_path: 索引数据集路径
    channels: 降维后的通道数
    windowSize: 每张图片切割后的尺寸
    batch_size: 每个 batch 中的图片数量
    num_workers: 使用几个工作进程进行 Dataloader 的加载
    '''

    print("Houston!")

    # Houston mat keys
    keys = ['houston_hsi', 'houston_lidar', 'houston_gt',
            'houston_train', 'houston_test', 'houston_all']

    return getData(hsi_path, lidar_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers)


def getTrentoData(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    '''
    hsi_path: 高光谱数据集路径
    lidar_path: Lidar数据集路径
    gt_path: 真实标签数据集路径
    index_path: 索引数据集路径
    channels: 降维后的通道数
    windowSize: 每张图片切割后的尺寸
    batch_size: 每个 batch 中的图片数量
    num_workers: 使用几个工作进程进行 Dataloader 的加载
    '''

    print("Trento!")

    # Trento mat keys
    keys = ['trento_hsi', 'trento_lidar', 'trento_gt',
            'trento_train', 'trento_test', 'trento_all']

    return getData(hsi_path, lidar_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers)

def getHouston2018Data(hsi_path, lidar_path, gt_path, index_path, channels, windowSize, batch_size, num_workers):
    '''
    hsi_path: 高光谱数据集路径
    lidar_path: Lidar数据集路径
    gt_path: 真实标签数据集路径
    index_path: 索引数据集路径
    channels: 降维后的通道数
    windowSize: 每张图片切割后的尺寸
    batch_size: 每个 batch 中的图片数量
    num_workers: 使用几个工作进程进行 Dataloader 的加载
    '''

    print("Houston2018!")

    # Trento mat keys
    keys = ['houston_hsi', 'houston_lidar', 'houston_gt',
            'houston_train', 'houston_test', 'houston_all']

    return getData(hsi_path, lidar_path, gt_path, index_path, keys, channels, windowSize, batch_size, num_workers)
