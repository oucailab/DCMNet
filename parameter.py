"""
parameter setting
"""
import torch
import argparse
import os
import torch
import data.Dataset as Dataset

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Houston2018','Houston2013','Trento'], default='Houston2018', help='dataset to use')
parser.add_argument('--lidar_channel', type=int, default=64)
parser.add_argument('--hsi_channel', type=int, default=64)
parser.add_argument('--hsi_windowSize', type=int, default=5)
parser.add_argument('--hsi_first_windowSize', type=int, default=13)
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--uni_dimension', type=int, default=64, help='feature dimension in cell output')
parser.add_argument('--embed_size', type=int, default=64, help='feature dimension in cell input')
parser.add_argument('--hid_router', type=int, default=256, help='Hidden size of MLP in routers')
parser.add_argument('--hid_size', type=int, default=64, help='Hidden size of FeedForward in Intra-Modal Reasoning Cell')
parser.add_argument('--size_SA', default=25, type=int, help='the size of spatial attention')
parser.add_argument('--num_classes', default=20, type=int, help='num_classes')
args = parser.parse_args(args=[])
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


trento_color_map = [
    [0, 43, 255],
    [78, 200, 237],
    [153, 205, 107],
    [254, 210, 13],
    [238, 52, 36],
    [125, 20, 22]
]

houston2013_color_map = [
    [0, 0, 131],
    [0,0,203],
    [0,19,255],
    [0,91,255],
    [0,167,255],
    [0, 239, 255],
    [55, 255, 199],
    [131,255,123],
    [203,255,51],
    [255,235,0],
    [255,163,0],
    [255,87,0],
    [255,15,0],
    [199,0,0],
    [127,0,0]
]
houston2018_color_map = [
    [50, 205, 51],
    [173, 255, 48],
    [0, 128, 129],
    [34, 139, 34],
    [46, 79, 78],
    [139, 69, 18],
    [0, 255, 255],
    [255, 255, 255],
    [211, 211, 211],
    [254, 0, 0],
    [169, 169, 169],
    [105, 105, 105],
    [139, 0, 1],
    [200, 100, 0],  #####
    [254, 165, 0],
    [255, 255, 0],
    [218, 165, 33],
    [255, 0, 254],
    [0, 0, 254],
    [63, 224, 208]
]

train_dataset = args.dataset  #Houston2018 Houston2013 Trento

Dataset.set_random_seed(0)
if train_dataset == "Houston2013":
    image_h = 349
    image_w = 1905
    train_loader, test_loader, trntst_loader, all_loader = Dataset.getHoustonData(
        hsi_path="data/Houston2013/houston_hsi.mat",
        lidar_path="data/Houston2013/houston_lidar.mat",
        gt_path="data/Houston2013/houston_gt.mat",
        index_path="data/Houston2013/houston_index.mat",
        channels=30,
        windowSize=11,
        batch_size=64,
        num_workers=0)
elif train_dataset == "Trento":
    image_h = 166
    image_w = 600
    train_loader, test_loader, trntst_loader, all_loader = Dataset.getTrentoData(
        hsi_path="data/Trento/trento_hsi.mat",
        lidar_path="data/Trento/trento_lidar.mat",
        gt_path="data/Trento/trento_gt.mat",
        index_path="data/Trento/trento_index.mat",
        channels=63,
        windowSize=11,
        batch_size=64,
        num_workers=0)
else:
    image_h = 1202
    image_w = 4768
    train_loader, test_loader, trntst_loader, all_loader = Dataset.getHouston2018Data(
        hsi_path="data/Houston2018/houston_hsi.mat",
        lidar_path="data/Houston2018/houston_lidar.mat",
        gt_path="data/Houston2018/houston_gt.mat",
        index_path="data/Houston2018/houston_index.mat",
        channels=20,
        windowSize=13,
        batch_size=64,
        num_workers=0)
print("completed!")
