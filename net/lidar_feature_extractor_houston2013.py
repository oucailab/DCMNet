import torch
import torch.nn as nn
import numpy as np
from parameter import args
class lidar_e(nn.Module):
    def __init__(self):
        super(lidar_e, self).__init__()
        self.conv1 = nn.Sequential(
            # torch.nn.Conv2d(kernel_size=3, in_channels=1, out_channels=8, padding=1),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=8),
            # torch.nn.Conv2d(kernel_size=3, in_channels=8, out_channels=16, padding=1),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=16),
            # torch.nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32, padding=0),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=32)
            torch.nn.Conv2d(kernel_size=3, in_channels=1, out_channels=64, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64)
            )
        self.conv2 = nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=64, out_channels=128, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128))
        self.conv3 = nn.Sequential(
            torch.nn.Conv2d(kernel_size=5, in_channels=128, out_channels=args.lidar_channel, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(args.lidar_channel)
        )
        self.out_conv1 = nn.Sequential(
            torch.nn.Conv2d(kernel_size=7, in_channels=64, out_channels=args.uni_dimension, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=args.uni_dimension))

        self.out_conv2 = nn.Sequential(
            torch.nn.Conv2d(kernel_size=5, in_channels=128, out_channels=args.uni_dimension, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=args.uni_dimension))
    def forward(self, x):
        X = []
        x1 = self.conv1(x)
        X.append(self.out_conv1(x1))
        x2 = self.conv2(x1)
        X.append(self.out_conv2(x2))
        x3 = self.conv3(x2)
        X.append(x3)
        X = torch.stack(X, dim=0)
        return X