import torch
import torch.nn as nn
import numpy as np
from parameter import args

class hsi_e(nn.Module):
    def __init__(self):
        super(hsi_e, self).__init__()
        self.hsi_step1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(9, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=8))

        self.hsi_step2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(7, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=16)
        )
        self.hsi_step3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5, 3, 3), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(num_features=32)
        )
        self.hsi_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=176, out_channels=args.uni_dimension, kernel_size=7, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=args.uni_dimension),
        )

        self.hsi_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=args.uni_dimension, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=args.uni_dimension),
        )
        self.hsi_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=args.uni_dimension, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=args.uni_dimension),
        )

    def forward(self, x):
        X = []
        # x = x.unsqueeze(1)
        x1 = self.hsi_step1(x)
        X1 = self.hsi_conv1(x1.reshape(-1, 8 * 22, x1.shape[3], x1.shape[4]))
        X.append(X1 )
        x2 = self.hsi_step2(x1)
        X2 = x2.reshape(-1, 16 * 16, x2.shape[3], x2.shape[4])
        X2 = self.hsi_conv2(X2)
        X.append(X2)
        x3 = self.hsi_step3(x2)
        X3 = x3.reshape(-1, 32 * 12, x3.shape[3], x3.shape[4])
        X3 = self.hsi_conv3(X3)
        X.append(X3)
        X = torch.stack(X, dim=0)
        return X
