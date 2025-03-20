import torch
import torch.nn as nn
import torch.nn.functional as F
from parameter import args


class classify(nn.Module):
    def __init__(self):
        super(classify, self).__init__()

        self.conv1 = torch.nn.Conv2d(kernel_size=1, in_channels=args.uni_dimension, stride=1, out_channels=512)
        self.linear1 = nn.Linear(in_features=512, out_features=args.num_classes)

    # f:B uni_dimension hsi_windowSize hsi_windowSize
    def forward(self, x):
        x = self.conv1(x)
        x = F.avg_pool2d(x, kernel_size=args.hsi_windowSize).reshape(-1, 512)
        x = self.linear1(x)
        return x.squeeze(-1).squeeze(-1)
