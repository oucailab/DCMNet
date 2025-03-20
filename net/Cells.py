import torch
import torch.nn as nn

from einops import rearrange, repeat

from .Router import Router

from parameter import args

import torch.nn.functional as F




class Cell_1_0(nn.Module):
    def __init__(self, num_out_path):
        super(Cell_1_0, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size * 2, args.hid_router)
        self.sa = Cross_Self_attention(args.size_SA, args.size_SA)
        self.conv = torch.nn.Conv2d(kernel_size=1, in_channels=args.embed_size * 2, out_channels=args.embed_size,
                                    stride=1)

    def forward(self, lidar,hsi):
        path_prob = self.router(torch.concat((lidar,hsi),1))
        l = rearrange(lidar, 'b h n d -> b h (n d)')
        h = rearrange(hsi, 'b h n d -> b h (n d)')
        h_emb = self.sa(l,h)
        l_emb = self.sa(h, l)
        h_emb = rearrange(h_emb, 'b h (n d) -> b h n d', d=args.hsi_windowSize)
        l_emb = rearrange(l_emb, 'b h (n d) -> b h n d', d=args.hsi_windowSize)
        emb = self.conv(torch.concat((l_emb,h_emb),1))
        return emb, path_prob

class Cell_1(nn.Module):
    def __init__(self, num_out_path):
        super(Cell_1, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size * 2, args.hid_router)
        self.sa = Cross_Self_attention(args.size_SA, args.size_SA)

        self.conv1 = torch.nn.Conv2d(kernel_size=1, in_channels=args.embed_size * 2, out_channels=args.embed_size,
                                    stride=1)


    def forward(self, x,lidar,hsi):
        path_prob = self.router(torch.concat((lidar,hsi),1))
        l = rearrange(lidar, 'b h n d -> b h (n d)')
        h = rearrange(hsi, 'b h n d -> b h (n d)')
        h_emb = self.sa(l, h)
        l_emb = self.sa(h, l)
        h_emb = rearrange(h_emb, 'b h (n d) -> b h n d', d=args.hsi_windowSize)
        l_emb = rearrange(l_emb, 'b h (n d) -> b h n d', d=args.hsi_windowSize)
        emb = self.conv1(torch.concat((l_emb,h_emb),1)) + x
        return emb, path_prob



class Cell_2_0(nn.Module):
    def __init__(self, num_out_path):
        super(Cell_2_0, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size * 2, args.hid_router)
        self.sa = Cross_Self_attention(args.embed_size, args.embed_size)

        self.conv = torch.nn.Conv2d(kernel_size=1, in_channels=args.embed_size * 2, out_channels=args.embed_size,
                                    stride=1)

    def forward(self, lidar, hsi):
        path_prob = self.router(torch.concat((lidar, hsi), 1))
        l = rearrange(lidar, 'b h n d -> b (n d) h')
        h = rearrange(hsi, 'b h n d -> b (n d) h')
        h_emb = self.sa(l, h)
        l_emb = self.sa(h, l)
        h_emb = rearrange(h_emb, 'b (n d) h -> b h n d', d=args.hsi_windowSize)
        l_emb = rearrange(l_emb, 'b (n d) h -> b h n d', d=args.hsi_windowSize)
        emb = self.conv(torch.concat((l_emb, h_emb), 1))
        return emb, path_prob


class Cell_2(nn.Module):
    def __init__(self, num_out_path):
        super(Cell_2, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size * 2, args.hid_router)
        self.sa = Cross_Self_attention(args.embed_size, args.embed_size)
        self.conv1 = torch.nn.Conv2d(kernel_size=1, in_channels=args.embed_size * 2, out_channels=args.embed_size,
                                    stride=1)


    def forward(self, lidar, hsi,x):
        path_prob = self.router(torch.concat((lidar, hsi), 1))
        l = rearrange(lidar, 'b c h w -> b (h w) c')
        h = rearrange(hsi, 'b c h w -> b (h w) c')
        h_emb = self.sa(l, h)
        l_emb = self.sa(h, l)
        h_emb = rearrange(h_emb, 'b (h w) c -> b c h w', h=args.hsi_windowSize)
        l_emb = rearrange(l_emb, 'b (h w) c -> b c h w', h=args.hsi_windowSize)
        emb = self.conv1(torch.concat((l_emb, h_emb), 1)) + x
        return emb, path_prob

class Cell_3(nn.Module):
    def __init__(self, num_out_path):
        super(Cell_3, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size * 3, args.hid_router)
        self.conv = nn.Sequential(
            torch.nn.Conv2d(kernel_size=1, in_channels=args.embed_size * 2, out_channels=args.embed_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=args.embed_size)
        )

    def forward(self, lidar,hsi,  pre):
        path_prob = self.router(torch.concat((hsi, lidar, pre), 1))
        sa_emb = self.conv(torch.concat((hsi, lidar), 1)) + pre
        return sa_emb, path_prob

class Cell_3_0(nn.Module):
    def __init__(self, num_out_path):
        super(Cell_3_0, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size * 2, args.hid_router)
        self.conv = nn.Sequential(
            torch.nn.Conv2d(kernel_size=1, in_channels=args.embed_size * 2, out_channels=args.embed_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=args.embed_size)
        )

    def forward(self, lidar,hsi):
        path_prob = self.router(torch.concat((hsi, lidar), 1))
        sa_emb = self.conv(torch.concat((hsi, lidar), 1))
        return sa_emb, path_prob

class Self_attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Self_attention, self).__init__()

        self.key = nn.Linear(input_size, hidden_size)
        self.query = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.dense = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        query_layer = self.query(x)
        key_layer = self.key(x)
        value_layer = self.value(x)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_porbs = nn.Softmax(dim=-1)(attention_scores)
        context = torch.matmul(attention_porbs, value_layer)
        transform = self.dense(context)
        transform = self.dropout(transform)
        return transform


class Cross_Self_attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Cross_Self_attention, self).__init__()
        self.key = nn.Linear(input_size, hidden_size)
        self.query = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.dense = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x,y):
        query_layer = self.query(x)
        key_layer = self.key(x)
        value_layer = self.value(y)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_porbs = nn.Softmax(dim=-1)(attention_scores)
        bilinear_v = query_layer * value_layer
        context = torch.matmul(attention_porbs, bilinear_v)
        transform = self.dropout(context)
        return transform

