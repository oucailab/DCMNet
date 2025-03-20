import torch
import torch.nn as nn

from net.Cells import Cell_1_0, Cell_2_0, Cell_3, Cell_1, Cell_2,Cell_3_0


def unsqueeze2d(x):
    return x.unsqueeze(-1).unsqueeze(-1)


def unsqueeze3d(x):
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)





class Feature_Interaction_Layer0(nn.Module):
    def __init__(self, args, num_cell, num_out_path):
        super(Feature_Interaction_Layer0, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path
        self.c1 = Cell_1_0(num_out_path)
        self.c2 = Cell_2_0(num_out_path)
        self.c3 = Cell_3_0(num_out_path)


    def forward(self, hsi, lidar):
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.c1(lidar, hsi)
        emb_lst[1], path_prob[1] = self.c2(lidar, hsi) 
        emb_lst[2], path_prob[2] = self.c3(lidar, hsi)



        all_path_prob = torch.stack(path_prob, dim=2)
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

        aggr_res_lst = []
        for i in range(self.num_out_path):
            res = 0
            for j in range(self.num_cell):
                cur_path = unsqueeze3d(path_prob[j][:, i])
                cur_emb = emb_lst[j]
                res = res + cur_path * cur_emb
            aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob


class Feature_Interaction_Layern(nn.Module):
    def __init__(self, args, num_cell, num_out_path, stage=1):
        super(Feature_Interaction_Layern, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path
        self.stage = stage
        self.c11 = Cell_1(num_out_path)
        self.c12 = Cell_2(num_out_path)
        self.c13 = Cell_3(num_out_path)
        self.c21 = Cell_1(num_out_path)
        self.c22 = Cell_2(num_out_path)
        self.c23 = Cell_3(num_out_path)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=args.uni_dimension * 2, out_channels=args.uni_dimension, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=args.uni_dimension))

    def forward(self, ref_rgn, hsi, lidar):

        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell

        if self.stage == 1:
            emb_lst[0], path_prob[0] = self.c11(lidar, hsi, ref_rgn[0])
            emb_lst[1], path_prob[1] = self.c12(lidar, hsi, ref_rgn[1])
            emb_lst[2], path_prob[2] = self.c13(lidar, hsi, ref_rgn[2])
        elif self.stage == 2:
            emb_lst[0], path_prob[0] = self.c21(lidar, hsi, ref_rgn[0])
            emb_lst[1], path_prob[1] = self.c22(lidar, hsi, ref_rgn[1])
            emb_lst[2], path_prob[2] = self.c23(lidar, hsi, ref_rgn[2])
        all_path_prob = torch.stack(path_prob, dim=2)

        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]
        aggr_res_lst = []
        for i in range(self.num_out_path):
            res = 0
            for j in range(self.num_cell):
                cur_path = unsqueeze3d(path_prob[j][:, i])
                res = res + cur_path * emb_lst[j]
            aggr_res_lst.append(res)
        return aggr_res_lst, all_path_prob
