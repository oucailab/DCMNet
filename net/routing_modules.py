import torch
import torch.nn as nn

from net.Feature_Interaction import Feature_Interaction_Layer0, Feature_Interaction_Layern

class RoutingModule(nn.Module):
    def __init__(self, args, num_layer_routing=5, path_hid=128):
        super(RoutingModule, self).__init__()
        self.args = args
        self.num_cells = num_cells = 3
        self.dynamic_itr_l0 = Feature_Interaction_Layer0(args, num_cells, num_cells)
        self.dynamic_itr_l1 = Feature_Interaction_Layern(args, num_cells, num_cells,stage=1)
        self.dynamic_itr_l2 = Feature_Interaction_Layern(args, num_cells, 1,stage=2)
        total_paths = num_cells ** 2 * (num_layer_routing - 1) + num_cells
        self.path_mapping = nn.Linear(total_paths, path_hid)
        self.bn = nn.BatchNorm1d(args.embed_size)



    def forward(self, hsi,lidar):
        pairs_emb_lst, paths_l0 = self.dynamic_itr_l0(hsi[0],lidar[0])
        pairs_emb_lst, paths_l1 = self.dynamic_itr_l1(pairs_emb_lst, hsi[1],lidar[1])
        pairs_emb_lst, paths_l2 = self.dynamic_itr_l2(pairs_emb_lst, hsi[2],lidar[2])


        return pairs_emb_lst,paths_l2
