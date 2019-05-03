import itertools

import torch
import torch.nn as nn
from colorization_pytorch.models import create_model
from flownet2_pytorch.models import FlowNet2
import torch.nn.functional as F
import numpy as np


class VideoColorization(nn.Module):
    def __init__(self, opt, args, batchNorm=False, div_flow=20.):
        IDX_RANGE = 44
        self.color_model = create_model(opt)
        self.color_model.setup(opt)
        self.color_model.print_networks(True)
        self.flownet = FlowNet2(args, batchNorm=batchNorm, div_flow=div_flow)
        self.ori_index = torch.tensor(itertools.product(np.arange(IDX_RANGE), np.arange(IDX_RANGE))). \
            reshape(IDX_RANGE, IDX_RANGE, -1)

    def forward(self, color_data, flow_data):
        self.color_model.set_input(color_data[0])
        _, _, previous_feature_map = self.color_model.encode()
        self.color_model.set_input(color_data[1])
        conv1_2, conv2_2, feature_map = self.color_model.encode()
        flow = self.flownet(flow_data)  # B * 2 * 384 * 1024
        FlowWeight = F.interpolate(flow, previous_feature_map.size()[-2:], mode='bilinear')
        H1, W1 = previous_feature_map.size()[-2:]
        FlowWeight = FlowWeight.permute(0, 2, 3, 1).int()
        FlowWeight += self.ori_index
        FlowWeight[FlowWeight < 0] = 0
        FlowWeight[FlowWeight >= H1] = H1 - 1
        previous_feature_map = previous_feature_map[FlowWeight]
        
        fake_B_class, fake_B_reg = self.color_model.decode(conv1_2, conv2_2, feature_map)
