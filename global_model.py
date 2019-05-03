import itertools

import torch
import torch.nn as nn
from colorization_pytorch.models import create_model
from flownet2_pytorch.models import FlowNet2
import torch.nn.functional as F
import numpy as np


class Mask(nn.Module):
    def __init__(self, in_channels):
        super(Mask).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, data):
        return self.cnn(data)


class VideoColorization(nn.Module):
    def __init__(self, opt, args, batchNorm=False, div_flow=20.):
        super(VideoColorization).__init__()
        IDX_RANGE = 44
        self.color_model = create_model(opt)
        self.color_model.setup(opt)
        self.color_model.print_networks(True)
        self.flownet = FlowNet2(args, batchNorm=batchNorm, div_flow=div_flow)
        self.ori_index = torch.tensor(itertools.product(np.arange(IDX_RANGE), np.arange(IDX_RANGE))). \
            reshape(IDX_RANGE, IDX_RANGE, -1)
        self.mask = Mask(512)

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
        predicted_feature_map = previous_feature_map[FlowWeight]  # B*512*22*22
        delta_feature_map = torch.abs(predicted_feature_map - feature_map)
        M = self.mask(delta_feature_map)
        output_feature_map = (1 - M) * feature_map + M * predicted_feature_map
        fake_B_class, fake_B_reg = self.color_model.decode(conv1_2, conv2_2, output_feature_map)
