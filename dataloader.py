import os
import sys
import csv
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader




def readFlow(flow_file):
    with open(flow_file, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print ('Magic number incorrect. Invalid .flo file')
            else:
                w = int(np.fromfile(f, np.int32, count=1))
                h = int(np.fromfile(f, np.int32, count=1))
                #print ('Reading %d x %d flo file' % (w, h))
                data = np.fromfile(f, np.float32, count=2*w*h)
                # Reshape data into 3D array (columns, rows, bands)
                data2D = np.resize(data, (w, h, 2))
    return torch.from_numpy(data2D).transpose(0,2)


# Each data point in the Trainset is 4 tensors: 
# img1 (3 * 436 * 1024 i.e. RGB_channel * height * width), img2, flow (2 * 436 * 1024 i.e. flow_channel * h * w), mask (1 * 436 * 1024 i.e. mask_channel * h * w)
                                          
class Trainset(Dataset):
    def __init__(self, data_dir):
        flow_root = "flow/"
        mask_root = "occlusions/"
        data_root = "clean/"
        self.image_pair = []
        self.flow = []
        self.mask = []
        for root, directory, file in os.walk(data_dir + data_root):
            for folder in directory:
                #print(folder)
                data_path = data_dir + data_root + folder + '/'
                flow_path = data_dir + flow_root + folder + '/'
                mask_path = data_dir + mask_root + folder + '/'
                for f in os.listdir(mask_path):
                    mask = Image.open(mask_path + f)
                    mask = torchvision.transforms.ToTensor()(mask)
                    

                    f = f.split('.')[0]
                    flow = readFlow(flow_path + f + '.flo')

                    name, idx = f.split('_')[:]
                
                    img1 = Image.open(data_path + f + '.png')
                    img1 = torchvision.transforms.ToTensor()(img1)

                    idx = int(idx)
                    idx += 1
                    if idx < 10:
                        filename = data_path + name + '_000' + str(idx) + '.png'
                    else:
                        filename = data_path + name + '_00' + str(idx) + '.png'


                    img2 = Image.open(filename)

                    img2 = torchvision.transforms.ToTensor()(img2)
                    self.image_pair.append([img1, img2])
                    self.flow.append(flow)
                    self.mask.append(mask)


    def __len__(self):
        return len(self.flow)

    def __getitem__(self, index):  
        return self.image_pair[index][0], self.image_pair[index][1], self.flow[index], self.mask[index]

def LoadData(data_dir):
    flow_root = "flow/"
    mask_root = "occlusions/"
    data_root = "clean/"


    for root, directory, file in os.walk(data_dir + data_root):
        for folder in directory:
            print(folder)
            data_path = data_dir + data_root + folder + '/'
            flow_path = data_dir + flow_root + folder + '/'
            mask_path = data_dir + mask_root + folder + '/'
            for f in os.listdir(mask_path):
                mask = Image.open(mask_path + f)
                mask = torchvision.transforms.ToTensor()(mask)
                

                f = f.split('.')[0]
                flow = readFlow(flow_path + f + '.flo')

                name, idx = f.split('_')[:]
            
                img1 = Image.open(data_path + f + '.png')
                img1 = torchvision.transforms.ToTensor()(img1)

                idx = int(idx)
                idx += 1
                if idx < 10:
                    filename = data_path + name + '_000' + str(idx) + '.png'
                else:
                    filename = data_path + name + '_00' + str(idx) + '.png'


                img2 = Image.open(filename)

                img2 = torchvision.transforms.ToTensor()(img2)
                print(img1.shape)
                print(img2.shape)
                print(flow.shape)
                print(mask.shape)
                ll = [img1, img2, flow, mask]

        







                



def main(argv):
    if len(argv) < 1:
        print("NOT ENOUGHT ARGUMENTS! PLEASE SPECIFY DATA DIR\n")
        sys.exit()

    data_root = argv[0]
    Trainset(data_root)


if __name__ == "__main__":
    main(sys.argv[1:])
