import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from catData import *


class AlicatDataset(Dataset):
    def __init__(self, data_list,train=None, transform=None):
        self.data = data_list
        self.length = len(data_list)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = self.data[index][0]
        label = torch.tensor(int(self.data[index][1]))

        if self.transform:
            image = self.transform(image)

        return (image, label)

    

    
