import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from alicatDataset import *

# hyperparameters and constants
batch_size = 8
learning_rate = 0.0001

input_size = 244*244
num_classes = 2

val_size = 155 # adjust accordingly when more images are added to dataset

# creating and loading data
dataset = AlicatDataset(create_image_data(), train=True, transform=torchvision.transforms.ToTensor())
train_ds, val_ds = random_split(dataset, [len(dataset)-val_size, val_size])
test_ds = AlicatDataset(create_image_data(), train=False, transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(train_ds, batch_size, learning_rate)
val_loader = DataLoader(val_ds, batch_size)
test_loader = DataLoader(test_ds, int(batch_size/2))

# Model


