import sys
import os
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from alicatDataset import *
from catData import *

# hyperparameters and constants
batch_size = 8
learning_rate = 0.5

input_size = 244*244
num_classes = 2
hidden_size = 3

val_size = 100 # adjust accordingly when more images are added to dataset

# creating and loading data
dataset = AlicatDataset(create_image_data(), train=True, transform=torchvision.transforms.ToTensor())
train_ds, val_ds = random_split(dataset, [len(dataset)-val_size, val_size])
test_ds = AlicatDataset(create_image_data(), train=False, transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(train_ds, batch_size, learning_rate)
val_loader = DataLoader(val_ds, batch_size)
test_loader = DataLoader(test_ds, int(batch_size/2))

# Model
class AlicatModel(nn.Module):
    # constructor with hidden layers
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)

    # take in a batch reshapes all images in them
    def forward(self, xb):
        xb = xb.reshape(-1, 59536)
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        return out

    # takes in a batch returns loss
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    # takes in batch calculates loss returns validation loss and acc
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}

    # takes outputs and updates loss and acc per epoch
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    # string representation of val losses and accs
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


# initialize the model
model = AlicatModel(input_size, hidden_size, num_classes)


# Training methods
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        #train
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #validate
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# test train model
def predTrain():
    history1 = fit(10, 0.01, model, train_loader, val_loader)
    history2 = fit(10, 0.00001, model, train_loader, val_loader)
    return history2

# individual image testing
def which_cat(path):
    TESTDIR = r"C:\Users\Ali--\Desktop\Machine Learning Pytorch\Alicat\TestImages"
    path1 = os.path.join(TESTDIR, path)
    img, display = individual_image(path1)
    timg = torchvision.transforms.functional.to_tensor(img)
    xb = timg.unsqueeze(0)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)

    if preds[0].item() == 0:
        print("Coosa")
    elif preds[0].item() == 1:
        print("Hobbes")
        
    #display
    plt.axis("off")
    display = plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    plt.show()
    return

    







##TESTDIR = r"C:\Users\Ali--\Desktop\Machine Learning Pytorch\Alicat\TestImages"
##predTrain()
##file_paths = sys.argv[1:]
##for p in file_paths:
##    path = os.path.join(TESTDIR, p)
##    t = which_cat(path)
##    print(t)
    








