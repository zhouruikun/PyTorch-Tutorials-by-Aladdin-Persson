# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# %%
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# hyper parameters
in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 64
num_epochs = 5

import sys

model = torchvision.models.vgg16(pretrained=True)
print(model)

# %%
