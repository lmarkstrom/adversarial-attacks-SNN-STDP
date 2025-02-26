import os

import matplotlib.pyplot as plt
import torchvision.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable

from buildModel import SpikingNet, download_mnist, test, batch_size, DATA_PATH

# Use GPU whever possible!
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

spiking_model = SpikingNet(device, n_time_steps=128, begin_eval=0)  # Reinitialize
spiking_model.load_state_dict(torch.load("spiking_model_state.pth"))
spiking_model.to(device)

training_set, testing_set = download_mnist(DATA_PATH)
test_set_loader = torch.utils.data.DataLoader(
    dataset=testing_set,
    batch_size=batch_size,
    shuffle=False)

test(spiking_model, device, test_set_loader)

