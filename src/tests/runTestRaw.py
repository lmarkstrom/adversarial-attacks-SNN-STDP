import torch
from model import SpikingNet, get_mnist, test, MODEL_PATH_BINARY, MODEL_PATH

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

spiking_model = SpikingNet(device, n_time_steps=128, begin_eval=0)
spiking_model.load_state_dict(torch.load(MODEL_PATH_BINARY))
spiking_model.to(device)
spiking_model.eval()

_, test_set_loader = get_mnist()

test(spiking_model, device, test_set_loader)