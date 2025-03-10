import torch
from model import SpikingNet, get_mnist, test, MODEL_PATH_BINARY, MODEL_PATH_BINARY_SIMPLE
from attacks.fgsm import run_fgsm
from attacks.deepfool import run_deepfool

# Use GPU whever possible!
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# _, test_set_loader = get_mnist(True)
_, test_set_loader = get_mnist(False)


def main():
    spiking_model = SpikingNet(device, n_time_steps=128, begin_eval=0)
    # spiking_model.load_state_dict(torch.load(MODEL_PATH_BINARY))
    spiking_model.load_state_dict(torch.load(MODEL_PATH_BINARY_SIMPLE))
    spiking_model.to(device)
    spiking_model.eval()

    run_fgsm(spiking_model, test_set_loader, device)
    # run_deepfool(spiking_model, test_set_loader, device)

main()
