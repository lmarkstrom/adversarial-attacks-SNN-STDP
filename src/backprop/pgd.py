from model import device, net, mnist_test, batch_size, dtype, num_steps
from torch.utils.data import DataLoader
import torchattacks
import torch.nn as nn
import numpy as np
import copy
import torch
import os
from scipy.stats import wasserstein_distance

class AttackWrapper(nn.Module):
    def __init__(self, snn_model):
        super().__init__()
        self.snn = snn_model

    def forward(self, x):
        x = x.view(x.size(0), -1)
        spk_rec, _ = self.snn(x)
        return spk_rec.sum(dim=0)

def pgd_attack(image, targets, attack) :
    image = image.view(image.size(0), -1)
    perturbed_data = attack(image, targets)
    return perturbed_data

def measure_perturbation(original, adversarial):
    if adversarial.shape != original.shape:
        adversarial = adversarial.view_as(original)
    perturbation = (adversarial - original).view(adversarial.size(0), -1)
    wass = wasserstein_distance(original.flatten(), adversarial.flatten())
    l0_norm = (torch.count_nonzero(perturbation).float() / perturbation.size(1)).mean().item()
    l1_norm = (torch.sum(torch.abs(perturbation), dim=1) / perturbation.size(1)).mean().item()
    l2_norm = torch.norm(perturbation, p=2, dim=1).mean().item()
    linf_norm = torch.max(torch.abs(perturbation)).item()
    return l0_norm, l1_norm, l2_norm, linf_norm, wass

def test(net):
    total = 0
    correct = 0
    batch = 1

    l0_norms = []
    l1_norms = []
    l2_norms = []
    linf_norms = []
    wasserstein = []

    # drop_last switched to False to keep all samples
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

    # attack = torchattacks.DeepFool(AttackWrapper(net), steps=20, overshoot=0.02)
    attack = torchattacks.PGD(AttackWrapper(net), eps=112/2550, alpha=1/255, steps=10, random_start=True)

    net.eval()
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        # print(f"Batch {batch}/{len(test_loader)}")
        # pert_data = deepfool(data, targets, net, attack)
        pert_data = pgd_attack(data, targets, attack)

        # measure l2-NORM
        # measure l2-NORM
        l0, l1, l2, linf, wass = measure_perturbation(data, pert_data)
        l0_norms.append(l0)
        l1_norms.append(l1)
        l2_norms.append(l2)
        linf_norms.append(linf)
        wasserstein.append(wass)

        # forward pass
        test_spk, _ = net(pert_data.view(pert_data.size(0), -1))

        # calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        print(f"Current Accuracy ({batch}): {100 * correct / total:.2f}%")
        print(f"Average L2-norm: {sum(l2_norms) / len(l2_norms):.2f}")
        batch += 1
    
    print(f"Total correctly classified test set images: {correct}/{total}")
    print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
    print(f"Average L0-norm: {sum(l0_norms) / len(l0_norms):.2f}")
    print(f"Average L1-norm: {sum(l1_norms) / len(l1_norms):.2f}")
    print(f"Average L2-norm: {sum(l2_norms) / len(l2_norms):.2f}")
    print(f"Average Linf-norm: {sum(linf_norms) / len(linf_norms):.2f}")
    print(f"Average Wass-dist: {sum(wasserstein) / len(wasserstein):.2f}")

def run_pgd():
    model_folder = 'models'
    model_path = os.path.join(model_folder, 'snn_model_2.pth')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    print(f"Model loaded from {model_path}")
    test(net)

run_pgd()