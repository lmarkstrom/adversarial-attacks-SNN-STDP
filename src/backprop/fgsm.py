from model import device, net, mnist_test, batch_size, dtype, num_steps
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os

epsilons = [0, 0.05, .1, .15, .2, .25, .3]

def fgsm_attack(data, targets, epsilon, net):
    data.requires_grad = True

    # Skicka in data i nätverket
    spk_rec, mem_rec = net(data.view(data.size(0), -1))
    loss = nn.CrossEntropyLoss() # lossfunction

    # Berökna loss, skillnad target och data, 
    loss_val = torch.zeros((1), dtype=dtype, device=device)
    for step in range(num_steps):
        loss_val += loss(mem_rec[step], targets)

    # Backward pass to compute gradient w.r.t. input
    net.zero_grad()
    loss_val.backward()

    # FGSM perturbation
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)  # keep pixel values in range

    return perturbed_data

def measure_perturbation(original, adversarial):
    perturbation = (adversarial - original).view(adversarial.size(0), -1)
    l2_norm = torch.norm(perturbation, p=2, dim=1)
    return l2_norm.mean().item()

def test(net, eps):
    total = 0
    correct = 0

    l2_norms = []

    # drop_last switched to False to keep all samples
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

    net.eval()
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        pert_data = fgsm_attack(data, targets, eps, net)

        # measure l2-NORM
        l2 = measure_perturbation(data, pert_data)
        l2_norms.append(l2)

        # forward pass
        test_spk, _ = net(pert_data.view(pert_data.size(0), -1))

        # calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    print(f"Epsilon: {eps}")    
    print(f"Total correctly classified test set images: {correct}/{total}")
    print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
    print(f"Average L2-norm: {sum(l2_norms) / len(l2_norms):.2f}")

def run_fgsm():
    model_folder = 'models'
    model_path = os.path.join(model_folder, 'snn_model_simple.pth')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    print(f"Model loaded from {model_path}")
    for eps in epsilons:
        test(net, eps)

run_fgsm()