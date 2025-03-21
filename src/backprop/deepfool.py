from model import device, net, mnist_test, batch_size, dtype, num_steps
from torch.utils.data import DataLoader
import torchattacks
import torch.nn as nn
import torch
import os

class AttackWrapper(nn.Module):
    def __init__(self, snn_model):
        super().__init__()
        self.snn = snn_model

    def forward(self, x):
        # Flatten input if needed
        x = x.view(x.size(0), -1)
        spk_rec, _ = self.snn(x)
        # Sum over time dimension to get final "logits"
        return spk_rec.sum(dim=0)


def deepfool(data, targets, net, attack):
    data = data.view(data.size(0), -1)
    # spk_rec, mem_rec = net(data.view(data.size(0), -1))
    # loss = nn.CrossEntropyLoss() # lossfunction

    # # Berökna loss, skillnad target och data, 
    # loss_val = torch.zeros((1), dtype=dtype, device=device)
    # for step in range(num_steps):
    #     loss_val += loss(mem_rec[step], targets)

    # # Backward pass to compute gradient w.r.t. input
    # net.zero_grad()
    # loss_val.backward()
    
    return attack(data, targets)

def measure_perturbation(original, adversarial):
    perturbation = (adversarial - original).view(adversarial.size(0), -1)
    l2_norm = torch.norm(perturbation, p=2, dim=1)
    return l2_norm.mean().item()

def test(net):
    total = 0
    correct = 0
    batch = 1

    l2_norms = []

    # drop_last switched to False to keep all samples
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

    attack = torchattacks.DeepFool(AttackWrapper(net), steps=8, overshoot=0.02)

    net.eval()
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        print(f"Batch {batch}/{len(test_loader)}")
        pert_data = deepfool(data, targets, net, attack)

        # measure l2-NORM
        l2 = measure_perturbation(data, pert_data)
        l2_norms.append(l2)

        # forward pass
        test_spk, _ = net(pert_data.view(pert_data.size(0), -1))

        # calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        batch += 1
    
    print(f"Total correctly classified test set images: {correct}/{total}")
    print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
    print(f"Average L2-norm: {sum(l2_norms) / len(l2_norms):.2f}")

def run_deepfool():
    model_folder = 'models'
    model_path = os.path.join(model_folder, 'snn_model_bin.pth')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    print(f"Model loaded from {model_path}")
    test(net)

run_deepfool()