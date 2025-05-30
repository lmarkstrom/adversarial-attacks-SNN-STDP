from model import device, net, mnist_test, batch_size, dtype, num_steps, num_outputs
from torch.utils.data import DataLoader
import torchattacks
import torch.nn as nn
import numpy as np
import copy
import torch
import os
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns

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
    l0_norm = (torch.count_nonzero(perturbation).float() / batch_size).item()
    l1_norm = (torch.sum(torch.abs(perturbation), dim=1).float() / batch_size).mean().item()
    l2_norm = torch.norm(perturbation, p=2, dim=1).mean().item()
    linf_norm = torch.max(torch.abs(perturbation)).item()
    return l0_norm, l1_norm, l2_norm, linf_norm, wass

# Plot unperturbed and perturbed images
def image_plot(data, pert_data, mispredictions):
    fig, axes = plt.subplots(nrows=10, ncols=2, figsize=(5, 20))
    fig.suptitle('Original vs Perturbed Images', fontsize=16)

    axes[0, 0].set_title('Original')
    axes[0, 1].set_title('Perturbed')
    for i in range(10):
        img_orig = data[i].view(28, 28)
        img_pert = pert_data[i].view(28, 28) 

        # Original image
        axes[i, 0].imshow(img_orig, cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(i)
        # Perturbed image
        axes[i, 1].imshow(img_pert, cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 1].set_title(mispredictions[i].item())

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    save_path = os.path.join("images", "PGD.png")
    plt.savefig(save_path)
    plt.show()

def plot_heatmap(heatmap, tot_images):
    # Normalize rows to get percentages (optional)
    heatmap_norm = heatmap / heatmap.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        heatmap_norm, 
        annot=True, 
        fmt=".2f", 
        cmap="rocket", 
        xticklabels=[str(i) for i in range(num_outputs)], 
        yticklabels=[str(i) for i in range(num_outputs)]
    )
    plt.xlabel("Target label")
    plt.ylabel("Starting label")
    plt.title(f"Confusion Matrix: SNN Classification after FGSM Attack\n")
    plt.tight_layout()

    # Save and show
    plt.savefig(os.path.join("images", f"fgsm_heatmap_eps.png"))
    plt.show()

def test(net):
    total = 0
    correct = 0
    batch = 1

    l0_norms = []
    l1_norms = []
    l2_norms = []
    linf_norms = []
    wasserstein = []

    image_found = [False, False, False, False, False, False, False, False, False, False]
    images = [0,0,0,0,0,0,0,0,0,0]
    pert_images = [0,0,0,0,0,0,0,0,0,0]
    image_plot_done = False
    mispredictions = [0,0,0,0,0,0,0,0,0,0]

    # heatmap
    heatmap = np.zeros((num_outputs, num_outputs))
    tot_images = np.zeros((1, num_outputs))

    # drop_last switched to False to keep all samples
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

    # attack = torchattacks.DeepFool(AttackWrapper(net), steps=20, overshoot=0.02)
    attack = torchattacks.PGD(AttackWrapper(net), eps=14860/255000, alpha=1/255, steps=20, random_start=True)

    net.eval()
    for data, targets in test_loader:
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

        # Plot first not correctly predicted images of each class
        # if not image_plot_done:
        #     for idx, target in enumerate(targets):
        #         if (not image_found[target]) and predicted[idx] != targets[idx]:
        #             mispredictions[target] = predicted[idx]
        #             images[target] = data[idx]
        #             pert_images[target] = pert_data[idx]
        #             image_found[target] = True 
        #             if all(image_found):
        #                 image_plot(images, pert_images, mispredictions)

        for idx, target in enumerate(targets):
            true_label = target.item()
            pred_label = predicted[idx].item()
            heatmap[true_label, pred_label] += 1
            tot_images[0, true_label] += 1
    
    print(f"Total correctly classified test set images: {correct}/{total}")
    print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
    print(f"Average L0-norm: {sum(l0_norms) / len(l0_norms):.2f}")
    print(f"Average L1-norm: {sum(l1_norms) / len(l1_norms):.2f}")
    print(f"Average L2-norm: {sum(l2_norms) / len(l2_norms):.2f}")
    print(f"Average Linf-norm: {sum(linf_norms) / len(linf_norms):.2f}")
    print(f"Average Wass-dist: {sum(wasserstein) / len(wasserstein):.4f}")
    plot_heatmap(heatmap, tot_images)

def run_pgd():
    model_folder = 'models'
    model_path = os.path.join(model_folder, 'snn_model_SMPL.pth')
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    print(f"Model loaded from {model_path}")
    test(net)

run_pgd()