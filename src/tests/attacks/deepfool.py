import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import math
import copy
import os
import matplotlib.pyplot as plt
from PIL import Image


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=10):
    """
        This is derived from: 
            https://github.com/aminul-huq/DeepFool
        Which has no licence och copyright guidelines attached to it.
    """
    f_image = net.forward(image).data.numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.detach().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = torch.tensor(pert_image[None, :],requires_grad=True)
    
    fs = net.forward(x[0])
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.numpy().copy()

        for k in range(1, num_classes):
            
            #x.zero_grad()
            
            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = torch.tensor(pert_image, requires_grad=True)
        fs = net.forward(x[0])
        k_i = np.argmax(fs.data.numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image

def test(model, test_set_loader, device):
    test_loss = 0
    correct = 0
    adv_examples = []
    
    for data, target in test_set_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        pred_init = output.max(1, keepdim=True)[1]
        loss = F.nll_loss(output, target)
        loss.backward()
        
        # data_grad = data.grad.data

        perturbed_batch = []
        for i in range(data.shape[0]):
            print("Image: ", i)
            im = data[i].cpu()
            img = torch.tensor(im[None,:,:,:],requires_grad =True)
            _, _, _, _, perturbed_img = deepfool(img, model, max_iter=50)
            perturbed_batch.append(perturbed_img.squeeze(0).detach())
        perturbed_batch = torch.stack(perturbed_batch).to(device)
        data = perturbed_batch

        output = model(data)
        test_loss += F.nll_loss(output, target, reduce=True).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        if len(adv_examples) < 5:
            adv_ex = data[0].squeeze().detach().cpu().numpy()
            adv_examples.append((pred_init[0].item(), pred[0].item(), adv_ex))

    test_loss /= len(test_set_loader.dataset)
    print("")
    accuracy = 100. * correct / len(test_set_loader.dataset)
    print('Test set (-): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, 
        correct, len(test_set_loader.dataset),
        accuracy))
    print("")
    return accuracy, adv_examples

def run_deepfool(model, test_set_loader, device):
    accuracies = []
    examples = []
    acc, ex = test(model, test_set_loader, device)

    print(acc)
    print(ex)
    return