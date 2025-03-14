import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
import torch.nn.functional as F
import torchattacks


def test(model, test_set_loader, device):
    test_loss = 0
    correct = 0
    adv_examples = []
    attack = torchattacks.DeepFool(model, steps=8, overshoot=0.02)
    i = 1
    for data, target in test_set_loader:
        print ("Batch: ", i)
        i += 1
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        pred_init = output.max(1, keepdim=True)[1]
        loss = F.nll_loss(output, target)
        loss.backward()
        
        # data_grad = data.grad.data

        perturbed_batch = attack(data, target)

        # for i in range(data.shape[0]):
        #     print("Image: ", i)
        #     image = data[i]
        #     # img = torch.tensor(im[None,:,:,:],requires_grad =True)
        #     attack(image)
        #     perturbed_batch.append(perturbed_img.squeeze(0).detach())
        # perturbed_batch = torch.stack(perturbed_batch).to(device)
        # data = perturbed_batch

        output = model(perturbed_batch)
        test_loss += F.nll_loss(output, target, reduction="sum").item()
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

    # print(acc)
    # print(ex)
    return