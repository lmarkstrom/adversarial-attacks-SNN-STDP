
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

from buildModel import SpikingNet, download_mnist, test, batch_size, DATA_PATH, MODEL_PATH_CL, MODEL_PATH

# Use GPU whever possible!
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


pretrained_model = './spiking_model_state.pth'
epsilons = [0, 0.05, .1, .15, .2, .25, .3]

training_set, testing_set = download_mnist(DATA_PATH)
test_set_loader = torch.utils.data.DataLoader(
    dataset=testing_set,
    batch_size=batch_size,
    shuffle=False)

spiking_model = SpikingNet(device, n_time_steps=128, begin_eval=0)
spiking_model.load_state_dict(torch.load(MODEL_PATH_CL))
spiking_model.to(device)
spiking_model.eval()

def fgsm_attack(image, epsilon, data_grad):
    """
        This is derived from: 
            https://pytorch.org/tutorials/beginner/fgsm_tutorial.html 
        Which is open for research purpose with citation. 
    """
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def test(model, test_set_loader, eps):
    # model.eval() perhaps?
    test_loss = 0
    correct = 0
    adv_examples = []
    
    for data, target in test_set_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        pred_init = output.max(1, keepdim=True)[1]
        # print(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        
        data_grad = data.grad.data

        data = fgsm_attack(data, eps, data_grad)
        output = model(data)
        # Note: with `reduce=True`, I'm not sure what would happen with a final batch size 
        # that would be smaller than regular previous batch sizes. For now it works.
        test_loss += F.nll_loss(output, target, reduce=True).item() # sum up batch loss
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        if len(adv_examples) < 5:
            adv_ex = data[0].squeeze().detach().cpu().numpy()
            adv_examples.append((pred_init[0].item(), pred[0].item(), adv_ex))

    test_loss /= len(test_set_loader.dataset)
    print("")
    accuracy = 100. * correct / len(test_set_loader.dataset)
    print('Test set ({}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        eps,
        test_loss, 
        correct, len(test_set_loader.dataset),
        accuracy))
    print("")
    return accuracy, adv_examples

accuracies = []
examples = []

for eps in epsilons:
    acc, ex = test(spiking_model, test_set_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

print(epsilons)
print(accuracies)

plt.figure(figsize=(5,5))

plt.plot(epsilons, accuracies, '*-')

plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")

plt.figure(figsize=(8,10))

cnt = 0
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[i]), cnt)
        orig, adv, ex = examples[i][j]

        plt.imshow(ex, cmap='gray')
        
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title("{} -> {}".format(orig, adv))
        
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        
plt.tight_layout()
plt.show()