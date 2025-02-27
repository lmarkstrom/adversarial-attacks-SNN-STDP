import copy
import numpy as np
import tensorflow as tf
import torch.nn.functional as F


def deepfool(image, model, num_classes=10, overshoot=0.02, max_iter=50, shape=(28, 28, 1)):
    """
        This is derived from: 
            https://github.com/MyRespect/AdversarialAttack
        Which has no licence och copyright guidelines attached to it.
    """
    print(image.shape)
    image_array = np.array(image)
    # print(np.shape(image_array)) # 28*28

    image_norm = tf.cast(image_array / 255.0 - 0.5, tf.float32)
    image_norm = np.reshape(image_norm, shape)  # 28*28*1
    image_norm = image_norm[tf.newaxis, ...]  # 1*28*28*1

    f_image = model(image_norm).numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]
    I = I[0:num_classes]
    label = I[0]
    # print(label, "label")

    input_shape = np.shape(image_norm)
    pert_image = copy.deepcopy(image_norm)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0
    x = tf.Variable(pert_image)
    fs = model(x)
    k_i = label

    print(fs)  # shape=(1, 10)

    def loss_func(logits, I, k):
        # return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return logits[0, I[k]]

    while k_i == label and loop_i < max_iter:

        pert = np.inf

        one_hot_label_0 = tf.one_hot(label, num_classes)
        with tf.GradientTape() as tape:
            tape.watch(x)
            fs = model(x)
            # loss_value = loss_func(one_hot_label_0, fs)
            loss_value = loss_func(fs, I, 0)
        # grad_orig = tape.gradient(fs[0, I[0]], x)
        grad_orig = tape.gradient(loss_value, x)

        for k in range(1, num_classes):
            one_hot_label_k = tf.one_hot(I[k], num_classes)
            with tf.GradientTape() as tape:
                tape.watch(x)
                fs = model(x)
                # loss_value = loss_func(one_hot_label_k, fs)
                loss_value = loss_func(fs, I, k)
            # cur_grad = tape.gradient(fs[0, I[k]], x)
            cur_grad = tape.gradient(loss_value, x)

            w_k = cur_grad - grad_orig

            f_k = (fs[0, I[k]] - fs[0, I[0]]).numpy()

            pert_k = abs(f_k) / np.linalg.norm(tf.reshape(w_k, [-1]))

            if pert_k < pert:
                pert = pert_k
                w = w_k

        # print(pert)  # 1.3409956
        # print(np.shape(w))  # (1, 28, 28, 1)
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image_norm + (1 + overshoot) * r_tot

        x = tf.Variable(pert_image)

        fs = model(x)
        k_i = np.argmax(np.array(fs).flatten())

        loop_i += 1

    r_tot = (1 + overshoot) * r_tot

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

        _, _, _, _, data = deepfool(data, model)
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
    print('Test set ({}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        eps,
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