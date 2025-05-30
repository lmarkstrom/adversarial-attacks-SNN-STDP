from model import test, device, net
import os
import torch

model_folder = 'models'
model_path = os.path.join(model_folder, 'snn_model_1.pth')
net.load_state_dict(torch.load(model_path, map_location=device))
net.eval()

print(f"Model loaded from {model_path}")

test(net)
