import warnings
import sys
sys.path.append('../')  # Assuming the parent directory containing pytorch_grad_cam is one level above

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt
import requests
import torchvision
import os
from quantus import IROF
from collections import OrderedDict
from PIL import Image
from models import SegNet, SegNetMtan
from data import Cityscapes
from collections import OrderedDict
import torchvision.transforms as transforms
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import numpy as np
import matplotlib.pyplot as plt
import webcolors
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM

dataset = Cityscapes(root="dataset", train=False, augmentation=False)

cityscapes_train_set = Cityscapes(
    root="dataset", train=False, augmentation=False
)
cityscapes_test_set = Cityscapes(root="dataset", train=False)


test_loader = torch.utils.data.DataLoader(
    dataset=cityscapes_test_set, batch_size=1, shuffle=False
)
test_batch = len(test_loader)

i_want_this_image = 1
for i, batch in enumerate(test_loader):
    if i == i_want_this_image:
        test_data, semantic_label, depth_label = batch
        break

rgb_img = np.float32(test_data) / 255
input_tensor = torch.tensor(test_data).unsqueeze(0)
input_tensor = input_tensor.float().detach()
test_data_np = test_data.squeeze(0).cpu().numpy()
test_data_np = np.transpose(test_data_np, (1, 2, 0))

# Visualize the image
plt.imshow(test_data_np)
plt.axis('off')
plt.show()

model = SegNetMtan()
model.load_state_dict(torch.load('model_weights_f.pth', map_location=torch.device('cpu'))) # Load model weights here
model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    input_tensor = input_tensor.cuda()

output = model(test_data)
for i, tensor in enumerate(output):
    print(f"Output {i}: type={type(tensor)}, shape={tensor.shape}")

output_keys = ['output_0', 'output_1']  

output_dict = OrderedDict(zip(output_keys, output))

print(type(output_dict), output_dict.keys())


depth_np = depth_label.squeeze(0).squeeze(0).numpy()
unique_classes = np.unique(depth_np)

# Visualize depth label
plt.imshow(depth_np, cmap='viridis')  
plt.axis('off')
plt.show()


with torch.no_grad():


    depth_output_np = output[1].squeeze(0).squeeze(0).numpy()
    # Normalize depth values to range [0, 1]
    depth_min = np.min(depth_output_np)
    depth_max = np.max(depth_output_np)
    depth_normalized = (depth_output_np - depth_min) / (depth_max - depth_min)

# Display the depth estimation map
plt.imshow(depth_normalized, cmap='viridis') 
plt.axis('off')
plt.show()


# GradCAM 

class DepthEstimationTarget:
        def __init__(self, depth_map):
                self.depth_map = torch.from_numpy(depth_map)
                if torch.cuda.is_available():
                        self.depth_map = self.depth_map.cuda()

        def __call__(self, depth_output_np):
                depth_output_np = depth_output_np.squeeze(0)
                return (depth_output_np).sum()  
  

# Define the depth estimation target using the depth map
depth_target = [DepthEstimationTarget(depth_normalized)]
input_tensor = input_tensor.squeeze(1)


# Choose the target layer for visualization
target_layers = [model.segnet.conv_block_dec[4]]


image_np_uint8 = (test_data_np *255).astype(np.uint8)
image_float = image_np_uint8.astype(np.float32) / 255.0

test_data_float = test_data.float() / 255.0
test_data_float_np = test_data_float.numpy().squeeze(0)

print(test_data_float_np.shape)


with torch.enable_grad():

        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=depth_target)[0, :]
        rgb_img = np.float32(test_data) / 255
        test_data_float_np = test_data_float_np.transpose(1,2,0)
        depth_normalized_rgb = np.stack((depth_normalized,) * 3, axis=-1)
        cam_image = show_cam_on_image(image_float, grayscale_cam, use_rgb=True)

plt.figure(figsize=(10, 8))
plt.imshow(cam_image) 
plt.axis('off')
plt.show()


x_batch = test_data_np
y_batch = depth_output_np
a_batch = np.array(grayscale_cam)

a_batch = np.expand_dims(a_batch, axis=0)
x_batch = np.expand_dims(x_batch, axis=0)
y_batch = np.expand_dims(y_batch, axis=0)
x_batch = np.transpose(x_batch, (0, 3, 1, 2))

# Check the shapes
print(f"x_batch shape: {x_batch.shape}")
print(f"y_batch shape: {y_batch.shape}")
print(f"a_batch shape: {a_batch.shape}")


# Instantiate IROF metric
irof_metric = IROF(task='depth')

# Evaluate batch of explanations
irof_scores = irof_metric(model=model, task='depth', x_batch=x_batch, y_batch=y_batch, a_batch=a_batch)


print("IROF Scores (AUC):", irof_scores)




