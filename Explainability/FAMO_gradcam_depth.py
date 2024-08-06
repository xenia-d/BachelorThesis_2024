#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import sys
sys.path.append('../')  # Assuming the parent directory containing pytorch_grad_cam is one level above

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
#from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt
# import requests
# import torchvision
import os
from collections import OrderedDict
from PIL import Image
from models import SegNet, SegNetMtan
from data import Cityscapes
from collections import OrderedDict
import torchvision.transforms as transforms
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM

# import webcolors

# In[2]:

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = Cityscapes(root="dataset", train=False, augmentation=False)

cityscapes_train_set = Cityscapes(
    root="dataset", train=False, augmentation=False
)
cityscapes_test_set = Cityscapes(root="dataset", train=False)


test_loader = torch.utils.data.DataLoader(
    dataset=cityscapes_test_set, batch_size=1, shuffle=False
)
test_batch = len(test_loader)

# Load the image, segmentation label, and depth map
# image_data, semantic, depth = dataset[i_want_this_image]
image_counter = 0
i_want_this_image = 33

# Variables to store IROF inputs
x_batches = []
y_batches = []
a_batches = []


parent_dir = "results"
output_dir = os.path.join(parent_dir, "original it 3")
os.makedirs(output_dir, exist_ok=True)

for i, batch in enumerate(test_loader):
    print("we are at image: ", image_counter)
    
    test_data, semantic_label, depth_label = batch

    rgb_img = np.float32(test_data.cpu().numpy()) / 255
    input_tensor = test_data.unsqueeze(0).float().detach().to(device)

    test_data_np = test_data.squeeze(0).cpu().numpy()
    test_data_np = np.transpose(test_data_np, (1, 2, 0))

    model = SegNetMtan()
    model.load_state_dict(torch.load('original_famo_weights_it3.pth', map_location=device))
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        test_data = test_data.cuda()
    
    output = model(test_data)
    output_keys = ['output_0', 'output_1']
    output_dict = OrderedDict(zip(output_keys, output))
    
    # Convert the label to numpy array
    depth_np = depth_label.squeeze(0).squeeze(0).cpu().numpy()

    with torch.no_grad():
        depth_output = output[1].squeeze(0).squeeze(0)
        depth_output_np = depth_output.cpu().numpy()

        depth_min = np.min(depth_output_np)
        depth_max = np.max(depth_output_np)
        depth_normalized_np = (depth_output_np - depth_min) / (depth_max - depth_min)
        depth_normalized = torch.from_numpy(depth_normalized_np).to(depth_output.device)

    class DepthEstimationTarget:
        def __init__(self, depth_map):
            self.depth_map = depth_map  # Expecting np.ndarray, no conversion needed
            if torch.cuda.is_available():
                self.depth_map = torch.from_numpy(self.depth_map).cuda()

        def __call__(self, depth_output_np):
            depth_output_np = depth_output_np.squeeze(0)
            return (depth_output_np).sum()

    depth_target = [DepthEstimationTarget(depth_normalized_np)]
    input_tensor = input_tensor.squeeze(1)
    target_layers = [model.segnet.conv_block_dec[4]]  # Adjust based on your model architecture

    image_np_uint8 = (test_data_np * 255).astype(np.uint8)
    image_float = image_np_uint8.astype(np.float32) / 255.0

    with torch.enable_grad():
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=depth_target)[0, :]
        
        rgb_img = np.float32(test_data.cpu().numpy()) / 255  # Move tensor to CPU before NumPy conversion
        test_data_float_np = test_data.float().cpu().numpy().squeeze(0).transpose(1, 2, 0)
        depth_normalized_rgb = np.stack((depth_normalized.cpu().numpy(),) * 3, axis=-1)  # Move tensor to CPU before NumPy conversion

        # print("img-shape", depth_normalized.shape)
        # print("cam shape", grayscale_cam.shape)
        cam_image = show_cam_on_image(image_float, grayscale_cam, use_rgb=True)

    # Prepare input for IROF metric
    x_batch = test_data_np
    y_batch = depth_output_np
    a_batch = np.array(grayscale_cam)

    a_batch = np.expand_dims(a_batch, axis=0)
    x_batch = np.expand_dims(x_batch, axis=0)
    y_batch = np.expand_dims(y_batch, axis=0)      # Reshape to (1, 128, 256)
    x_batch = np.transpose(x_batch, (0, 3, 1, 2))


    x_batches.append(x_batch)
    y_batches.append(y_batch)
    a_batches.append(a_batch)

    # if image_counter <= 50:
    #     image_dir = os.path.join(output_dir, f"image_{image_counter}")
    #     os.makedirs(image_dir, exist_ok=True)

    #     original_image_path = os.path.join(image_dir, f"{image_counter}_image.png")
    #     Image.fromarray(image_np_uint8).save(original_image_path)

    #     depth_output_path = os.path.join(image_dir, f"{image_counter}_depth_map.png")
    #     # Save depth map with viridis colormap
    #     plt.imshow(depth_normalized_np, cmap='viridis')
    #     plt.axis('off')  # Turn off axis
    #     plt.savefig(depth_output_path)
    #     plt.close()
        
        
    #     gradcam_output_path = os.path.join(image_dir, f"{image_counter}_gradcam.png")
    #     Image.fromarray(cam_image).save(gradcam_output_path)

    image_counter += 1

from quantus import IROF

irof_scores = []

x_batches = np.concatenate(x_batches, axis=0)
y_batches = np.concatenate(y_batches, axis=0)
a_batches = np.concatenate(a_batches, axis=0)


# print(f"x_batch shape: {x_batch.shape}")
# print(f"y_batch shape: {y_batch.shape}")
# print(f"a_batch shape: {a_batch.shape}")

irof_metric = IROF(task='depth')
metric_batch = irof_metric(model=model, task='depth', x_batch=x_batches, y_batch=y_batches, a_batch=a_batches)
irof_scores.extend(metric_batch)

print("IROF Scores (AUC):", irof_scores)
average_irof_score = np.mean(irof_scores)
print("Average IROF Score:", average_irof_score)

print("This is for original famo it 3")