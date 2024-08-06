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
# import webcolors
from torch.utils.data import DataLoader
from quantus import IROF


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
output_dir = os.path.join(parent_dir, "segmentation_class_00_original_it3")
os.makedirs(output_dir, exist_ok=True)


for i, batch in enumerate(test_loader):
    print("we are at image: ", image_counter)
    # if i == i_want_this_image:
    #     test_data, semantic_label, depth_label = batch
    #     break

    test_data, semantic_label, depth_label = batch

    rgb_img = np.float32(test_data) / 255
    input_tensor = torch.tensor(test_data).unsqueeze(0)

    input_tensor = input_tensor.float()

    # print(input_tensor.shape)

    test_data_np = test_data.squeeze(0).cpu().numpy()

    test_data_np = np.transpose(test_data_np, (1, 2, 0))
    # Print the shape of the image to verify its dimensions
    # print("Image shape:", test_data_np.shape)


    # # Visualize the image
    # plt.imshow(test_data_np)
    # plt.axis('off')
    # plt.show()


    # In[3]:


    model = SegNetMtan()
    model.load_state_dict(torch.load('original_famo_weights_it3.pth', map_location=device))
    model.eval()

    print("original model it3")
    # print(model_state_dict.keys())

    # print(model.state_dict().keys())
    if torch.cuda.is_available():
        model = model.cuda()
        test_data = test_data.cuda()

    output = model(test_data)  # Corrected this line to pass input_tensor to the model
    # print(output)
    # for i, tensor in enumerate(output):
        # print(f"Output {i}: type={type(tensor)}, shape={tensor.shape}")

    output_keys = ['output_0', 'output_1']  # Add more keys if needed

    # Create a dictionary from the list of tensors using the predefined keys
    output_dict = OrderedDict(zip(output_keys, output))

    # print(type(output_dict), output_dict.keys())


    # In[4]:


    # Convert the semantic segmentation label to numpy array
    semantic_np = semantic_label.squeeze(0).numpy()

    # # Get the unique classes present in the semantic map
    # unique_classes = np.unique(semantic_np)
    # print(len(unique_classes))

    # Map class indices back to class labels using the sem_classes list
    sem_classes = [
        'road', 'sidewalk', 'parking', 'rail track', 'person',  'rider', 'car', 
    ]

    # Define colormap for visualization
    num_classes = len(sem_classes)
    color_map = plt.cm.get_cmap('viridis', num_classes)

    # Create a color map dictionary assigning a unique color to each class label
    class_color_map = {class_label: color_map(i) for i, class_label in enumerate(sem_classes)}



    # Iterate through unique classes present in the segmentation map
    unique_classes = np.unique(semantic_np)
    for class_index in unique_classes:
        # Check if the class index is within the range of available class labels
        if class_index < len(sem_classes):
            class_name = sem_classes[int(class_index)]
            color_rgb = class_color_map[class_name][:3]  # Extract RGB values
            # Convert RGB values to hexadecimal format
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255))
            # print(f"Class: {class_name}, Color: {hex_color}")


    # # Visualize the segmentation label with class labels overlayed
    # plt.imshow(semantic_np, cmap='viridis')  # Adjust the colormap as needed
    # plt.axis('off')
    # plt.show()


    # In[5]:


    import matplotlib.pyplot as plt
    import numpy as np


    #SHOWING THE SEGMAP
    
    

    normalized_masks = torch.nn.functional.softmax(output[0], dim=1).cpu()

    # Get the class prediction by selecting the class index with maximum probability
    class_predict = normalized_masks.argmax(dim=1).numpy()

    # Get the unique classes present in the prediction
    unique_classes = np.unique(class_predict)

    # Define a colormap based on the number of unique classes
    color_map = plt.cm.get_cmap('viridis', len(unique_classes))

    # Apply the colormap to the class prediction directly
    color_image = color_map(class_predict)

    # Squeeze the color_image if it has an extra dimension
    color_image = np.squeeze(color_image)


    # Ensure values are in the correct range (0 to 1)
    color_image = np.clip(color_image, 0, 1)

    # # Display the color image
    # plt.imshow(color_image)
    # plt.axis('off')
    # plt.show()



    # In[6]:

    normalized_masks = torch.nn.functional.softmax(output[0], dim=1).cpu()

    # image_np = np.transpose(image_np, (1, 2, 0))

    # car_category = sem_class_to_idx["car"]
    class_category = 0


    class_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().numpy()
    class_mask_uint8 = 255 * np.uint8(class_mask == class_category)
    class_mask_float = np.float32(class_mask == class_category)

    image_np_uint8 = (test_data_np * 255).astype(np.uint8)
    car_mask_uint8 = np.uint8(class_mask_uint8)


    # plt.imshow(image_np_uint8)
    # plt.axis('off')
    # plt.show()

    # Concatenate image_data and car_mask_uint8 horizontally
    both_images = np.hstack((image_np_uint8, np.repeat(class_mask_uint8[:, :, None], 3, axis=-1)))
    Image.fromarray(both_images)


    # In[7]:


    from pytorch_grad_cam import GradCAM

    class SemanticSegmentationTarget:
        def __init__(self, category, mask):
            self.category = category
            self.mask = torch.from_numpy(mask)
            if torch.cuda.is_available():
                self.mask = self.mask.cuda()
            
        def __call__(self, segmentation_output_tp):
            segmentation_output_tp = segmentation_output_tp.squeeze(0)
            return (segmentation_output_tp[self.category, :, :] * self.mask).sum()
        
    # # Print shapes and values for debugging
    # print("Input tensor shape:", input_tensor.shape)
    # print("RGB image shape:", rgb_img.shape)
    # print("Semantic segmentation mask shape:", semantic_np.shape)
    # print("Car mask shape:", car_mask_float.shape)

    # print(semantic_np.shape)

    # segmentation_output_tp = torch.transpose(semantic_np, 0, 1)

    segmentation_output_tp = semantic_np
    input_tensor = input_tensor.squeeze(1)
    # print(input_tensor.shape)
    # print("Semantic segmentation mask shape (tp):", segmentation_output_tp.shape)

    target_layers = [model.segnet.conv_block_dec[4]]  # Choose the desired layer for visualization

    targets = [SemanticSegmentationTarget(category=class_category, mask=class_mask_float)]


    with torch.enable_grad():
        # Initialize GradCAM
        cam = GradCAM(model=model, target_layers=target_layers)

        # print(input_tensor)
        # Compute GradCAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

        # Transpose the original image to match the dimensions expected by show_cam_on_image

        # image_np_tp = image_np.transpose(2, 0, 1)
        image_float = image_np_uint8.astype(np.float32) / 255.0

        # Visualize GradCAM
        cam_image = show_cam_on_image(image_float, grayscale_cam, use_rgb=True)
    

    # # Visualize GradCAM using matplotlib
    # plt.figure(figsize=(10, 8))  # Set the size of the figure
    # plt.imshow(cam_image)
    # plt.axis('off')  # Turn off axis
    # plt.show()

    # Prepare input for IROF metric
    x_batch = test_data_np
    y_batch = class_predict 
    a_batch = np.array(grayscale_cam)

    a_batch = np.expand_dims(a_batch, axis=0)
    x_batch = np.expand_dims(x_batch, axis=0)
    x_batch = np.transpose(x_batch, (0, 3, 1, 2))

    # Append only if a_batch is not all zeros
    # if not np.all(a_batch == 0):
    x_batches.append(x_batch)
    y_batches.append(y_batch)
    a_batches.append(a_batch)

    if image_counter <= 50:
    
        # make image dir
        image_dir = os.path.join(output_dir, f"image_{image_counter}")
        os.makedirs(image_dir, exist_ok=True)

        # save input image
        original_image_path = os.path.join(image_dir, f"{image_counter}_image.png")
        Image.fromarray(image_np_uint8).save(original_image_path)

        # save mask
        segmentation_output_path = os.path.join(image_dir, f"{image_counter}_class_map.png")
        segmentation_image = Image.fromarray(car_mask_uint8)
        segmentation_image.save(segmentation_output_path)

        # save explanation
        gradcam_output_path = os.path.join(image_dir, f"{image_counter}_gradcam.png")
        Image.fromarray(cam_image).save(gradcam_output_path)

        # Save the semantic output as a PNG file
        semantic_output_path = os.path.join(image_dir, f"{image_counter}_semantic_output.png")
        semantic_output_image = Image.fromarray((color_image * 255).astype(np.uint8))
        semantic_output_image.save(semantic_output_path)
        

    image_counter += 1

# In[9]:


irof_scores = []


# Check if batches are not empty before concatenating
if x_batches:
    x_batches = np.concatenate(x_batches, axis=0)
    y_batches = np.concatenate(y_batches, axis=0)
    a_batches = np.concatenate(a_batches, axis=0)
else:

    print("x batch is empty")
    x_batches = np.array([])
    y_batches = np.array([])
    a_batches = np.array([])
# print(x_batches)

# Instantiate IROF metric
irof_metric = IROF(task='segmentation')

metric_batch = irof_metric(model=model, task='segmentation', x_batch=x_batches, y_batch=y_batches, a_batch=a_batches)

# Evaluate batch of explanations
irof_scores.extend(metric_batch)

print("This is for class category: ", class_category)
# Print all the IROF scores
print("IROF Scores:", irof_scores)

# Compute and print the average IROF score
average_irof_score = np.mean(irof_scores)
print("Average IROF Score:", average_irof_score)
    