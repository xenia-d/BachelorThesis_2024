import warnings
import sys
sys.path.append('../')  

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt
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
from torch.utils.data import DataLoader
from quantus import IROF
from pytorch_grad_cam import GradCAM


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

image_counter = 0

# For IROF outputs
x_batches = []
y_batches = []
a_batches = []


parent_dir = "results"
output_dir = os.path.join(parent_dir, "segmentation_class_00_original_it3")
os.makedirs(output_dir, exist_ok=True)


for i, batch in enumerate(test_loader):
    print("we are at image: ", image_counter)

    test_data, semantic_label, depth_label = batch

    rgb_img = np.float32(test_data) / 255
    input_tensor = torch.tensor(test_data).unsqueeze(0)
    input_tensor = input_tensor.float()
    test_data_np = test_data.squeeze(0).cpu().numpy()
    test_data_np = np.transpose(test_data_np, (1, 2, 0))


    # # Visualize the image
    # plt.imshow(test_data_np)
    # plt.axis('off')
    # plt.show()

    model = SegNetMtan()
    model.load_state_dict(torch.load('original_famo_weights_it3.pth', map_location=device)) # Add path to model weights here
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        test_data = test_data.cuda()

    output = model(test_data) 
    output_keys = ['output_0', 'output_1'] 

    output_dict = OrderedDict(zip(output_keys, output))

    semantic_np = semantic_label.squeeze(0).numpy()
    # Define colormap for visualization
    num_classes = 7
    color_map = plt.cm.get_cmap('viridis', num_classes)

    # Create a color map dictionary assigning a unique color to each class label
    class_color_map = {class_label: color_map(i) for i, class_label in range(7)}



    # Iterate through unique classes present in the segmentation map
    unique_classes = np.unique(semantic_np)
    for class_index in unique_classes:
        if class_index < 7:
            color_rgb = class_color_map[class_index][:3]  # Extract RGB values
            # Convert RGB values to hexadecimal format
            hex_color = '#{:02x}{:02x}{:02x}'.format(int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255))
            # print(f"Class: {class_name}, Color: {hex_color}")


    # # Visualize the segmentation label with class labels overlayed
    # plt.imshow(semantic_np, cmap='viridis')  # Adjust the colormap as needed
    # plt.axis('off')
    # plt.show()

    #SHOWING THE SEGMAP
    
    normalized_masks = torch.nn.functional.softmax(output[0], dim=1).cpu()
    class_predict = normalized_masks.argmax(dim=1).numpy()
    unique_classes = np.unique(class_predict)
    color_map = plt.cm.get_cmap('viridis', len(unique_classes))
    color_image = color_map(class_predict)
    color_image = np.squeeze(color_image)


    color_image = np.clip(color_image, 0, 1)

    # # Display the color image
    # plt.imshow(color_image)
    # plt.axis('off')
    # plt.show()


    class_category = 0 # Enter desired class for explanation here (index 0-6 for Cityscapes)


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


    # GradCAM part

    class SemanticSegmentationTarget:
        def __init__(self, category, mask):
            self.category = category
            self.mask = torch.from_numpy(mask)
            if torch.cuda.is_available():
                self.mask = self.mask.cuda()
            
        def __call__(self, segmentation_output_tp):
            segmentation_output_tp = segmentation_output_tp.squeeze(0)
            return (segmentation_output_tp[self.category, :, :] * self.mask).sum()
        

    segmentation_output_tp = semantic_np
    input_tensor = input_tensor.squeeze(1)

    target_layers = [model.segnet.conv_block_dec[4]]  # Specift target layer for explanation
    targets = [SemanticSegmentationTarget(category=class_category, mask=class_mask_float)]


    with torch.enable_grad():
        # Initialize GradCAM
        cam = GradCAM(model=model, target_layers=target_layers)

        # Compute GradCAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
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
    