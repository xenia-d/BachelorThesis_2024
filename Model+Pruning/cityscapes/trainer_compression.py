import os
import logging
import wandb
from argparse import ArgumentParser
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
from netslim import update_bn, prune, load_pruned_model, network_slimming
from utils import ns_post_process_segnetmtan
import torchvision.models as models
from torchsummary import summary

from data import Cityscapes
from models import SegNet, SegNetMtan
from utils_cityscapes import ConfMatrix, delta_fn, depth_error
from utils_experiments import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from methods.weight_methods import WeightMethods

set_logger()

def update_model_architecture(model, pruned_weights):
    
    # Iterate through the layers of the model

    model.eval()  # Set the model to evaluation mode to ensure parameters are updated
    for name, module in model.named_modules():
        # Check if the module is a convolutional layer affected by pruning
        if isinstance(module, torch.nn.Conv2d):
            # Retrieve the corresponding pruned weights
            pruned_weight_name = name + ".weight"  # Assuming weight tensors are named "weight"
            pruned_weight = pruned_weights[pruned_weight_name]

            # Update the number of input and output channels in the convolutional layer
            module.out_channels = pruned_weight.shape[0]
            module.in_channels = pruned_weight.shape[1]

            
        if isinstance(module, torch.nn.BatchNorm2d):
            module.reset_running_stats()
            module.num_features = module.num_features if isinstance(module.num_features, int) else module.running_mean.shape[0]

            # Update running statistics tensors to match the new number of channels
            new_running_mean = module.running_mean[:module.num_features]
            new_running_var = module.running_var[:module.num_features]

            module.running_mean = torch.nn.Parameter(new_running_mean, requires_grad=False)
            module.running_var = torch.nn.Parameter(new_running_var, requires_grad=False)


    dummy_input = torch.randn(1, 3, 256, 128)

    with torch.no_grad():
        try:
            _ = model(dummy_input)  # Forward pass to trigger parameter updates
        except Exception as e:
            print("Error during forward pass:", e)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_trainable_parameters_zero(model):
    total_params = 0
    for param in model.parameters():
        total_params += param[param != 0].numel()
    return total_params

# Function to print Conv2d layers and their shapes
def print_conv_layers(model):
    print("Conv2d layers:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(name, module.weight.shape)

# Function to prune the model
def apply_pruning(model, prune_ratio):

    num_params = count_trainable_parameters_zero(model)
    print("Number of trainable parameters in the model:", num_params)

    print("pruning now...")
    # Prune the model using network slimming method
    model, pruned_weights = prune(model, (128, 256, 3), prune_ratio=prune_ratio, prune_method=network_slimming)
    print("Pruning successful!!")

    conv_filter_sizes = []
    

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_filter_sizes.append((name, module.weight.shape))


    pruned_num_params = count_trainable_parameters_zero(model)
    print("Number of trainable parameters in the pruned model:", pruned_num_params)
    print("Ratio Pruned: ", 1-(pruned_num_params/num_params))
    print("Ratio that is left: ", (pruned_num_params/num_params))

    return model, conv_filter_sizes

def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss


def main(path, lr, bs, device, output_dir):
    # ----
    # Nets
    # ---
    
    model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    model = model.to(device)

                           
    # Load the pretrained model
    pretrained_model_path = "model_compressed_weights.pth"
    if os.path.exists(pretrained_model_path):
        print("Loading pretrained model...")
        pretrained_model_state_dict = torch.load(pretrained_model_path,  map_location=torch.device('cpu'))
        model.load_state_dict(pretrained_model_state_dict)
    else:
        print(f"Pretrained model not found at {pretrained_model_path}. Training from scratch.")

    # Prune the model if a pruning ratio is specified
    if args.prune_ratio > 0:
        model, filter_list = apply_pruning(model, args.prune_ratio)
        pruned_model_path = "pruned_model_weights.pth"
        torch.save(model.state_dict(), pruned_model_path)
        print(f"Pruned model saved at {pruned_model_path}")


    # Load the pruned model for training if it exists
    pruned_model_path = "pruned_model_weights.pth"
    if os.path.exists(pruned_model_path):
        print("Loading pruned model...")
        model.load_state_dict(torch.load(pruned_model_path, map_location=device))
    else:
        print("Pruned model not found. Training with the original model.")

    # Apply modification to the model using the custom_list
    model.modify_conv_layers(filter_list)
    print(model)

    # dataset and dataloaders
    log_str = (
        "Applying data augmentation on NYUv2."
        if args.apply_augmentation
        else "Standard training strategy without data augmentation."
    )
    logging.info(log_str)

    cityscapes_train_set = Cityscapes(
        root=path.as_posix(), train=True, augmentation=args.apply_augmentation
    )
    cityscapes_test_set = Cityscapes(root=path.as_posix(), train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=cityscapes_train_set, batch_size=bs, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=cityscapes_test_set, batch_size=bs, shuffle=False
    )

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(
        args.method, n_tasks=2, device=device, **weight_methods_parameters[args.method]
    )

    # optimizer
    optimizer = torch.optim.Adam(
        [
            dict(params=model.parameters(), lr=lr),
            dict(params=weight_method.parameters(), lr=args.method_params_lr),
        ],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 12], dtype=np.float32)
    custom_step = -1
    conf_mat = ConfMatrix(model.segnet.class_nb)
    deltas = np.zeros([epochs,], dtype=np.float32)

    # some extra statistics we save during training
    loss_list = []

    for epoch in epoch_iter:
        cost = np.zeros(12, dtype=np.float32)

        for j, batch in enumerate(train_loader):
            custom_step += 1

            model.train()
            optimizer.zero_grad()

            train_data, train_label, train_depth = batch
            train_data, train_label = train_data.to(device), train_label.long().to(
                device
            )
            train_depth = train_depth.to(device)

            train_pred, features = model(train_data, return_representation=True)

            losses = torch.stack(
                (
                    calc_loss(train_pred[0], train_label, "semantic"),
                    calc_loss(train_pred[1], train_depth, "depth"),
                )
            )

            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
            )
            loss_list.append(losses.detach().cpu())
            update_bn(model)
            optimizer.step()

            if "famo" in args.method:
                with torch.no_grad():
                    train_pred = model(train_data, return_representation=False)
                    new_losses = torch.stack(
                        (
                            calc_loss(train_pred[0], train_label, "semantic"),
                            calc_loss(train_pred[1], train_depth, "depth"),
                        )
                    )
                    weight_method.method.update(new_losses.detach())

            # accumulate label prediction for every pixel in training images
            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            avg_cost[epoch, :6] += cost[:6] / train_batch

            epoch_iter.set_description(
                f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f}, "
            )

        # scheduler
        scheduler.step()
        # compute mIoU and acc
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        # evaluating test data
        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():  # operations inside don't track history
            for test_data, test_label, test_depth in test_loader:
                test_data, test_label = test_data.to(device), test_label.long().to(device)
                test_depth = test_depth.to(device)
        
                test_pred = model(test_data)
                test_loss = torch.stack(
                    (
                        calc_loss(test_pred[0], test_label, "semantic"),
                        calc_loss(test_pred[1], test_depth, "depth"),
                    )
                )
        
                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())
        
                cost[6] = test_loss[0].item()
                cost[9] = test_loss[1].item()
                cost[10], cost[11] = depth_error(test_pred[1], test_depth)
                avg_cost[epoch, 6:] += cost[6:] / len(test_loader)
        
            # compute mIoU and acc
            avg_cost[epoch, 7:9] = conf_mat.get_metrics()
        
            # Test Delta_m
            test_delta_m = delta_fn(
                avg_cost[epoch, [7, 8, 10, 11]]
            )
            deltas[epoch] = test_delta_m

            # print results
            print(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR ",
                flush=True
            )
            print(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | {avg_cost[epoch, 6]:.4f} "
                f"TEST: {avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} {avg_cost[epoch, 9]:.4f} | "
                f"{avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f}"
                f"| {test_delta_m:.3f}",
                flush=True
            )

            if wandb.run is not None:
                wandb.log({"Train Semantic Loss": avg_cost[epoch, 0]}, step=epoch)
                wandb.log({"Train Mean IoU": avg_cost[epoch, 1]}, step=epoch)
                wandb.log({"Train Pixel Accuracy": avg_cost[epoch, 2]}, step=epoch)
                wandb.log({"Train Depth Loss": avg_cost[epoch, 3]}, step=epoch)
                wandb.log({"Train Absolute Error": avg_cost[epoch, 4]}, step=epoch)
                wandb.log({"Train Relative Error": avg_cost[epoch, 5]}, step=epoch)

                wandb.log({"Test Semantic Loss": avg_cost[epoch, 6]}, step=epoch)
                wandb.log({"Test Mean IoU": avg_cost[epoch, 7]}, step=epoch)
                wandb.log({"Test Pixel Accuracy": avg_cost[epoch, 8]}, step=epoch)
                wandb.log({"Test Depth Loss": avg_cost[epoch, 9]}, step=epoch)
                wandb.log({"Test Absolute Error": avg_cost[epoch, 10]}, step=epoch)
                wandb.log({"Test Relative Error": avg_cost[epoch, 11]}, step=epoch)
                wandb.log({"Test ∆m": test_delta_m}, step=epoch)



            keys = [
                "Train Semantic Loss",
                "Train Mean IoU",
                "Train Pixel Accuracy",
                "Train Depth Loss",
                "Train Absolute Error",
                "Train Relative Error",

                "Test Semantic Loss",
                "Test Mean IoU",
                "Test Pixel Accuracy",
                "Test Depth Loss",
                "Test Absolute Error",
                "Test Relative Error",
            ]

            if "famo" in args.method:
                name = f"{args.method}_gamma{args.gamma}_sd{args.seed}"
            else:
                name = f"{args.method}_sd{args.seed}"
                
            # the stats file 

            torch.save({
                "delta_m": deltas,
                "keys": keys,
                "avg_cost": avg_cost,
                "losses": loss_list,
            }, f"./results/{name}.stats")
            
            
            #the pth file 
            
            torch.save(model.state_dict(), "./results/model_weights.pth")
        





if __name__ == "__main__":
    parser = ArgumentParser("Cityscapes", parents=[common_parser])

    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-4,
        n_epochs=200,
        batch_size=8,
        prune_ratio=0,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mtan",
        choices=["segnet", "mtan"],
        help="model type",
    )
    parser.add_argument(
        "--apply-augmentation", type=str2bool, default=True, help="data augmentations"
    )
    parser.add_argument("--wandb_project", type=str, default=None, help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Name of Weights & Biases Entity.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for saving results.")
    parser.add_argument("--prune-ratio", type=float, default=0, help="Ratio of pruning to apply")
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    device = get_device(gpus=args.gpu)
    main(path=args.data_path, lr=args.lr, bs=args.batch_size, device=device, output_dir=args.output_dir)

    if wandb.run is not None:
        wandb.finish()
