import matplotlib.pyplot as plt
import numpy as np

# Data
semantic_loss = [
    [0.2013, 0.2069, 0.2108],
    [0.2186, 0.2199, 0.2156],
    [0.2239, 0.2255, 0.2233],
    [0.2295, 0.2282, 0.2249],
    [0.2455, 0.2439, 0.2405],
    [0.7843, 0.7757, 0.7855]
]

pix_acc = [
    [0.9344, 0.9337, 0.9319], 
    [0.9239, 0.9228, 0.9246],
    [0.9218, 0.9209, 0.9217],
    [0.9197, 0.9204, 0.9215],
    [0.9144, 0.9152, 0.9164],
    [0.707, 0.7015, 0.6979]
]

mIOU = [
    [0.7476, 0.7472, 0.7332],
    [0.7166, 0.7163, 0.7151],
    [0.7093, 0.7075, 0.7093],
    [0.7051, 0.708, 0.7086],
    [0.6878, 0.6873, 0.6923],
    [0.3196, 0.3134, 0.3063]
]

depth_loss = [
    [0.0165, 0.0146, 0.0145],
    [0.0138, 0.0138, 0.0139],
    [0.0141, 0.0145, 0.0142],
    [0.0143, 0.0147, 0.0143],
    [0.015, 0.0153, 0.0147],
    [0.0576, 0.0468, 0.0483]
]

relative_loss = [
    [30.3527, 33.0248, 33.6863],
    [45.1175, 46.6774, 47.2279],
    [51.7863, 53.1629, 54.6916],
    [49.7038, 56.6077, 55.763],
    [65.2998, 72.8895, 63.5809],
    [92.1228, 101.6101, 100.9037]
]

delta_m = [
    [9.955, 8.576, 9.491],
    [19.237, 20.73, 21.369],
    [26.106, 28.167, 28.997],
    [24.769, 31.816, 30.04],
    [41.112, 48.364, 38.62],
    [168.302, 155.716, 158.3]
]

metrics = {
    "Semantic Segmentation Loss": semantic_loss,
    "Pixel Accuracy": pix_acc,
    "Mean IOU": mIOU,
    "Depth Loss": depth_loss,
    "Relative Error": relative_loss,
    "∆m%": delta_m
}

# Bar plot for each metric
labels = ["0", "0.5", "0.6", "0.7", "0.8", "0.9"]
colors = ["darkblue", "lightsteelblue", "lightsteelblue", "lightsteelblue", "lightsteelblue", "lightsteelblue"]

for metric_name, data in metrics.items():
    means = [np.mean(d) for d in data]
    
    fig, ax = plt.subplots()
    bars = ax.bar(labels, means, color=colors)
    
    # Add labels and title with increased fontsize
    ax.set_xlabel('Prune ratio of model', fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    
    # Increase fontsize of x-ticks
    plt.xticks(fontsize=12)
    
    plt.show()
