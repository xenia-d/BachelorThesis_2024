import numpy as np
from scipy.stats import wilcoxon
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


######### For segmentation - enter IROF scores as a txt file ###########

pruned_05 = np.loadtxt("pruned_05_irofs_seg.txt", dtype = float, delimiter=',')

pruned_06 = np.loadtxt("pruned_06_irofs_seg.txt", dtype =float, delimiter=',')

pruned_07 = np.loadtxt("pruned_07_irofs_seg.txt", dtype =float, delimiter=',')

pruned_08 = np.loadtxt("pruned_08_irofs_seg.txt", dtype =float, delimiter=',')

pruned_09 = np.loadtxt("pruned_09_irofs_seg.txt", dtype =float, delimiter=',')

original_seg = np.loadtxt("original_famo_irofs_seg.txt", dtype=float, delimiter=',')

original_rm_2 = np.loadtxt("original_removedclass2_irofs.txt", dtype=float, delimiter=',')

all_segmentation = [original_seg, pruned_05, pruned_06, pruned_07, pruned_08, pruned_09]


######### For depth estimation - enter IROF scores as a txt file ###########

original_depth = np.loadtxt("original famo depth irofs.txt", dtype=float, delimiter=',')

pruned05_depth = np.loadtxt("pruned 05 depth irofs.txt", dtype=float, delimiter=',')

pruned06_depth = np.loadtxt("pruned 06 depth irofs.txt", dtype=float, delimiter=',')

pruned07_depth = np.loadtxt("pruned 07 depth irofs.txt", dtype=float, delimiter=',')

pruned08_depth = np.loadtxt("pruned 08 depth irofs.txt", dtype=float, delimiter=',')

pruned09_depth = np.loadtxt("pruned 09 depth irofs.txt", dtype=float, delimiter=',')

all_depth = [original_depth, pruned05_depth, pruned06_depth, pruned07_depth, pruned08_depth, pruned09_depth]

### QQ PLOTS ##########

# Labels for the plots
labels = ["Unpruned", "Pruned 0.5", "Pruned 0.6", "Pruned 0.7", "Pruned 0.8", "Pruned 0.9"]

# Q-Q Plots for segmentation data
plt.figure(figsize=(15, 10))
for i, data in enumerate(all_segmentation):
    plt.subplot(2, 3, i+1)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot: {labels[i]} Segmentation')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')

plt.tight_layout()
plt.show()

# Q-Q Plots for depth data
plt.figure(figsize=(15, 10))
for i, data in enumerate(all_depth):
    plt.subplot(2, 3, i+1)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f'Q-Q Plot: {labels[i]} Depth')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')

plt.tight_layout()
plt.show()



print("Length of original", len(original_depth))
print("Length of pruned list", len(pruned05_depth))

median1 = np.median(original_depth)
median2 = np.median(pruned06_depth)

# Calculate the percentage change
# percentage_change_av = ((average2 - average1) / average1) * 100
percentage_change_mdn = ((median2 - median1) / median1) * 100

iqr1 = np.percentile(original_depth, 75) - np.percentile(original_depth, 25)
iqr2 = np.percentile(pruned09_depth, 75) - np.percentile(pruned09_depth, 25)

# Print the results
print(f"Median of original: {median1}")
print(f"Median of pruned: {median2}")
print(f"Percentage change median: {percentage_change_mdn:.2f}%")
print(f"IQR of original: {iqr1}")
print(f"IQR of pruned: {iqr2}")


# Perform Wilcoxon signed-rank test
statistic, p_value = wilcoxon(original_seg, pruned_05)



# Print the results
print(f"Wilcoxon signed-rank statistic: {statistic}")
print(f"P-value: {p_value}")



# Data for boxplot SEGMENTATION
data_segmentation = [original_seg, pruned_05, pruned_06, pruned_07, pruned_08, pruned_09]
labels = ["Unpruned", "Pruned 0.5", "Pruned 0.6", "Pruned 0.7", "Pruned 0.8", "Pruned 0.9"]
colors = ["darkblue", "lightsteelblue", "lightsteelblue", "lightsteelblue", "lightsteelblue", "lightsteelblue"]

# Create the boxplot for segmentation
fig, ax = plt.subplots()
box = ax.boxplot(data_segmentation, patch_artist=True, labels=labels)

# Color the boxplot and change the median line color to white
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set_color('white')

# Customize the plot
plt.xlabel('Model Type')
plt.ylabel('Median IROF values')
# plt.title('Boxplot of Different Models (Segmentation)')
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()

####################################################################################################################################

# Data for boxplot depth
data_depth = [original_depth, pruned05_depth, pruned06_depth, pruned07_depth, pruned08_depth, pruned09_depth]
labels = ["Unpruned", "Pruned 0.5", "Pruned 0.6", "Pruned 0.7", "Pruned 0.8", "Pruned 0.9"]
colors = ["darkblue", "lightsteelblue", "lightsteelblue", "lightsteelblue", "lightsteelblue", "lightsteelblue"]

# Create the boxplot for depth
fig, ax = plt.subplots()
box = ax.boxplot(data_depth, patch_artist=True, labels=labels)

# Color the boxplot and change the median line color to white
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
for median in box['medians']:
    median.set_color('white')

# Customize the plot
plt.xlabel('Model Type')
plt.ylabel('Median IROF values')
# plt.title('Boxplot of Different Models (Depth)')
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plot
plt.show()