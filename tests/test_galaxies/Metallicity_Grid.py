from pathlib import Path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

SCRIPT_DIR = Path(__file__).resolve().parent

Z_sun = 0.0142
Fm = -1
nabla_Fe_to_H = -0.075
def cal_Z(R, tau):
    Fe_to_H = Fm + nabla_Fe_to_H * R - (Fm + nabla_Fe_to_H * 8.7) * (1 - tau / 12) ** 0.3
    log10_Z = 0.977 * Fe_to_H + np.log10(Z_sun)
    Z = 10 ** log10_Z
    return Z

# print(cal_Z(20, 10))
# Define the number of grid cells.
R_grid = 20
tau_grid = 12

# Define the R and tau ranges.
R_edges = np.linspace(0, R_grid, R_grid + 1)  # R bin edges.
tau_edges = np.linspace(0, tau_grid, tau_grid + 1)  # tau bin edges.

# Initialize the mean-metallicity array.
Z_mean = np.zeros((R_grid, tau_grid))

# Target metallicity array.
target_array = np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
                         1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
                         1e-2, 2e-2, 3e-2])

RNG_SEED = 1
rng = np.random.default_rng(RNG_SEED)

# Compute the mean metallicity in each grid cell.
for i in range(tau_grid):
    for j in range(R_grid):
        # Define the grid-cell range.
        R_min, R_max = R_edges[j], R_edges[j + 1]
        tau_min, tau_max = tau_edges[i], tau_edges[i + 1]

        # Randomly sample within the grid cell.
        R_samples = rng.uniform(R_min, R_max, 1000)
        tau_samples = rng.uniform(tau_min, tau_max, 1000)

        # Compute metallicities and take the mean.
        Z_samples = cal_Z(R_samples, tau_samples)
        mean_value = np.mean(Z_samples)

        # Find the closest value in the target array.
        closest_value = target_array[np.argmin(np.abs(target_array - mean_value))]
        Z_mean[j, i] = closest_value  # Update the grid with the closest value.

        # print(mean_value, closest_value)

# # Flatten and sort in descending order.
# Z_flat = Z_mean.flatten()
# Z_sorted = np.sort(Z_flat)[::-1]  # Sort in descending order.
#
# # Output the sorted result.
# print("Flattened and sorted metallicity values:", np.unique(Z_sorted))
print(Z_mean)

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# Plot the heat map with seaborn.
plt.figure(figsize=(12, 8))
sns.heatmap(Z_mean, cmap='viridis', cbar_kws={'label': 'Z', 'pad': 0.03,}, annot=True,
            xticklabels=np.arange(tau_grid),  # tau tick labels.
            yticklabels=np.arange(R_grid),    # R tick labels.
            norm=LogNorm())

plt.xlabel(r'$\tau$ (Gyr)')
plt.ylabel(r'$R$ (kpc)')
# plt.title('Metallicity Space-Time Grid')

plt.xticks(ticks=np.arange(tau_grid+1), labels=np.arange(0, tau_grid+1))  # Set tau ticks from 0 to 12.
plt.yticks(ticks=np.arange(R_grid+1), labels=np.arange(0, R_grid+1))    # Set R ticks from 0 to 15.
# Invert the y-axis so that R increases from bottom to top.
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'figures' / 'Metallicity_Grid.pdf', dpi=300, bbox_inches='tight')
plt.show()
