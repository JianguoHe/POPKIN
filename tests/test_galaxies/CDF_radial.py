from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from popkin.galaxies import MilkyWay

SCRIPT_DIR = Path(__file__).resolve().parent


galaxy = MilkyWay()


# R range.
R = np.linspace(0, 25, 1000)

# Different tau values.
tau_values = range(0, 13)

# Use a color palette.
cmap = plt.get_cmap('coolwarm', len(tau_values))

# Get colors from the colormap.
colors = cmap(np.linspace(0, 1, len(tau_values)))

plt.rcParams.update({
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 13,
    'legend.title_fontsize': 14,
})

max_slope_points = []  # Store (tau, R_max_slope, CDF_max_slope).

# Plot.
plt.figure(figsize=(8, 6))

for i, tau in enumerate(tau_values):
    # Compute CDF values.
    CDF_values = galaxy.radial_cdf(R, tau)

    dR = R[1] - R[0]  # R is uniformly spaced, so the step size is fixed.
    PDF_values = np.gradient(CDF_values, dR)  # Compute the derivative (slope).
    max_slope_idx = np.argmax(PDF_values)  # Find the index of the maximum slope.
    R_max_slope = R[max_slope_idx]  # Corresponding R value.
    CDF_max_slope = CDF_values[max_slope_idx]  # Corresponding CDF value.
    max_slope_points.append((tau, R_max_slope, CDF_max_slope))
    # print(tau, R_max_slope, CDF_max_slope)

    plt.plot(R, CDF_values, label=f'τ = {tau} Gyr', color=cmap(i))

# Extract and plot the maximum-slope points.
max_slope_taus = [point[0] for point in max_slope_points]
max_slope_R = [point[1] for point in max_slope_points]
max_slope_CDF = [point[2] for point in max_slope_points]

# Mark maximum-slope points with scatter points.
plt.scatter(max_slope_R, max_slope_CDF, marker='o', edgecolor='black', facecolor=colors, s=30, zorder=5)

plt.xlabel(r'$R$ (kpc)')
plt.ylabel('CDF')
plt.grid(True, which="both", ls="-", alpha=0.3)
ax = plt.gca()
ax.tick_params(axis='both', which='both', top=True, right=True,
               labeltop=False, labelright=False, direction='in')
# plt.legend(fontsize=12)
plt.legend(title='Lookback Time')
plt.xlim(0, 25)
plt.ylim(0, max(galaxy.radial_cdf(25, tau) for tau in tau_values) * 1.1)

plt.tight_layout()
plt.savefig(SCRIPT_DIR / 'figures' / 'CDF_radial.pdf', dpi=300, bbox_inches='tight')

plt.show()
