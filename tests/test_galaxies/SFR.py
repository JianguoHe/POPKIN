import json
import sys
import tracemalloc
from pathlib import Path

import numpy as np
from popkin.galaxies import MilkyWay
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent


def sfr():
    galaxy = MilkyWay()

    # Create a time grid.
    tau = np.linspace(0, 12, 1000)

    # Compute the SFR of each component.
    sfr_bulge = galaxy.bulge.sfr(tau)
    sfr_thin = galaxy.thin_disk.sfr(tau)
    sfr_thick = galaxy.thick_disk.sfr(tau)

    # Compute the total SFR.
    sfr_total = sfr_bulge + sfr_thin + sfr_thick

    # Plot.
    plt.rcParams.update({
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
    })
    plt.figure(figsize=(8, 6))

    plt.plot(tau, sfr_bulge, label='Bulge', linewidth=2)
    plt.plot(tau, sfr_thin, label='Thin Disk', linewidth=2)
    plt.plot(tau, sfr_thick, label='Thick Disk', linewidth=2)
    plt.plot(tau, sfr_total, label='Total', linewidth=2, linestyle='--', color='black')

    plt.xlabel(r'$\tau~ (\mathrm{Gyr})$')
    plt.ylabel(r'SFR $(M_{\odot} / \mathrm{yr})$')
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', top=True, right=True,
                   labeltop=False, labelright=False, direction='in')
    plt.legend()
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / 'figures' / 'SFR.pdf', dpi=300, bbox_inches='tight')
    plt.show()

sfr()
