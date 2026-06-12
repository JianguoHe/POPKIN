import json
import sys
import tracemalloc
import numpy as np
from popkin.galaxies import MilkyWay
import matplotlib.pyplot as plt


def sfr():
    galaxy = MilkyWay()

    # 创建时间网格
    tau = np.linspace(0, 12, 1000)

    # 计算各成分的 SFR
    sfr_bulge = galaxy.bulge.sfr(tau)
    sfr_thin = galaxy.thin_disk.sfr(tau)
    sfr_thick = galaxy.thick_disk.sfr(tau)

    # 计算总 SFR
    sfr_total = sfr_bulge + sfr_thin + sfr_thick

    # 绘图
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
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figures/SFR.pdf', dpi=300, bbox_inches='tight')
    plt.show()

sfr()