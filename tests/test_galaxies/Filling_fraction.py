import numpy as np
from popkin.galaxies import MilkyWay
import matplotlib.pyplot as plt


def filling_fraction():
    plt.rcParams.update({
        'axes.labelsize': 25,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 22,
        'legend.title_fontsize': 22,
    })

    galaxy = MilkyWay()

    r_values = np.linspace(0, 30, 1000)

    filling_fractions_H2 = galaxy.MCs.filling_fraction(R=r_values)
    filling_fractions_cold_HI = galaxy.cold_HI.filling_fraction(R=r_values)
    filling_fractions_warm_HI = galaxy.warm_HI.filling_fraction(R=r_values)
    filling_fractions_warm_HII = galaxy.warm_HII.filling_fraction(R=r_values)
    filling_fractions_hot_HII = galaxy.hot_HII.filling_fraction(R=r_values)

    # 绘制填充因子图
    plt.figure(figsize=(10, 7))
    plt.plot(r_values, filling_fractions_H2, label='H2', color='#1f77b4', linewidth=2)
    plt.plot(r_values, filling_fractions_cold_HI, label='cold HI', color='#ff7f0e', linewidth=2)
    plt.plot(r_values, filling_fractions_warm_HI, label='warm HI', color='#2ca02c', linewidth=2)
    plt.plot(r_values, filling_fractions_warm_HII, label='warm HII', color='#d62728', linewidth=2)
    plt.plot(r_values, filling_fractions_hot_HII, label='hot HII', color='#9467bd', linewidth=2)

    plt.xlabel(r'$R ~ \rm(kpc)$')
    plt.ylabel(r'$f_v~(z=0)$')
    plt.xlim(0, 20)
    plt.yscale('log')
    # plt.grid(color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./figures/filling_fraction.pdf', dpi=300)
    plt.show()

filling_fraction()








