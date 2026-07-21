import json
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from popkin.galaxies import MilkyWay
import matplotlib.pyplot as plt
import seaborn as sns
import time

SCRIPT_DIR = Path(__file__).resolve().parent




def use_example():
    # Create a Milky Way instance.
    galaxy = MilkyWay()

    print("=" * 50)
    print(" " * 15, "Milky Way model demo")
    print("=" * 50)

    # 1. Basic Milky Way information.
    print("\n1. Basic Milky Way information:")
    galaxy.info(pretty_print=True)

    # 2. Star formation rate.
    print("\n2. Star formation rate:")
    tau = 7.0
    print(f"At tau={tau} Gyr:")
    print(f"  Thin disk: {galaxy.thin_disk.sfr(tau):.3f} Msun/yr")
    print(f"  Thick disk: {galaxy.thick_disk.sfr(tau):.3f} Msun/yr")
    print(f"  Bulge: {galaxy.bulge.sfr(tau):.3f} Msun/yr")
    print(f"  Total: {galaxy.sfr(tau):.3f} Msun/yr")

    # 3. Radial distribution.
    print("\n3. Radial cumulative distribution:")
    R = 20.0  # kpc
    print(f"At R={R} kpc:")
    print(f"  Thin-disk distribution: {galaxy.thin_disk.radial_cdf(R, tau=0):.4f}")
    print(f"  Thick-disk distribution: {galaxy.thick_disk.radial_cdf(R):.4f}")
    print(f"  Bulge distribution: {galaxy.bulge.radial_cdf(R):.4f}")
    print(f"  Total distribution: {galaxy.radial_cdf(R, tau=0):.4f}")

    # 4. ISM information.
    print("\n4. ISM information:")
    print(f"At R=8 kpc:")
    print(f"  Molecular-cloud surface density: {galaxy.molecular_clouds.surface_density(8):.3f} Msun/pc^2")
    print(f"  Total HI surface density: {galaxy.cold_hi.total_surface_density(8):.3f} Msun/pc^2")
    print(f"  Cold HI surface density: {galaxy.cold_hi.surface_density(8):.3f} Msun/pc^2")
    print(f"  Warm HI surface density: {galaxy.warm_hi.surface_density(8):.3f} Msun/pc^2")
    print(f"  Molecular-cloud filling factor: {galaxy.molecular_clouds.filling_fraction(8):.4f}")
    print(f"  Cold HI filling factor: {galaxy.cold_hi.filling_fraction(8):.4f}")
    print(f"  Warm HI filling factor: {galaxy.warm_hi.filling_fraction(8):.4f}")

    # 5. Generate stars.
    print("\n6. Generate stars:")
    tau_single = 3
    tau_mutil = np.array([3.2, 4.3, 12])
    stars = galaxy.generate_star(tau=tau_mutil)
    print(json.dumps(stars, indent=4, ensure_ascii=False))
    print(f"  Galactocentric radial distances R for these stars: {np.array([star['R'] for star in stars])}")



def pdf():
    galaxy = MilkyWay(metallicity_model='constant', Z=0.01)
    tau = np.array([-2, 0, 2, 4, 6, 8, 10, 12, 13])
    stars = galaxy.generate_star(tau, weight=1e-6)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)  # Do not wrap lines.
    stars = pd.DataFrame(stars)
    print(stars)
    stars.to_csv(SCRIPT_DIR / 'stars.csv', index=False)
    R = stars['ini_rho']
    z = stars['ini_z']
    plt.figure(figsize=(10, 6))
    sns.histplot(R, bins=30, ax=plt.gca())

    # plt.show()

    # start = time.time()
    # for _ in range(1000):
    #     tau = np.repeat(8.1, 2000)
    #     # tau = np.array([0, 2, 9])
    #     stars = galaxy.generate_star(tau, weight=1e-6)
    # time_original = time.time() - start
    # print(f"Elapsed time: {time_original:.4f} s)")


def test_ISM():
    galaxy = MilkyWay()
    MCs = galaxy.molecular_clouds
    cold_HI = galaxy.cold_hi
    warm_HI = galaxy.warm_hi
    warm_HII = galaxy.warm_hii
    hot_HII = galaxy.hot_hii
    print(MCs.n, MCs.mu, MCs.cs(200))
    print(cold_HI.n, cold_HI.mu, cold_HI.cs)
    print(warm_HI.n, warm_HI.mu, warm_HI.cs)
    print(warm_HII.n, warm_HII.mu, warm_HII.cs)
    print(hot_HII.n, hot_HII.mu, hot_HII.cs)
    n_MCs, w_MCs = galaxy.molecular_clouds.get_discrete_number_density(n_points=10)
    print(n_MCs)
    print(w_MCs)
    print(w_MCs.sum())
    n_coldHI, w_coldHI = galaxy.cold_hi.get_discrete_number_density(n_points=10)
    print(n_coldHI)
    print(w_coldHI)
    print(w_coldHI.sum())


if __name__ == "__main__":
    # use_example()
    pdf()
    # test_ISM()

    pass
