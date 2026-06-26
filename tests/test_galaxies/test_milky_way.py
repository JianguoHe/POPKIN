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
    # 创建银河系实例
    galaxy = MilkyWay()

    print("=" * 50)
    print(" " * 15, "银河系模型演示")
    print("=" * 50)

    # 1. 银河系基本信息
    print("\n1. 银河系基本信息:")
    galaxy.info(pretty_print=True)

    # 2. 恒星形成率
    print("\n2. 恒星形成率:")
    tau = 7.0
    print(f"在 tau={tau} Gyr 时:")
    print(f"  薄盘: {galaxy.thin_disk.sfr(tau):.3f} Msun/yr")
    print(f"  厚盘: {galaxy.thick_disk.sfr(tau):.3f} Msun/yr")
    print(f"  核球: {galaxy.bulge.sfr(tau):.3f} Msun/yr")
    print(f"  总计: {galaxy.sfr(tau):.3f} Msun/yr")

    # 3. 径向分布
    print("\n3. 径向累计分布:")
    R = 20.0  # kpc
    print(f"在 R={R} kpc 处:")
    print(f"  薄盘分布: {galaxy.thin_disk.radial_cdf(R, tau=0):.4f}")
    print(f"  厚盘分布: {galaxy.thick_disk.radial_cdf(R):.4f}")
    print(f"  核球分布: {galaxy.bulge.radial_cdf(R):.4f}")
    print(f"  总分布: {galaxy.radial_cdf(R, tau=0):.4f}")

    # 4. ISM信息
    print("\n4. 星际介质信息:")
    print(f"R=8 kpc 处:")
    print(f"  分子云面密度: {galaxy.molecular_clouds.surface_density(8):.3f} Msun/pc^2")
    print(f"  HI总面密度: {galaxy.cold_hi.total_surface_density(8):.3f} Msun/pc^2")
    print(f"  冷HI面密度: {galaxy.cold_hi.surface_density(8):.3f} Msun/pc^2")
    print(f"  暖HI面密度: {galaxy.warm_hi.surface_density(8):.3f} Msun/pc^2")
    print(f"  分子云填充因子: {galaxy.molecular_clouds.filling_fraction(8):.4f}")
    print(f"  冷HI填充因子: {galaxy.cold_hi.filling_fraction(8):.4f}")
    print(f"  暖HI填充因子: {galaxy.warm_hi.filling_fraction(8):.4f}")

    # 5. 生成恒星
    print("\n6. 生成恒星:")
    tau_single = 3
    tau_mutil = np.array([3.2, 4.3, 12])
    stars = galaxy.generate_star(tau=tau_mutil)
    print(json.dumps(stars, indent=4, ensure_ascii=False))
    print(f"  这批恒星的银心径向距离R为: {np.array([star['R'] for star in stars])}")



def pdf():
    galaxy = MilkyWay(metallicity_model='constant', Z=0.01)
    tau = np.array([-2, 0, 2, 4, 6, 8, 10, 12, 13])
    stars = galaxy.generate_star(tau, weight=1e-6)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)  # 不换行
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
    # print(f"耗时: {time_original:.4f} 秒)")


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
