import numpy as np
import matplotlib.pyplot as plt
from popkin.galaxies import MilkyWay


galaxy = MilkyWay()


# R的范围
R = np.linspace(0, 25, 1000)

# 不同的tau值
tau_values = range(0, 13)

# 使用调色板
cmap = plt.get_cmap('coolwarm', len(tau_values))

# 从 colormap 获取颜色
colors = cmap(np.linspace(0, 1, len(tau_values)))

plt.rcParams.update({
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 13,
    'legend.title_fontsize': 14,
})

max_slope_points = []  # 存储 (tau, R_max_slope, CDF_max_slope)

# 绘图
plt.figure(figsize=(8, 6))

for i, tau in enumerate(tau_values):
    # 计算CDF值
    CDF_values = galaxy.radial_cdf(R, tau)

    dR = R[1] - R[0]  # R 是均匀间隔的，所以步长固定
    PDF_values = np.gradient(CDF_values, dR)  # 计算导数（斜率）
    max_slope_idx = np.argmax(PDF_values)  # 找到最大斜率的索引
    R_max_slope = R[max_slope_idx]  # 对应的 R 值
    CDF_max_slope = CDF_values[max_slope_idx]  # 对应的 CDF 值
    max_slope_points.append((tau, R_max_slope, CDF_max_slope))
    # print(tau, R_max_slope, CDF_max_slope)

    plt.plot(R, CDF_values, label=f'τ = {tau} Gyr', color=cmap(i))

# 提取并绘制最大斜率点
max_slope_taus = [point[0] for point in max_slope_points]
max_slope_R = [point[1] for point in max_slope_points]
max_slope_CDF = [point[2] for point in max_slope_points]

# 用散点标记最大斜率点
plt.scatter(max_slope_R, max_slope_CDF, marker='o', edgecolor='black', facecolor=colors, s=30, zorder=5)

plt.xlabel(r'$R$ (kpc)')
plt.ylabel('CDF')
plt.grid(True, which="both", ls="-", alpha=0.3)
# plt.legend(fontsize=12)
plt.legend(title='Lookback Time')
plt.xlim(0, 25)
plt.ylim(0, max(galaxy.radial_cdf(25, tau) for tau in tau_values) * 1.1)

plt.tight_layout()
plt.savefig('./figures/CDF_radial.pdf', dpi=300, bbox_inches='tight')

plt.show()
