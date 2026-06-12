import numpy as np
from popkin.utils import imf_kroupa2002
from scipy import integrate


def average_stellar_mass() -> float:

    # 积分网格
    masses = np.linspace(0.08, 100, 100000)

    # 初始质量函数值
    imf = imf_kroupa2002(masses)

    # 双星比例
    fb = 0.5 + 0.25 * np.log10(masses)

    # 系统总质量：双星系统平均 1.5 × M，单星系统为 M
    system_mass = fb * 1.5 * masses + (1 - fb) * masses
    system_mass = (1 + 0.5 * fb) * masses

    # 数值积分
    numerator = integrate.trapezoid(imf * system_mass, masses)
    denominator = integrate.trapezoid(imf, masses)

    return numerator / denominator

result = average_stellar_mass()
print(result)
