import numpy as np
from popkin.utils import imf_kroupa2002
from scipy import integrate


def average_stellar_mass() -> float:

    # Integration grid.
    masses = np.linspace(0.08, 100, 100000)

    # Initial mass function values.
    imf = imf_kroupa2002(masses)

    # Binary fraction.
    fb = 0.5 + 0.25 * np.log10(masses)

    # Total system mass: binary systems average 1.5 x M, while single-star systems have M.
    system_mass = fb * 1.5 * masses + (1 - fb) * masses
    system_mass = (1 + 0.5 * fb) * masses

    # Numerical integration.
    numerator = integrate.trapezoid(imf * system_mass, masses)
    denominator = integrate.trapezoid(imf, masses)

    return numerator / denominator

result = average_stellar_mass()
print(result)
