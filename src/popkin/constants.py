import numpy as np


# Distance units.
km: float = 1e5                                # Kilometers to centimeters.
au: float = 1.49597870700e13                   # Astronomical units to centimeters.
au_R_sun: float = 215.029448208                # Astronomical units to solar radii.
pc: float = 3.08567758e18                      # Parsecs to centimeters.
kpc: float = 3.08567758e21                     # Kiloparsecs to centimeters.

# Time-unit conversions.
day_per_year: float = 365.25                   # Years to days.
sec_per_year: float = 3.15576e7                # Years to seconds.
sec_per_day: float = 86400                     # Days to seconds.

# Solar parameters.
M_sun: float = 1.9884e33                       # Solar mass (g).
R_sun: float = 6.957e10                        # Solar radius (cm).
L_sun: float = 3.83e33                         # Solar luminosity (erg/s).
Z_sun: float = 0.02                            # Solar metallicity.
T_eff_sun: float = 5780.0                      # Solar effective temperature (K).

# Physical constants.
G: float = 6.6743e-8                           # Gravitational constant (cm^3 g^-1 s^-2).
c_light: float = 2.99792458e10                 # Speed of light (cm/s).
m_p: float = 1.67262192595e-24                 # Proton mass (g).

# Kepler-law conversion constants.
period_to_sep: float = 215.029448208           # Orbital period (yr) to separation (R_sun).
sep_to_period: float = 0.00031714151           # Separation (R_sun) to orbital period (yr).



# Numeric tolerances.
tol = 1e-7
tiny = 1e-14


# Nuclear-burning efficiency constants.
ahe = 4
aco = 16


# Stellar-type mapping.
type_mapping = np.array([
    'MS',      # 0
    'MS',      # 1
    'HG',      # 2
    'GB',      # 3
    'CHeB',    # 4
    'EAGB',    # 5
    'TPAGB',   # 6
    'HeMS',    # 7
    'HeHG',    # 8
    'HeGB',    # 9
    'HeWD',    # 10
    'COWD',    # 11
    'ONeWD',   # 12
    'NS',      # 13
    'BH',      # 14
    'massless' # 15
], dtype='U8')


# Structured-array dtype for binary-star evolution.
struct_dtype_binary = np.dtype([
    ('time', np.float64),
    ('ecc', np.float64),
    ('period', np.float64),
    ('sep', np.float64),
    ('type1', np.int32),
    ('type2', np.int32),
    ('m1', np.float64),
    ('m2', np.float64),
    ('mc1', np.float64),
    ('mc2', np.float64),
    ('R1_div_RL1', np.float64),
    ('R2_div_RL2', np.float64),
    ('jorb', np.float64),
    ('jdot', np.float64),
    ('jdot_wind', np.float64),
    ('jdot_tide', np.float64),
    ('jdot_mt', np.float64),
    ('jdot_gr', np.float64),
    ('edot', np.float64),
    ('edot_wind', np.float64),
    ('edot_tide', np.float64),
    ('edot_gr', np.float64),
    ('v_offset_x', np.float64),
    ('v_offset_y', np.float64),
    ('v_offset_z', np.float64),
    ('v1_offset_x', np.float64),
    ('v1_offset_y', np.float64),
    ('v1_offset_z', np.float64),
    ('v2_offset_x', np.float64),
    ('v2_offset_y', np.float64),
    ('v2_offset_z', np.float64),
    ('event', 'S20'),
    ('state', 'S20')
])


# Structured-array dtype for single-star evolution.
struct_dtype_single = np.dtype([
    ('time', np.float64),
    ('type', np.int32),
    ('mass', np.float64),
    ('M_core', np.float64),
    ('M_conv_env', np.float64),
    ('R', np.float64),
    ('R_core', np.float64),
    ('R_conv_env', np.float64),
    ('L', np.float64),
    ('L_core', np.float64),
    ('spin', np.float64),
    ('Teff', np.float64),
    ('mdot', np.float64),
    ('mdot_wind', np.float64),
    ('mdot_mt', np.float64),
    ('jspin', np.float64),
    ('jdot', np.float64),
    ('jdot_wind', np.float64),
    ('jdot_tide', np.float64),
    ('jdot_mt', np.float64),
    ('jdot_mb', np.float64),
    ('v_kick_x', np.float64),
    ('v_kick_y', np.float64),
    ('v_kick_z', np.float64),
    ('event', 'S20'),
])
