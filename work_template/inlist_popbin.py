# inlist_popbin.py - binary-star population synthesis configuration
# Edit this file to customize popbin parameters.
# For detailed descriptions and defaults, see src/popkin/config/controls_default.py.

# ============================================================================================================
#                                               Popbin Parameters
# ============================================================================================================

# Number of grid points per initial-parameter dimension.
n_grid_popbin = 50

# Primary-star mass range [unit: M_sun].
m1_range = (5, 100.0)

# Secondary-star mass range [unit: M_sun].
m2_range = (0.1, 100.0)

# Log10 orbital-period range [unit: days].
# log10_P_range = (0.15, 5.5)

# Orbital-separation range [unit: R_sun].
# sep_range = (3.0, 10000.0)

# Initial orbit model. Options: 'Sana2012', 'Hurley2002'.
# ini_orbit_scheme = 'Sana2012'

# Initial eccentricity distribution. Options: 'zero', 'uniform', 'thermal'.
# ini_ecc_scheme = 'zero'

# Initial spin model. Options: 'fitting', 'spin-orbit-resonance'.
# ini_spin_scheme = 'fitting'


# ============================================================================================================
#                                               Popbin Targets
# ============================================================================================================

target_popbin = [
    # Select all black-hole systems, including isolated black holes and BH binaries.
    {
      'filename': "BH",
      'star1': "type == 'BH'",
      'star2': "",
      'binary': "",
      'events': {}
    },
]


# Available target types:
#   Hydrogen-rich stars: MS, HG, GB, CHeB, EAGB, TPAGB
#   Helium-rich stars: HeMS, HeHG, HeGB
#   Compact objects: HeWD, COWD, ONeWD, NS, BH
#   Massless remnant: massless


# =============================================================================================================
#                                               Popbin Output
# =============================================================================================================

# popbin_data_dir = '~/'

popbin_output_columns = {
    'binary': [
        'index', 'time', 'ecc', 'period', 'sep', 'type1', 'type2', 'm1', 'm2', 'mc1', 'mc2',
        'R1_div_RL1', 'R2_div_RL2', 'event', 'bound', 'state', 'weight', 'rate', 'dt', 'num', 'Z',
        'v_offset_x', 'v_offset_y', 'v_offset_z',
        'v1_offset_x', 'v1_offset_y', 'v1_offset_z', 'v2_offset_x', 'v2_offset_y', 'v2_offset_z',
        'v1_kick_x', 'v1_kick_y', 'v1_kick_z', 'v2_kick_x', 'v2_kick_y', 'v2_kick_z',
        'origin', 'ini_x', 'ini_y', 'ini_z', 'ini_rho', 'ini_phi', 'ini_dist',
    ],
    'star1': [],
    'star2': [],
}

# Whether to simplify output.
# simplify_output = True

# Output format: ['parquet', 'csv', 'hdf5', 'npy'].
# output_format = 'parquet'

# Number of significant digits retained in output.
# output_precision = 6








