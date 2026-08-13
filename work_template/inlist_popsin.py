# inlist_popsin.py - single-star population synthesis configuration
# Edit this file to customize popsin parameters.
# For detailed descriptions and defaults, see src/popkin/config/controls_default.py.


# ============================================================================================================
#                                              Popsin Parameters
# ============================================================================================================

# Number of single-star mass grid points.
n_grid_popsin = 1000

# Stellar mass range [unit: M_sun].
m_range = (0.1, 100.0)


# ============================================================================================================
#                                              Popsin Targets
# ============================================================================================================

# Target source list.
target_popsin = [
    {
      'filename': "BH",
      'star': "type == 'BH'",
      'events': []
    },
]

# Available target types:
#   Hydrogen-rich stars: MS, HG, GB, CHeB, EAGB, TPAGB
#   Helium-rich stars: HeMS, HeHG, HeGB
#   Compact objects: HeWD, COWD, ONeWD, NS, BH
#   Massless remnant: massless


# =============================================================================================================
#                                               Popsin Output
# =============================================================================================================

# popsin_data_dir = '~/'

popsin_output_columns = [
    'index', 'time', 'type', 'mass', 'M_core', 'R', 'L',
    'event', 'weight', 'rate', 'dt', 'num', 'Z', 'v_kick_x', 'v_kick_y', 'v_kick_z',
    'origin', 'ini_x', 'ini_y', 'ini_z', 'ini_rho', 'ini_phi', 'ini_dist',
]

# Whether to simplify output.
# simplify_output = True

# Output format: ['parquet', 'csv', 'hdf5', 'npy'].
# output_format = 'parquet'

# Number of significant digits retained in output.
# output_precision = 6
