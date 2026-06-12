# inlist_bse.py - binary-star evolution configuration
# Edit this file to customize BSE parameters.
# For detailed descriptions and defaults, see src/popkin/config/controls_default.py.


# ============================================================================================================
#                                                BSE Parameters
# ============================================================================================================


m1 = 8.67                                  # Initial primary-star mass [unit: M_sun].
m2 = 4.50                                   # Initial secondary-star mass [unit: M_sun].
period = 28.85                            # Initial orbital period [unit: days]. Used before sep when both are set.

# sep = 1000.0                             # Initial orbital separation [unit: R_sun].
# ecc = 0.0                                # Initial orbital eccentricity, range [0, 1).
# type1 = 1                                # Initial type of star 1, range 0-14.
# type2 = 1                                # Initial type of star 2, range 0-14.
index_bse = 2129                           # Random seed.


# =============================================================================================================
#                                                BSE Output
# =============================================================================================================

# Output columns.
# bse_output_columns = [
#     'time', 'ecc', 'period', 'sep', 'type1', 'type2', 'm1', 'm2', 'mc1', 'mc2',
#     'R1_div_RL1', 'R2_div_RL2', 'event', 'bound', 'state', 'v_offset_x', 'v_offset_y', 'v_offset_z',
#     'v1_offset_x', 'v1_offset_y', 'v1_offset_z', 'v2_offset_x', 'v2_offset_y', 'v2_offset_z',
#     'v1_kick_x', 'v1_kick_y', 'v1_kick_z', 'v2_kick_x', 'v2_kick_y', 'v2_kick_z',
#     'jorb', 'jdot', 'jdot_wind', 'jdot_tide', 'jdot_mt', 'jdot_gr', 'edot', 'edot_wind', 'edot_tide', 'edot_gr'
# ]

# Number of significant digits retained in output.
# output_precision = 6



