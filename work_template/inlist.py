# inlist.py - user configuration file
# Edit this file to customize global POPKIN parameters.
# For detailed descriptions and defaults, see src/popkin/config/controls_default.py.


# ==================================================================================
#                                    Evolution Parameters
# ==================================================================================

# Evolution program:
#   - 'sse':     single-star evolution
#   - 'bse':     binary-star evolution
#   - 'popsin':  single-star population synthesis
#   - 'popbin':  binary-star population synthesis
program = 'sse'

# Whether to enable numba JIT acceleration. Recommended mainly for popbin.
jit_enabled = False

# Number of worker processes. Used only by population synthesis drivers.
parallel = 4

# Whether to enable Galactic orbit integration.
enable_orbital_integration = False

# Whether to calculate gravitational-wave SNR for double compact objects.
calculate_gw_snr = False


# ==================================================================================
#                                    Metallicity Parameters
# ==================================================================================

# Metallicity value [unit: Z_sun].
Z = 0.02

# Metallicity model:
#   - 'constant':   fixed metallicity
#   - 'enrichment': enrichment metallicity list
metallicity_model = 'constant'

# Metallicity values used by the enrichment model.
Z_list = [
    # 0.03,
    0.01, 0.02, 0.03,
    0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009
]


# ==================================================================================
#                                      Supernova
# ==================================================================================
# Supernova model ('rapid', 'delayed', 'stochastic').
SNtype = 'rapid'


# ==================================================================================
#                                  Mass Transfer Parameters
# ==================================================================================
# Mass accretion model ('rotation dependent', 'half accretion', 'thermal equilibrium').
mass_accretion_model = 'rotation dependent'


# ==================================================================================
#                                  Common Envelope Parameters
# ==================================================================================

# Common-envelope ejection efficiency.
alpha_CE = 1.0

# Binding-energy parameter model ('XL2010', 'WJL2016').
# lambda_binding: str = 'WJL2016'




