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
# Core-collapse supernova remnant prescription ('fryer2012_rapid', 'fryer2012_delayed', 'mandel2020', 'maltsev2025').
ccsn_remnant_prescription = 'maltsev2025'

# Fallback fraction used by the Maltsev remnant prescription.
ccsn_remnant_maltsev_fallback = 0.5

# Natal-kick prescription for CCSNe ('zero', 'maxwellian', 'lognormal', 'mandel2020').
ccsn_kick_prescription = 'maxwellian'


# ==================================================================================
#                                      Stellar Wind
# ==================================================================================
# Wind mass loss model ('hurley2000', 'belczynski2010', 'merritt2026')
wind_model: str = 'merritt2026'


# ==================================================================================
#                                  Mass Transfer Parameters
# ==================================================================================
# Mass accretion model ('rotation dependent', 'half accretion', 'thermal equilibrium').
mass_accretion_model = 'rotation dependent'


# ==================================================================================
#                                  Common Envelope Parameters
# ==================================================================================

# Common-envelope ejection efficiency.
ce_alpha = 1.0

# Binding-energy parameter model ('xl2010', 'wjl2016').
ce_lambda_prescription: str = 'xl2010'

