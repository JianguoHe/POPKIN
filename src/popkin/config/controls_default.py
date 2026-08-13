# controls_default.py - Default control parameters file

# ------------------------------------------------------------------------------------------------------------------
#                                              Program runtime configuration
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Evolution program type:
# - 'sse':     Single Star Evolution
# - 'bse':     Binary Star Evolution
# - 'popsin':  Single Population Synthesis
# - 'popbin':  Binary Population Synthesis
program: str = 'sse'

# Maximum number of evolution iterations
max_step: int = 20000

# Maximum evolution time [unit: Myr]
#
# Note:
#     Galactic age is 12 Gyr (12000 Myr). Systems exceeding this time will be excluded from:
#         - Position generation
#         - Number count calculation
#         - Orbit integration
max_time: float = 12000.0

# Maximum timestep for compact stars (WD/NS/BH), in Myr.
compact_star_max_timestep: float = 200.0

# Whether to use numba.jit decorator for acceleration. Recommended for binary population synthesis.
jit_enabled: bool = False

# Number of parallel processes. Only used in population synthesis.
parallel: int = 4

# ------------------------------------------------------------------------------------------------------------------
#                                             Orbit integration configuration
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Whether to enable orbital integration
enable_orbital_integration: bool = False

# Whether to include the Galactic Center supermassive black hole (Sgr A*); disabled by default
include_GC_SMBH: bool = False

# Orbit output object attributes
#
# Galactocentric (left-handed system in galpy, rotation direction aligns with Galactic rotation, Sun at [8,0,0.0208] kpc):
#   x, y, z           : cartesian coordinates
#   rho, phi          : galactocentric radial distance, azimuthal angle
#   vx, vy, vz        : velocities
#   vR, vT            : radial/tangential velocities (cylindrical)
#
# Velocity:
#   v, v_circ, v_esc  : total, circular, escape velocity
#   v_pec, v_pec_i    : peculiar, initial peculiar velocity
#
# Galactic (heliocentric, origin at Sun, +x toward GC, +z toward b=90°):
#   l, b              : longitude, latitude
#   dist              : distance from Sun
#   pm_l, pm_b        : proper motion components
#   v_radial          : radial velocity
#   helioX/Y/Z, U/V/W : coordinates and velocities
#
# ICRS (equatorial, solar system barycenter):
#   ra, dec           : right ascension, declination
#   pm_ra, pm_dec     : proper motion components
info_orbit: dict = {
    # Galactocentric Cartesian/cylindrical coordinates
    'Galactocentric': ['x', 'y', 'z', 'rho', 'phi', 'vx', 'vy', 'vz', 'vR', 'vT'],

    # Galactocentric velocity derivatives
    'velocity': ['v', 'v_circ', 'v_esc', 'v_pec', 'v_pec_i'],

    # Galactic coordinates (heliocentric)
    'Galactic': ['l', 'b', 'dist', 'pm_l', 'pm_b', 'v_radial', 'helioX', 'helioY', 'helioZ', 'U', 'V', 'W'],

    # ICRS sky coordinates (heliocentric)
    'ICRS': ['ra', 'dec', 'pm_ra', 'pm_dec'],
}

# ------------------------------------------------------------------------------------------------------------------
#                                          Gravitational wave calculation configuration
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Whether to calculate signal-to-noise ratio for potential gravitational wave sources
calculate_gw_snr: bool = False

# ------------------------------------------------------------------------------------------------------------------
#                                                    Stellar metallicity
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Metallicity for single/binary star evolution and population synthesis with constant metallicity model
Z: float = 0.02

# Metallicity model for population synthesis:
# - 'constant':    constant metallicity model (assuming uniform Galactic metallicity in both time and space)
# - 'enrichment':  enrichment model (assuming Galactic metallicity evolves with time/space)
metallicity_model: str = 'constant'

# Metallicity range for population synthesis (enrichment model)
Z_list: list = [
    0.01, 0.02, 0.03,
    0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009
]

# ------------------------------------------------------------------------------------------------------------------
#                                                 Star formation history
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Initial mass function:
# - 'kroupa1993':
# - 'kroupa2002':                 increased proportion of stars above 1 M_sun compared to kroupa1993
# - 'weisz2015':                  IMF for M31 galaxy
IMF_scheme: str = 'kroupa2002'

# Binary fraction
#
# Can be:
#     - str: predefined model name
#         - 'haaften2013': Haaften et al. (2013) model  →  f_b = 0.5 + 0.25 * log10(M)
#     - float: directly specify binary fraction in range 0-1
binary_fraction: str | float = 'haaften2013'

# Galactic age [unit: Gyr]
#
# Note:
#     Galaxy formation timescale, modification not recommended
GALACTIC_AGE: float = 12.0

# ------------------------------------------------------------------------------------------------------------------
#                           Valid parameter ranges for singles/binaries (for weight calculation)
# ------------------------------------------------------------------------------------------------------------------
# NOTE: Not recommended to modify. If modification is necessary, please edit in this page only.

# Minimum logarithmic orbital period [unit: days]
log10_P_min: float = 0.15

# Maximum logarithmic orbital period [unit: days]
log10_P_max: float = 5.5

# Power-law index for orbital period distribution in log space
log10_P_index: float = -0.55

# Minimum orbital semi-major axis [unit: R_sun]
sep_min: float = 3.0

# Maximum orbital semi-major axis [unit: R_sun]
sep_max: float = 10000.0

# ------------------------------------------------------------------------------------------------------------------
#                                          Parameters for single star evolution
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist_sse.py

# Initial stellar mass [unit: M_sun]
mass: float = 1.0

# Stellar type, range 0-14
star_type: int = 1

# Random number seed
index_sse: int = 10

# ------------------------------------------------------------------------------------------------------------------
#                                          Parameters for binary star evolution
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist_bse.py

# Initial mass of primary star [unit: M_sun]
m1: float = 10.0

# Initial mass of secondary star [unit: M_sun]
m2: float = 5.0

# Binary orbital period [unit: days]. If None, sep is used.
period: float | None = 1000

# Binary orbital semi-major axis [unit: R_sun]. If None, period is used.
sep: float | None = 1000

# Binary eccentricity, range [0, 1)
ecc: float = 0.0

# Type of primary star, range 0-14
type1: int = 1

# Type of secondary star, range 0-14
type2: int = 1

# Random number seed
index_bse: int = 10

# ------------------------------------------------------------------------------------------------------------------
#                                         Parameters for single population synthesis
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist_popsin.py

# Number of grid points for evolution (single stars)
n_grid_popsin: int = 1000

# Stellar mass range [unit: M_sun], allowed range: [0.1, 100.0]
m_range: tuple[float, float] = (0.1, 100.0)

# ------------------------------------------------------------------------------------------------------------------
#                                         Parameters for binary population synthesis
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist_popbin.py

# Number of grid points for evolution (binaries)
n_grid_popbin: int = 50

# Mass range of primary star [unit: M_sun], allowed range: [0.1, 100.0]
m1_range: tuple[float, float] = (0.1, 100.0)

# Mass range of secondary star [unit: M_sun], allowed range: [0.1, 100.0]
m2_range: tuple[float, float] = (0.1, 100.0)

# Logarithmic orbital period range [unit: days]
log10_P_range: tuple[float, float] = (0.15, 5.5)

# Orbital semi-major axis range [unit: R_sun]
sep_range: tuple[float, float] = (3.0, 10000.0)

# Initial orbit model:
# - 'hurley2002':                 initial parameter space: primary mass + secondary mass + orbital semi-major axis
# - 'sana2012':                   initial parameter space: primary mass + secondary mass + orbital period
ini_orbit_scheme: str = 'sana2012'

# Initial eccentricity distribution model:
# - 'zero':                       zero distribution, assumes initial circular orbit, no eccentricity
# - 'uniform':                    uniform distribution p(e)=1
# - 'thermal':                    thermal distribution p(e)=2e
ini_ecc_scheme: str = 'zero'

# Initial spin model:
# - 'fitting':                    initial stellar spin follows the fitting formula from Hurley2000
# - 'spin-orbit-resonance':       assumes initial spin-orbit resonance
ini_spin_scheme: str = 'fitting'

# ------------------------------------------------------------------------------------------------------------------
#                                             Evolution output configuration
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist_sse.py / inlist_bse.py / inlist_popsin.py / inlist_popbin.py

# SSE program output columns
sse_output_columns: list = [
    'time', 'type', 'mass', 'M_core', 'R', 'R_core', 'L', 'L_core',
    'M_conv_env', 'R_conv_env', 'spin', 'Teff', 'event', 'v_kick_x', 'v_kick_y', 'v_kick_z',
    'mdot', 'mdot_wind', 'mdot_mt', 'jspin', 'jdot', 'jdot_wind', 'jdot_tide', 'jdot_mt', 'jdot_mb'
]

# BSE program output columns
bse_output_columns: list = [
    'time', 'ecc', 'period', 'sep', 'type1', 'type2', 'm1', 'm2', 'mc1', 'mc2',
    'R1_div_RL1', 'R2_div_RL2', 'event', 'bound', 'state', 'v_offset_x', 'v_offset_y', 'v_offset_z',
    'v1_offset_x', 'v1_offset_y', 'v1_offset_z', 'v2_offset_x', 'v2_offset_y', 'v2_offset_z',
    'v1_kick_x', 'v1_kick_y', 'v1_kick_z', 'v2_kick_x', 'v2_kick_y', 'v2_kick_z',
    'jorb', 'jdot', 'jdot_wind', 'jdot_tide', 'jdot_mt', 'jdot_gr', 'edot', 'edot_wind', 'edot_tide', 'edot_gr',
]

# POPSIN program output columns
popsin_output_columns: list = [
    'index', 'time', 'type', 'mass', 'M_core', 'R', 'L',
    'event', 'weight', 'rate', 'dt', 'num', 'Z', 'v_kick_x', 'v_kick_y', 'v_kick_z',
    'origin', 'ini_x', 'ini_y', 'ini_z', 'ini_rho', 'ini_phi', 'ini_dist',
    'R_core', 'L_core', 'M_conv_env', 'R_conv_env', 'spin', 'Teff',
    'mdot', 'mdot_wind', 'mdot_mt', 'jspin', 'jdot', 'jdot_wind', 'jdot_tide', 'jdot_mt', 'jdot_mb'
]

# POPBIN program output columns
# - binary:             binary-related parameters
# - star1/star2:        single star-related parameters. Naming convention in output files: star1_R, star2_mdot, etc.
popbin_output_columns: dict[str, list[str]] = {
    'binary': [
        'index', 'time', 'ecc', 'period', 'sep', 'type1', 'type2', 'm1', 'm2', 'mc1', 'mc2',
        'R1_div_RL1', 'R2_div_RL2', 'event', 'bound', 'state', 'weight', 'rate', 'dt', 'num', 'Z',
        'v_offset_x', 'v_offset_y', 'v_offset_z',
        'v1_offset_x', 'v1_offset_y', 'v1_offset_z', 'v2_offset_x', 'v2_offset_y', 'v2_offset_z',
        'v1_kick_x', 'v1_kick_y', 'v1_kick_z', 'v2_kick_x', 'v2_kick_y', 'v2_kick_z',
        'origin', 'ini_x', 'ini_y', 'ini_z', 'ini_rho', 'ini_phi', 'ini_dist',
        'jorb', 'jdot', 'jdot_wind', 'jdot_tide', 'jdot_mt', 'jdot_gr',
        'edot', 'edot_wind', 'edot_tide', 'edot_gr',
    ],
    'star1': [
        'R', 'R_core', 'L', 'L_core', 'M_conv_env', 'R_conv_env',
        'spin', 'Teff', 'event', 'v_kick_x', 'v_kick_y', 'v_kick_z',
        'mdot', 'mdot_wind', 'mdot_mt', 'jspin', 'jdot', 'jdot_wind', 'jdot_tide', 'jdot_mt', 'jdot_mb',
    ],
    'star2': [
        'R', 'R_core', 'L', 'L_core', 'M_conv_env', 'R_conv_env',
        'spin', 'Teff', 'event', 'v_kick_x', 'v_kick_y', 'v_kick_z',
        'mdot', 'mdot_wind', 'mdot_mt', 'jspin', 'jdot', 'jdot_wind', 'jdot_tide', 'jdot_mt', 'jdot_mb',
    ],
}

# Simplify output (only for population synthesis)
# - popsin: keep only type changes and target sources
# - popbin: keep only type changes and target sources
simplify_output: bool = True

# Output format (only for population synthesis)
# - 'parquet': Parquet binary format (recommended, faster, smaller file size)
# - 'csv'    : CSV text format (human-readable, large file size, slower)
# - 'hdf5'   : HDF5 binary format (recommended, faster, smaller file size)
# - 'npy'    : NumPy binary format (fastest for structured arrays, minimal overhead)
output_format: str = 'parquet'

# Number of significant digits in output (CSV only)
output_precision: int = 6

# ------------------------------------------------------------------------------------------------------------------
#                                        Target objects for single population synthesis
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist_popsin.py

# Single population synthesis target object list
#
# Fields:
#   filename : output filename
#   star     : filter condition using SSE program output columns
#              (supports <, >, <=, >=, ==, !=, &, |, ~)
#   events   : event list for star
#              Format: [...]
#              Available events: 'AIC', 'ECSN', 'CCSN', 'Ia'
#
# Available stellar types for star filters:
#   Hydrogen-rich : 'MS', 'HG', 'GB', 'CHeB', 'EAGB', 'TPAGB'
#   Helium-rich   : 'HeMS', 'HeHG', 'HeGB'
#   Degenerates   : 'HeWD', 'COWD', 'ONeWD', 'NS', 'BH'
#   Massless      : 'massless'
#
# Example 1: Find all black holes
#   {
#       'filename': "BH",
#       'star': "type == 'BH'",
#       'events': []
#   }
#
# Example 2: Find all helium stars
#   {
#       'filename': "HeStar",
#       'star': "(type == 'HeMS') | (type == 'HeHG') | (type == 'HeGB')",
#       'events': []
#   }
#
# Note:
#   Multiple targets can be searched simultaneously by adding multiple dictionaries to the list.
target_popsin: list = [
    {
        'filename': "data",
        'star': "",
        'events': [],
    },
]

# ------------------------------------------------------------------------------------------------------------------
#                                          Target objects for binary population synthesis
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist_popbin.py

# Binary population synthesis target object list
#
# Fields:
#   filename : output filename
#   star1    : filter condition using SSE program output columns
#              (supports <, >, <=, >=, ==, !=, &, |, ~)
#   star2    : same as star1, for secondary star
#   binary   : filter condition using BSE program output columns
#   events   : event lists for star1/star2/binary
#              Format: {'star1': [...], 'star2': [...], 'binary': [...]}
#              Available events:
#                  star1/star2 : 'AIC', 'ECSN', 'CCSN', 'Ia'
#                  binary      : 'CE', 'RLOF begin', 'RLOF end', 'merge', 'disrupt'
#
# Available stellar types for star1/star2 filters:
#   Hydrogen-rich : 'MS', 'HG', 'GB', 'CHeB', 'EAGB', 'TPAGB'
#   Helium-rich   : 'HeMS', 'HeHG', 'HeGB'
#   Degenerates   : 'HeWD', 'COWD', 'ONeWD', 'NS', 'BH'
#   Massless      : 'massless'
#
# Available binary states for binary filters:
#   'detached', 'semidetached', 'contact', 'disrupted'
#
# Example 1: Find all black hole systems, including isolated black holes and black hole binaries
#   {
#       'filename': "BH",
#       'star1': "type == 'BH'",
#       'star2': "",
#       'binary': "",
#       'events': {}
#   }
#
# Example 2: Find bound NS-WD systems in detached state
#   {
#       'filename': "NS_WD",
#       'star1': "type == 'NS'",
#       'star2': "(type == 'HeWD') | (type == 'COWD') | (type == 'ONeWD')",
#       'binary': "bound & (state == 'detached')",
#       'events': {}
#   }
#
# Note:
#   Multiple targets can be searched simultaneously by adding multiple dictionaries to the list.

target_popbin: list = [
    {
        'filename': "data",
        'star1': "",
        'star2': "",
        'binary': "",
        'events': {
            'star1': [],
            'star2': [],
            'binary': [],
        },
    },
]

# ------------------------------------------------------------------------------------------------------------------
#                                                   Supernovae
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Core-collapse supernova remnant prescription:
# - 'fryer2012_rapid':              rapid model, Fryer et al. 2012 (doi:10.1088/0004-637X/749/1/91)
# - 'fryer2012_delayed':            delayed model, Fryer et al. 2012
# - 'mandel2020':                   stochastic model, Mandel et al. 2020 (doi:10.1093/mnras/staa3043)
# - 'maltsev2025':                  metallicity- and mass-transfer-history-dependent model, Maltsev et al. 2025
#
# Notes:
# - The Fryer-based prescriptions compute remnant masses and then classify NSs/BHs using the maximum NS mass
#   adopted by the code.
# - The Maltsev prescription contains complete-collapse BH, fallback BH, and NS branches that depend on metallicity
#   and the first pre-SN mass-transfer history.
ccsn_remnant_prescription: str = 'maltsev2025'

# Fallback fraction for the fallback-BH branch in the 'maltsev2025' remnant prescription [range: 0-1].
#
# Formula:
#     M_BH = M_proto_NS + (M_He - M_proto_NS) * f_fb
#
# In the current prescription, M_proto_NS is typically taken as 1.4 M_sun.
# This parameter is ignored unless ccsn_remnant_prescription == 'maltsev2025'.
ccsn_remnant_maltsev_fallback: float = 0.5

# Fallback-model choice for the successful-SN window in the 'maltsev2025' remnant prescription.
# - 'A': NS-guaranteed windows plus 15% fallback-BH probability outside them.
# - 'B': 10% fallback-BH probability throughout the intermediate region.
#
# Default: 'B' (matches the COMPAS implementation).
ccsn_remnant_maltsev_fallback_model: str = 'B'

# Core-collapse supernova natal-kick prescription:
# - 'zero':                         zero natal kick
# - 'maxwellian':                   Maxwellian kick model
# - 'lognormal':                    lognormal kick model
# - 'mandel2020':                   remnant-dependent kick model from Mandel et al. 2020
ccsn_kick_prescription: str = 'maxwellian'

# Maxwellian natal-kick sigma for CCSNe [unit: km/s].
#
# Default: 217 km/s, following the corrected single-peak Maxwellian adopted by
# Disberg & Mandel 2025 as a revision of the commonly used Hobbs et al. 2005 value.
ccsn_kick_maxwellian_sigma: float = 217.0

# Lognormal natal-kick parameters for CCSNe [unit: km/s].
#
# These correspond to the fiducial lognormal model of Disberg & Mandel 2025.
ccsn_kick_lognormal_mu: float = 5.60
ccsn_kick_lognormal_sigma: float = 0.68
ccsn_kick_lognormal_vmax: float | None = 1000.0

# Black-hole kick scaling applied when the remnant is a BH:
# - 'full':                         no reduction, v_BH = v_k
# - 'zero':                         direct-collapse limit, v_BH = 0
# - 'fallback':                     fallback-scaled kick, v_BH = v_k * (1 - f_fb)
#
# Notes:
# - This parameter only applies to BH remnants.
# - It is ignored when ccsn_kick_prescription == 'mandel2020', because that prescription 
#   already defines the BH kick behavior internally.
ccsn_kick_bh_scaling: str = 'fallback'

# Maxwellian natal-kick sigma for ECSNe [unit: km/s].
ecsn_kick_maxwellian_sigma: float = 30.0

# Maxwellian natal-kick sigma for neutron stars/black holes formed via AIC [unit: km/s].
aic_kick_maxwellian_sigma: float = 30.0

# ------------------------------------------------------------------------------------------------------------------
#                                                     Wind parameters
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Wind mass loss model:
# - 'hurley2000':
# - 'belczynski2010':
# - 'merritt2026':
wind_model: str = 'merritt2026'

# --------------------------------
# Wind loss related constants
# --------------------------------

# Reimers mass-loss coefficient
reimers_eta: float = 0.5

# Tidal-enhancement factor for Reimers mass loss
reimers_tidal_enhancement: float = 0.0

# Hamann-based WR mass-loss scaling factor, range 0-1
f_WR_hamann: float = 1.0

# LBV mass-loss scaling factor for the Belczynski et al. (2010) prescription
f_LBV_belczynski: float = 1.5

# --------------------------------
# Wind accretion related constants
# --------------------------------

# Bondi-Hoyle wind accretion factor (3/2)
wind_bhl_factor: float = 1.5

# Wind velocity factor: proportional to vwind**2 (1/8)
wind_velocity_scale: float = 0.125

# Transfer efficiency of specific angular momentum in wind accretion
wind_angular_momentum_efficiency: float = 1.0

# ------------------------------------------------------------------------------------------------------------------
#                                                    Magnetic braking
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Magnetic-braking prescription
# - 'rappaport1983':
# - 'hurley2002':
# - 'van2019':
magnetic_braking_prescription: str = 'hurley2002'

# Radius exponent in the Rappaport-style magnetic-braking law
magnetic_braking_radius_exponent: float = 4.0

# ------------------------------------------------------------------------------------------------------------------
#                                                Mass transfer parameters
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Mass accretion model:
# - 'rotation dependent':             rotation-dependent model, Shao & Li 2014 (doi:10.1088/0004-637X/796/1/37)
# - 'half accretion':                 half accretion model, Shao & Li 2014
# - 'thermal equilibrium':            thermal equilibrium-limited model, Shao & Li 2014
mass_accretion_model: str = 'rotation dependent'

# White dwarf super-critical accretion model:
# - 'CE-wind':                       common envelope wind model, Cui et al. 2022 (doi.org/10.1051/0004-6361/202141335)
# - 'CE':                            common envelope model
# - 'OTW':                           optically thick wind model
wd_supercritical_accretion_model: str = 'CE-wind'

# Maximum WD mass for stable mass transfer to NS/BH [unit: M_sun]. Recommended range: 0.2 - 1.25
max_wd_mass_stable_mt_to_ns_bh: float = 1.25

# Fraction of transferred material retained by a WD after nova outbursts
wd_nova_retention_fraction: float = 0.001

# Multiplicative factor applied to the Eddington accretion limit during mass transfer
mass_transfer_eddington_factor: float = 1.0

# ------------------------------------------------------------------------------------------------------------------
#                                              Common envelope parameters
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Common envelope ejection efficiency
ce_alpha: float = 1.0

# Binding energy parameter:
# - 'xl2010':                     λ from Nanjing group (Xu & Li 2010) (doi:10.1088/0004-637X/716/1/114)
# - 'wjl2016':                    λ from Wang et al. (2016) (doi: 10.1088/1674–4527/16/8/126)
ce_lambda_prescription: str = 'xl2010'

# Fraction of internal energy available to eject the envelope
#
# Formula: λ = α_th · λ_b + (1 - α_th) · λ_g
#
# Where:
# - λ_b : binding energy parameter including internal energy contribution
# - λ_g : binding energy parameter considering only gravitational energy
#
# Physical interpretation:
# - α_th = 0 : no internal energy contribution (conservative estimate)
# - α_th = 1 : all internal energy used to eject envelope (optimistic estimate)
#
# Reference: Xu & Li 2010, doi:10.1088/0004-637X/716/1/114
ce_internal_energy_fraction: float = 1.0

# Whether HG stars can survive common envelope evolution
ce_allow_hg_survival: bool = True

# ------------------------------------------------------------------------------------------------------------------
#                                                    Compact objects
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Chandrasekhar limit for white dwarfs [unit: M_sun]
M_ch: float = 1.44

# ONe core mass threshold for ECSN explosion [unit: M_sun]
M_ECSN: float = 1.38

# Maximum neutron star mass [unit: M_sun]
max_ns_mass: float = 2.5

# White dwarf cooling model:
# - True:                         use modified-Mestel cooling model
# - False:                        use standard model
wd_use_modified_mestel_cooling: bool = False
