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
# - 'Kroupa1993':
# - 'Kroupa2002':                 increased proportion of stars above 1 M_sun compared to Kroupa1993
# - 'Weisz2015':                  IMF for M31 galaxy
IMF_scheme: str = 'Kroupa2002'

# Binary fraction
#
# Can be:
#     - str: predefined model name
#         - 'Haaften2013': Haaften et al. (2013) model  →  f_b = 0.5 + 0.25 * log10(M)
#     - float: directly specify binary fraction in range 0-1
binary_fraction: str | float = 'Haaften2013'

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
# - 'Sana2012':                   initial parameter space: primary mass + secondary mass + orbital period
# - 'Hurley2002':                 initial parameter space: primary mass + secondary mass + orbital semi-major axis
ini_orbit_scheme: str = 'Sana2012'

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
#                                              Common envelope parameters
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Binding energy parameter:
# - 'WJL2016':                    λ from Wang et al. (2016) (doi: 10.1088/1674–4527/16/8/126)
# - 'XL2010':                     λ from Nanjing group (Xu & Li 2010) (doi:10.1088/0004-637X/716/1/114)
lambda_binding: str = 'XL2010'

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
alpha_th: float = 1.0

# Common envelope ejection efficiency
alpha_CE: float = 1.0

# Whether HG stars can survive common envelope evolution
HG_survive_CE: bool = True

# ------------------------------------------------------------------------------------------------------------------
#                                                      Supernovae
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Supernova model:
# - 'rapid':                          rapid model, Fryer et al. 2012 (doi:10.1088/0004-637X/749/1/91)
# - 'delayed':                        delayed model, Fryer et al. 2012
# - 'stochastic':                     stochastic model, Mandel et al. 2020 (doi:10.1093/mnras/staa3043)
SNtype: str = 'rapid'

# Core-collapse supernova natal-kick model:
# - 'hobbs2005': Maxwellian model, Hobbs et al. 2005
# - 'disberg2025': lognormal model, Disberg & Mandel 2025
CCSN_kick_model: str = 'hobbs2005'

# Hobbs et al. 2005 Maxwellian kick parameter [unit: km/s]
sigma_CCSN: float = 265.0

# Disberg & Mandel 2025 fiducial lognormal kick parameters.
CCSN_kick_lognormal_mu: float = 5.60
CCSN_kick_lognormal_sigma: float = 0.68
CCSN_kick_lognormal_vmax: float | None = 1000.0

# Kick velocity for neutron stars formed via ECSN
sigma_ECSN: float = 30.0

# Kick velocity for neutron stars/black holes formed via AIC
sigma_AIC: float = 30.0

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
WD_crit_accretion: str = 'CE-wind'

# Maximum WD mass for stable mass transfer to NS/BH [unit: M_sun]. Recommended range: 0.2 - 1.25
M_wd_ns_crit: float = 1.25

# Fraction of accreted material retained by WD after nova outburst
epsnov: float = 0.001

# Eddington limit factor for mass transfer
eddfac: float = 1.0

# ------------------------------------------------------------------------------------------------------------------
#                                                     Wind parameters
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Wind mass loss model:
# - 'Hurley':
# - 'Belczynski':
wind_model: str = 'Belczynski'

# --------------------------------
# Wind loss related constants
# --------------------------------

# Reimers mass loss coefficient
neta: float = 0.5

# Tidal enhancement parameter for Reimers mass loss
bwind: float = 0.0

# Helium star mass loss scaling factor, range 0-1
f_WR: float = 0.5

# LBV mass loss scaling factor
f_LBV: float = 1.5

# --------------------------------
# Wind accretion related constants
# --------------------------------

# Bondi-Hoyle wind accretion factor (3/2)
alpha_wind: float = 1.5

# Wind velocity factor: proportional to vwind**2 (1/8)
beta_wind: float = 0.125

# Transfer efficiency of specific angular momentum in wind accretion
mu_wind: float = 1.0

# ------------------------------------------------------------------------------------------------------------------
#                                                    Magnetic braking
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Magnetic braking model:
# - 'Rappaport1983':
# - 'Hurley2002':
# - 'Van2019':
mb_model: str = 'Hurley2002'

# Magnetic braking power-law index
gamma_mb: float = 4.0

# ------------------------------------------------------------------------------------------------------------------
#                                                    Compact objects
# ------------------------------------------------------------------------------------------------------------------
# User configuration file: inlist.py

# Chandrasekhar limit for white dwarfs [unit: M_sun]
M_ch: float = 1.44

# ONe core mass threshold for ECSN explosion [unit: M_sun]
M_ECSN: float = 1.38

# Maximum neutron star mass [unit: M_sun]
M_ns_max: float = 2.5

# White dwarf cooling model:
# - True:                         use modified-Mestel cooling model
# - False:                        use standard model
WD_flag: bool = False
