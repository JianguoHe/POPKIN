"""Shared internal utilities for POPKIN drivers and model helpers."""

import os
import fcntl
import numpy as np
from numba import njit
from numba.experimental import jitclass
from scipy import integrate
from popkin.constants import type_mapping
from popkin.config.logger import get_logger
from popkin.config.controls_default import jit_enabled as _default_jit_enabled
from popkin.config.controls_default import log10_P_min, log10_P_max, log10_P_index, sep_min, sep_max
from popkin.config.controls_default import IMF_scheme, binary_fraction
from popkin.config.controls_default import ini_orbit_scheme, ini_ecc_scheme
from popkin.config.controls_default import m_range, n_grid_popsin
from popkin.config.controls_default import m1_range, m2_range, log10_P_range, sep_range, n_grid_popbin


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# JIT configuration helpers
# ---------------------------------------------------------------------------


def _load_jit_enabled():
    try:
        import inlist  # type: ignore
    except ModuleNotFoundError as e:
        if e.name != "inlist":
            raise
        return _default_jit_enabled

    return bool(getattr(inlist, "jit_enabled", _default_jit_enabled))


jit_enabled = _load_jit_enabled()


def conditional_njit(func):
    """Apply njit only when jit_enabled is true."""
    return njit(func) if jit_enabled else func


def conditional_jitclass(spec):
    """Apply jitclass only when jit_enabled is true."""
    def decorator(cls):
        return jitclass(spec)(cls) if jit_enabled else cls

    return decorator


# ---------------------------------------------------------------------------
# Evolution data conversion
# ---------------------------------------------------------------------------


def process_single_star_data(star):
    """Process single star evolution data

    Args:
        star: Single star instance

    Returns:
        Structured array
    """
    star_data = star.data

    # Build new dtype
    exclude_fields = {'event', 'type'}
    new_dtype_fields = []
    for name in star_data.dtype.names:
        if name not in exclude_fields:
            new_dtype_fields.append((name, star_data.dtype[name]))

    # Add new fields
    new_dtype_fields.extend([
        ('event', 'U16'),
        ('type', 'U8'),
    ])

    # Create result array
    result = np.zeros(len(star_data), dtype=new_dtype_fields)
    for name in star_data.dtype.names:
        if name not in exclude_fields:
            result[name] = star_data[name]

    # Process events
    events = star_data['event']
    mask_event = events != b'None'
    result['event'][mask_event] = events[mask_event].astype('U16')

    # Process type mapping
    result['type'] = type_mapping[star_data['type']]

    return result


def process_binary_star_data(binary, star1, star2):
    """Process binary star evolution data

    Args:
        binary: Binary instance
        star1: Primary star instance
        star2: Secondary star instance

    Returns:
        Structured array
    """
    binary_data = binary.data
    star1_data = star1.data
    star2_data = star2.data

    # Build new dtype
    exclude_fields = {'event', 'state', 'type1', 'type2'}
    new_dtype_fields = []

    for name in binary_data.dtype.names:
        if name not in exclude_fields:
            new_dtype_fields.append((name, binary_data.dtype[name]))

    # Add new fields
    new_dtype_fields.extend([
        ('v1_kick_x', 'f8'),
        ('v1_kick_y', 'f8'),
        ('v1_kick_z', 'f8'),
        ('v2_kick_x', 'f8'),
        ('v2_kick_y', 'f8'),
        ('v2_kick_z', 'f8'),
        ('event', 'U16'),
        ('state', 'U16'),
        ('type1', 'U8'),
        ('type2', 'U8'),
        ('bound', 'bool'),
    ])

    # Create result array
    result = np.zeros(len(binary_data), dtype=new_dtype_fields)
    for name in binary_data.dtype.names:
        if name not in exclude_fields:
            result[name] = binary_data[name]

    # Copy kick velocity vectors (v1_kick/v2_kick)
    result['v1_kick_x'] = star1_data['v_kick_x']
    result['v1_kick_y'] = star1_data['v_kick_y']
    result['v1_kick_z'] = star1_data['v_kick_z']

    result['v2_kick_x'] = star2_data['v_kick_x']
    result['v2_kick_y'] = star2_data['v_kick_y']
    result['v2_kick_z'] = star2_data['v_kick_z']

    # Process events
    events = binary_data['event']
    mask_event = events != b'None'
    result['event'][mask_event] = events[mask_event].astype('U16')

    # Process state
    result['bound'] = (result['ecc'] >= 0) & (result['ecc'] < 1)
    result['state'] = binary_data['state'].astype('U16')

    # Process type mapping
    result['type1'] = type_mapping[binary_data['type1']]
    result['type2'] = type_mapping[binary_data['type2']]

    return result


# ---------------------------------------------------------------------------
# Structured-array helpers
# ---------------------------------------------------------------------------


def merge_structured_data(arrays):
    """Fast merge of multiple structured arrays

    Args:
        arrays: List of structured arrays, e.g., [structured_array_1, structured_array_2, ...]

    Returns:
        Merged structured array
    """
    if not arrays:
        raise ValueError("merge_structured_data requires at least one array")

    n = len(arrays[0])
    if any(len(arr) != n for arr in arrays):
        raise ValueError("all arrays must have the same length")

    new_dtype = []
    seen = set()

    for arr in arrays:
        for name in arr.dtype.names:
            if name in seen:
                raise ValueError(f"duplicate field name: {name}")
            seen.add(name)

            field_dtype = arr.dtype.fields[name][0]
            if field_dtype.shape:
                new_dtype.append((name, field_dtype.base, field_dtype.shape))
            else:
                new_dtype.append((name, field_dtype))

    result = np.empty(n, dtype=new_dtype)

    for arr in arrays:
        for name in arr.dtype.names:
            result[name] = arr[name]

    return result


def select_structured_data(arr, cols, mask=None):
    """Select fields from a structured array, optionally filtering rows by mask."""
    dtype = [(col, arr.dtype[col]) for col in cols]

    if mask is None:
        out = np.empty(len(arr), dtype=dtype)
        for col in cols:
            out[col] = arr[col]
        return out

    idx = np.flatnonzero(mask)
    out = np.empty(len(idx), dtype=dtype)

    for col in cols:
        np.take(arr[col], idx, axis=0, out=out[col])

    return out


# ---------------------------------------------------------------------------
# Population parameter spaces and weights
# ---------------------------------------------------------------------------


def create_popsin_parameter_space(
        m_range: tuple[float, float] = m_range,
        n_grid_popsin: int = n_grid_popsin,
        IMF_scheme: str = IMF_scheme,
        binary_fraction: float | str = binary_fraction,
) -> np.ndarray:
    """Generate the mass parameter space for single star population synthesis

    Sample stellar masses uniformly in logarithmic space.

    Args:
        m_range: Stellar mass range [unit: M_sun], allowed range: [0.1, 100.0]
        n_grid_popsin: Number of grid points in logarithmic mass space
        IMF_scheme: IMF model name. Must be one of: 'Kroupa2002', 'Kroupa1993', or 'Weisz2015'
        binary_fraction: Binary fraction. Can be a float in [0, 1] or 'Haaften2013'

    Returns:
        2D array with shape (n_grid_popsin, 2)
        Columns: mass, weight_single
    """
    logger.info(f"Generating parameter space for popsin: n_grid={n_grid_popsin}, m_range={m_range}",
                extra={"console": True})

    # Generate mass grid
    m_values = np.logspace(
        np.log(m_range[0]), np.log(m_range[1]), num=n_grid_popsin, endpoint=True, base=np.e
    )

    # Pre-compute single star weights
    weights = weight_single(
        M=m_values,
        m_range=m_range,
        n_grid_popsin=n_grid_popsin,
        IMF_scheme=IMF_scheme,
        binary_fraction=binary_fraction,
    )

    # Combine mass and weight into a 2D array
    result = np.column_stack((m_values, weights))

    logger.info(f"Parameter space for popsin generation complete: {len(result):,} points, shape {result.shape}")

    return result


def weight_single(
        M: float | np.ndarray,
        m_range: tuple[float, float] = m_range,
        n_grid_popsin: int = n_grid_popsin,
        IMF_scheme: str = IMF_scheme,
        binary_fraction: float | str = binary_fraction,
) -> float | np.ndarray:
    """Calculate the weight contribution of single stars to the stellar population

    Weight formula for primordial single stars
        W_single(M) = M × ξ(M) × ΔlnM × (1 - f_binary(M))

    where:
        - M × ξ(M): Density in logarithmic mass space (Φ_lnM)
        - ΔlnM: Logarithmic mass interval
        - 1 - f_binary: Fraction of single stars

    Args:
        M: Stellar mass [unit: M_sun]
        m_range: Mass range for grid calculation (min_m, max_m)
        n_grid_popsin: Number of grid points in logarithmic mass space
        IMF_scheme: IMF model name. Must be one of: 'Kroupa2002', 'Kroupa1993', or 'Weisz2015'
        binary_fraction: Binary fraction. Can be a float in [0, 1] or 'Haaften2013'

    Returns:
        Weight value(s) for single stars

    Raises:
        ValueError: When IMF_scheme or binary_fraction is not supported
    """
    # Density in logarithmic mass space
    Phi_lnM = M * initial_mass_function(M, IMF_scheme)

    # Logarithmic mass interval
    delta_lnM = np.log(m_range[1] / m_range[0]) / (n_grid_popsin - 1)

    # Single star weight
    weight = Phi_lnM * delta_lnM * (1 - frac_binary(M, binary_fraction))

    return weight


def create_popbin_parameter_space(
        m1_range: tuple[float, float] = m1_range,
        m2_range: tuple[float, float] = m2_range,
        orbit_param_range: tuple[float, float] | None = None,
        n_grid_popbin: int = n_grid_popbin,
        ini_orbit_scheme: str = ini_orbit_scheme,
        ini_ecc_scheme: str = ini_ecc_scheme,
        IMF_scheme: str = IMF_scheme,
        binary_fraction: float | str = binary_fraction,
        random_seed: int = 1
) -> np.ndarray:
    """Generate the initial parameter space for binary star population synthesis

    Generates a 5D parameter grid containing:
        - m1: Primary star mass [M_sun]
        - m2: Secondary star mass [M_sun]
        - orbit: Orbital parameter (period or semi-major axis)
        - ecc: Eccentricity
        - weight: Binary system weight

    Args:
        m1_range: Primary star mass range [unit: M_sun], allowed range: [0.1, 100.0]
        m2_range: Secondary star mass range [unit: M_sun], allowed range: [0.1, 100.0]
        orbit_param_range: Orbital period range [unit: days] (Sana2012) or semi-major axis range
            [unit: R_sun] (Hurley2002). If None, the default range is inferred from ini_orbit_scheme.
        n_grid_popbin: Number of grid points for each parameter in logarithmic space
        ini_orbit_scheme: Initial orbit model. Must be one of: 'Sana2012', 'Hurley2002'
        ini_ecc_scheme: Initial eccentricity distribution. Must be one of: 'zero', 'uniform', 'thermal'
        IMF_scheme: IMF model name. Must be one of: 'Kroupa2002', 'Kroupa1993', or 'Weisz2015'
        binary_fraction: Binary fraction. Can be a float in [0, 1] or 'Haaften2013'
        random_seed: Random seed for eccentricity sampling, defaults to 1

    Returns:
        Parameter space array with shape (n_grid_popbin³, 5)
        Column order: [m1, m2, orbit, ecc, weight]

    Raises:
        ValueError: When ini_orbit_scheme or ini_ecc_scheme is not supported
    """
    logger.info(f"Generating parameter space for popbin: n_grid={n_grid_popbin}, m1_range={m1_range}, m2_range={m2_range}",
                extra={"console": True})

    # 1. Generate mass grids
    m1_values = np.logspace(
        np.log(m1_range[0]), np.log(m1_range[1]), num=n_grid_popbin, endpoint=True, base=np.e
    )
    m2_values = np.logspace(
        np.log(m2_range[0]), np.log(m2_range[1]), num=n_grid_popbin, endpoint=True, base=np.e
    )

    # 2. Generate orbital parameter grid
    if orbit_param_range is None:
        if ini_orbit_scheme == 'Sana2012':
            orbit_param_range = (10 ** log10_P_range[0], 10 ** log10_P_range[1])
        elif ini_orbit_scheme == 'Hurley2002':
            orbit_param_range = sep_range

    if ini_orbit_scheme == 'Sana2012':
        orbit_values = np.logspace(
            np.log10(orbit_param_range[0]), np.log10(orbit_param_range[1]), num=n_grid_popbin, endpoint=True
        )
        logger.info(f"Using Sana2012 orbital model: period range {orbit_param_range}", extra={"console": True})
    elif ini_orbit_scheme == 'Hurley2002':
        orbit_values = np.logspace(
            np.log(orbit_param_range[0]), np.log(orbit_param_range[1]), num=n_grid_popbin, endpoint=True, base=np.e
        )
        logger.info(f"Using Hurley2002 orbital model: semi-major axis range {orbit_param_range}", extra={"console": True})
    else:
        raise ValueError(f"Unsupported orbital model: {ini_orbit_scheme}, available options: 'Sana2012', 'Hurley2002'")

    # 3. Generate 3D grid (m1, m2, orbit)
    parameter_space = np.array(np.meshgrid(m1_values, m2_values, orbit_values)).T.reshape(-1, 3)

    # 4. Generate eccentricity array
    logger.info(f"Eccentricity distribution: {ini_ecc_scheme}", extra={"console": True})
    rng = np.random.default_rng(random_seed)
    n_points = n_grid_popbin ** 3
    if ini_ecc_scheme == 'zero':
        ecc_array = np.zeros(n_points)
    elif ini_ecc_scheme == 'uniform':
        ecc_array = rng.choice(np.arange(0.0, 1.0, 0.1), size=n_points)
        # ecc_array = rng.uniform(0, 1, size=n_points)
    elif ini_ecc_scheme == 'thermal':
        ecc_array = np.sqrt(rng.random(size=n_points))
    else:
        raise ValueError(
            f"Unsupported eccentricity scheme: '{ini_ecc_scheme}'. "
            f"Available options: 'zero', 'uniform', 'thermal'"
        )

    # 5. Pre-compute binary weights
    weights = weight_binary(
        M1=parameter_space[:, 0],
        M2=parameter_space[:, 1],
        orbit_param=parameter_space[:, 2],
        m1_range=m1_range,
        m2_range=m2_range,
        orbit_param_range=orbit_param_range,
        n_grid_popbin=n_grid_popbin,
        ini_orbit_scheme=ini_orbit_scheme,
        IMF_scheme=IMF_scheme,
        binary_fraction=binary_fraction,
    )

    # 6. Combine into 5-column parameter space [m1, m2, orbit, ecc, weight]
    parameter_space = np.column_stack([parameter_space, ecc_array, weights])

    logger.info(
        f"Parameter space for popbin generation complete: {len(parameter_space):,} points, shape {parameter_space.shape}"
    )

    return parameter_space


def weight_binary(
        M1: float | np.ndarray,
        M2: float | np.ndarray,
        orbit_param: float | np.ndarray,
        m1_range: tuple[float, float] = m1_range,
        m2_range: tuple[float, float] = m2_range,
        orbit_param_range: tuple[float, float] | None = None,
        n_grid_popbin: int = n_grid_popbin,
        ini_orbit_scheme: str = ini_orbit_scheme,
        IMF_scheme: str = IMF_scheme,
        binary_fraction: float | str = binary_fraction,
) -> float | np.ndarray:
    """Calculate the weight contribution of binary systems to the stellar population

    The weight formula for binary systems depends on the orbital model:
        - Sana2012: Uses period distribution (log10 P), requires period parameter
        - Hurley2002: Uses semi-major axis distribution (ln a), independent of specific sep

    Args:
        M1: Primary star mass [unit: M_sun]
        M2: Secondary star mass [unit: M_sun]
        orbit_param: Orbital period [unit: days] (Sana2012 model) or orbital semi-major axis [unit: R_sun] (Hurley2002 model)
        m1_range: Primary star mass range
        m2_range: Secondary star mass range
        orbit_param_range: Orbital period range [unit: days] (Sana2012) or semi-major axis range
            [unit: R_sun] (Hurley2002). If None, the default range is inferred from ini_orbit_scheme.
        n_grid_popbin: Number of grid points for each parameter in logarithmic space
        ini_orbit_scheme: Initial orbit model. Support 'Sana2012' and 'Hurley2002'
        IMF_scheme: IMF model name. Must be one of: 'Kroupa2002', 'Kroupa1993', or 'Weisz2015'
        binary_fraction: Binary fraction. Can be a float in [0, 1] or 'Haaften2013'

    Returns:
        Weight value(s) for binary systems

    Raises:
        ValueError: When ini_orbit_scheme, IMF_scheme or binary_fraction is not supported
    """
    # Density in logarithmic mass space
    Phi_lnM1 = M1 * initial_mass_function(M1, IMF_scheme)
    varphi_lnM2 = M2 / M1

    # Logarithmic mass intervals
    delta_lnM1 = np.log(m1_range[1] / m1_range[0]) / (n_grid_popbin - 1)
    delta_lnM2 = np.log(m2_range[1] / m2_range[0]) / (n_grid_popbin - 1)

    # Binary mass weight
    weight = frac_binary(M1, binary_fraction) * Phi_lnM1 * delta_lnM1 * varphi_lnM2 * delta_lnM2

    # Calculate based on orbital model
    if orbit_param_range is None:
        if ini_orbit_scheme == 'Sana2012':
            orbit_param_range = (10 ** log10_P_range[0], 10 ** log10_P_range[1])
        elif ini_orbit_scheme == 'Hurley2002':
            orbit_param_range = sep_range

    if ini_orbit_scheme == 'Sana2012':
        # Period distribution function
        Psi_log10_P = _get_log10_P_normalize_factor() * np.log10(orbit_param) ** log10_P_index

        # Logarithmic period interval
        delta_log10_P = np.log10(orbit_param_range[1] / orbit_param_range[0]) / (n_grid_popbin - 1)

        weight = weight * Psi_log10_P * delta_log10_P

    elif ini_orbit_scheme == 'Hurley2002':
        # Semi-major axis distribution function (uniform in ln a space)
        Psi_lna = 1 / np.log(sep_max / sep_min)

        # Logarithmic semi-major axis interval
        delta_lna = np.log(orbit_param_range[1] / orbit_param_range[0]) / (n_grid_popbin - 1)

        weight = weight * Psi_lna * delta_lna

    else:
        raise ValueError(f"Unsupported orbital model: {ini_orbit_scheme}, available options: 'Sana2012', 'Hurley2002'")

    return weight


def _get_log10_P_normalize_factor() -> float:
    """Calculate the normalization factor for the log-period distribution.

    Normalization condition: ∫ P(log10 P) d(log10 P) = 1.

    Returns:
        Normalization factor C.
    """
    x = np.linspace(log10_P_min, log10_P_max, 10000)
    y = x ** log10_P_index

    integral = integrate.trapezoid(y, x)
    C = 1.0 / integral

    return C


# ---------------------------------------------------------------------------
# Initial mass functions and binary fractions
# ---------------------------------------------------------------------------


def initial_mass_function(
        M: float | np.ndarray, 
        IMF_scheme: str = IMF_scheme
) -> float | np.ndarray:
    """Calculate the stellar Initial Mass Function (IMF)

    Supports three commonly used IMF models:
        - Kroupa2002: Kroupa (2002) three-segment power law (default)
        - Kroupa1993: Kroupa et al. (1993) three-segment power law
        - Weisz2015: Weisz et al. (2015) three-segment power law

    Args:
        M: Stellar mass [unit: M_sun], range 0.08 - 150
        IMF_scheme: IMF model name. Must be one of: 'Kroupa2002', 'Kroupa1993', or 'Weisz2015'

    Returns:
        IMF value, representing the relative probability density at the given mass

    Raises:
        ValueError: When IMF_scheme is not supported
    """
    if IMF_scheme == 'Kroupa2002':
        return imf_kroupa2002(M)
    elif IMF_scheme == 'Kroupa1993':
        return imf_kroupa1993(M)
    elif IMF_scheme == 'Weisz2015':
        return imf_weisz2015(M)
    else:
        raise ValueError(
            f"Unsupported IMF model: {IMF_scheme}, "
            f"available options: 'Kroupa2002', 'Kroupa1993', 'Weisz2015'"
        )


def imf_kroupa2002(M: float | np.ndarray) -> float | np.ndarray:
    """Kroupa (2002) IMF

    Formula:
        ξ(M) ∝ M^{-α}
        α = 1.3  (0.08 ≤ M ≤ 0.5)
        α = 2.2  (0.5 < M ≤ 1)
        α = 2.3  (1 < M ≤ 150)

    Normalization: ∫ ξ(M) dM = 1 over 0.08-150 M_sun.
    """
    M_arr = np.asarray(M)
    result = np.zeros_like(M_arr, dtype=float)

    # Normalization constants.
    a = 0.25009247
    b = a * 0.5 ** 0.9

    # Mass intervals.
    mask1 = (M_arr >= 0.08) & (M_arr <= 0.5)
    mask2 = (M_arr > 0.5) & (M_arr <= 1.0)
    mask3 = (M_arr > 1.0) & (M_arr <= 150.0)

    result[mask1] = a * M_arr[mask1] ** -1.3
    result[mask2] = b * M_arr[mask2] ** -2.2
    result[mask3] = b * M_arr[mask3] ** -2.3

    return result if isinstance(M, np.ndarray) else result.item()


def imf_kroupa1993(M: float | np.ndarray) -> float | np.ndarray:
    """Kroupa et al. (1993) IMF

    Formula:
        ξ(M) ∝ M^{-α}
        α = 1.3  (0.1 < M ≤ 0.5)
        α = 2.2  (0.5 < M ≤ 1)
        α = 2.7  (1 < M)

    Normalization: ∫ ξ(M) dM = 1 over 0.1-inf M_sun.
    """
    M_arr = np.asarray(M)
    result = np.zeros_like(M_arr, dtype=float)

    mask1 = (M_arr > 0.1) & (M_arr <= 0.5)
    mask2 = (M_arr > 0.5) & (M_arr <= 1.0)
    mask3 = M_arr > 1.0

    result[mask1] = 0.29056 * M_arr[mask1] ** -1.3
    result[mask2] = 0.15571 * M_arr[mask2] ** -2.2
    result[mask3] = 0.15571 * M_arr[mask3] ** -2.7

    return result if isinstance(M, np.ndarray) else result.item()


def imf_weisz2015(M: float | np.ndarray) -> float | np.ndarray:
    """Weisz et al. (2015) IMF

    Formula:
        ξ(M) ∝ M^{-α}
        α = 1.3  (0.08 < M ≤ 0.5)
        α = 2.3  (0.5 < M ≤ 1)
        α = 2.45 (1 ≤ M ≤ 100)

    Normalization: ∫ ξ(M) dM = 1 over 0.08-100 M_sun.
    """
    M_arr = np.asarray(M)
    result = np.zeros_like(M_arr, dtype=float)

    c = 0.2074

    mask1 = (M_arr > 0.08) & (M_arr <= 0.5)
    mask2 = (M_arr > 0.5) & (M_arr <= 1.0)
    mask3 = (M_arr >= 1.0) & (M_arr <= 100.0)

    result[mask1] = c * M_arr[mask1] ** -1.3
    result[mask2] = c * M_arr[mask2] ** -2.3
    result[mask3] = c * M_arr[mask3] ** -2.45

    return result if isinstance(M, np.ndarray) else result.item()


def frac_binary(
        M: float | np.ndarray,
        binary_fraction: float | str = binary_fraction
) -> float | np.ndarray:
    """Calculate the fraction of binary systems among stars

    Supports two input modes:
        1. String 'Haaften2013': Use Haaften et al. (2013) model
        2. Float: Directly return the specified fraction

    Args:
        M: Stellar mass [unit: M_sun]
        binary_fraction: Binary fraction. Can be a float in [0, 1] or 'Haaften2013' for the mass-dependent model.

    Returns:
        Binary fraction, range [0, 1]

    Raises:
        ValueError: When binary_fraction is not a supported type or value
    """
    # Case 1: Direct float value
    if isinstance(binary_fraction, float):
        if not 0 <= binary_fraction <= 1:
            raise ValueError(
                f"Binary fraction must be in [0, 1], got: {binary_fraction}"
            )
        return binary_fraction

    # Case 2: String preset model
    if binary_fraction == 'Haaften2013':
        return 0.5 + 0.25 * np.log10(M)
    else:
        raise ValueError(
            f"binary_fraction must be a float in [0, 1] or 'Haaften2013', "
            f"got: {binary_fraction}"
        )


def average_stellar_mass(
        IMF_scheme: str = IMF_scheme,
        binary_fraction: float | str = binary_fraction
) -> float:
    """Calculate the average stellar mass (considering both single and binary systems)

    For binary systems, the total system mass is 1.5 × M_primary (assuming an average mass ratio q = 0.5)
    For single stars, the system mass equals the stellar mass.

    Args:
        IMF_scheme: IMF model name. Must be one of: 'Kroupa2002', 'Kroupa1993', or 'Weisz2015'
        binary_fraction: Binary fraction. Can be a float in [0, 1] or 'Haaften2013'

    Returns:
        Average system mass [unit: M_sun]

    Notes:
        Integration range: 0.08 - 100 M_sun
        Calculation formula: ⟨M⟩ = ∫ ξ(M) × M_system(M) dM / ∫ ξ(M) dM
    """
    # Integration grid
    masses = np.linspace(0.08, 100, 100000)

    # Initial mass function values
    imf = initial_mass_function(masses, IMF_scheme)

    # Binary fraction
    fb = frac_binary(masses, binary_fraction)

    # System total mass: binary systems average 1.5 × M, single systems are M
    system_mass = fb * 1.5 * masses + (1 - fb) * masses

    # Numerical integration using Simpson's rule (more accurate than trapezoidal)
    numerator = integrate.trapezoid(imf * system_mass, masses)
    denominator = integrate.trapezoid(imf, masses)

    return numerator / denominator


# ---------------------------------------------------------------------------
# Target filtering
# ---------------------------------------------------------------------------


def search_star(
        data: np.ndarray,
        condition: str | None = None,
        required_events: list[str] | None = None
) -> np.ndarray:
    """
    Filter star data and return a boolean mask

    Parameters:
    -----------
    data : np.ndarray
        Star data
    condition : str | None
        Numerical filter condition, must return boolean values

        Supported operations:
        1. Comparison: <, >, <=, >=, ==, !=
        2. Logical: &, |, ~
        3. Direct boolean column: 'bound' or '~bound'

        Examples:
          - Comparison: "type == 'MS'"
          - Range: "(mass >= 5) & (mass <= 8)"
          - Boolean column: "bound" / "~bound"
          - Combination: "bound & (ecc >= 0.1)"
    required_events : List[str] | None
        Required event list, succeeds if any event exists in the data

    Returns:
    --------
    np.ndarray
        Boolean mask array, length equals len(data)
    """
    # Check if any required event exists
    if required_events and 'event' in data.dtype.names:
        if not np.any(np.isin(data['event'], required_events)):
            return np.zeros(len(data), dtype=bool)

    # Execute numerical filter
    if condition:
        namespace = {name: data[name] for name in data.dtype.names}
        mask = eval(condition, namespace)
        # Validate that condition returns boolean values
        if mask.dtype != bool:
            raise ValueError(
                f"condition must return boolean values, but got {mask.dtype}\n"
                f"  Your input: {condition}\n"
                f"  Hint: Please use comparison operators (<, >, <=, >=, ==, !=, &, |, ~)\n"
            )
    else:
        mask = np.ones(len(data), dtype=bool)

    return mask


def search_popsin_source(
        star_data: np.ndarray,
        target: dict,
) -> np.ndarray:
    """
    Search single population synthesis targets with filtering.

    Parameters:
    -----------
    star_data : np.ndarray
        Star dataset
    target : dict
                Configuration dictionary for the target source

    Returns:
    --------
    np.ndarray
        Boolean mask for the stars that satisfy the condition.
        Length equals len(star_data).
    """
    # Get filter condition
    star_condition = target.get('star')

    # Get event requirements
    star_events = target.get('events')

    # Search star
    mask = search_star(star_data, star_condition, star_events)

    return mask


def search_popbin_source(
        star1_data: np.ndarray,
        star2_data: np.ndarray,
        binary_data: np.ndarray,
        target: dict,
) -> np.ndarray:
    """
    Search binary population synthesis targets with automatic role matching.

    Parameters:
    -----------
    star1_data : np.ndarray
        Primary star dataset
    star2_data : np.ndarray
        Secondary star dataset
    binary_data : np.ndarray
        Binary system dataset
    target : dict
        Configuration dictionary for the target source

    Returns:
    --------
    np.ndarray
        Boolean mask for the binary systems that satisfy either combination.
        Length equals len(binary_data).
    """
    # Get filter conditions
    star1_condition = target.get('star1')
    star2_condition = target.get('star2')
    binary_condition = target.get('binary')

    # Get event requirements
    events = target.get('events')
    star1_events = events.get('star1') if events else None
    star2_events = events.get('star2') if events else None
    binary_events = events.get('binary') if events else None

    # Search star1/star2
    # Combination 1: star1_data uses star1 condition, star2_data uses star2 condition
    mask_star1_1 = search_star(star1_data, star1_condition, star1_events)
    mask_star2_2 = search_star(star2_data, star2_condition, star2_events)
    # Combination 2: star1_data uses star2 condition, star2_data uses star1 condition
    mask_star1_2 = search_star(star1_data, star2_condition, star2_events)
    mask_star2_1 = search_star(star2_data, star1_condition, star1_events)

    # Search binary systems
    mask_binary = search_star(binary_data, binary_condition, binary_events)

    # Combine masks for both combinations
    mask_1 = mask_star1_1 & mask_star2_2
    mask_2 = mask_star1_2 & mask_star2_1
    mask = (mask_1 | mask_2) & mask_binary

    return mask


# ---------------------------------------------------------------------------
# Naming, geometry, and small physical conversions
# ---------------------------------------------------------------------------


def get_metallicity_str(z):
    """Convert metallicity value to filename suffix

    Examples:
        0.02  -> 'Z002'
        0.009 -> 'Z0009'
        0.0001 -> 'Z00001'
    """
    s = f"{z:.9f}".rstrip('0')
    num = s[2:]
    return f"Z0{num}"


def random_rotation_matrix():
    """Generate a random rotation matrix (uniform distribution)"""

    # Randomly generate rotation axis (u_x, u_y, u_z)
    u = np.random.normal(size=3)
    u /= np.linalg.norm(u)      # Normalize to unit vector
    # u = np.array([0, 0, 1])

    # Randomly select rotation angle theta
    theta = np.random.uniform(0, 2 * np.pi)
    # theta = np.pi / 2

    # Extract rotation axis components
    ux, uy, uz = u
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Construct rotation matrix
    R = np.array([
        [cos_theta + ux ** 2 * (1 - cos_theta),
         ux * uy * (1 - cos_theta) - uz * sin_theta,
         ux * uz * (1 - cos_theta) + uy * sin_theta],

        [uy * ux * (1 - cos_theta) + uz * sin_theta,
         cos_theta + uy ** 2 * (1 - cos_theta),
         uy * uz * (1 - cos_theta) - ux * sin_theta],

        [uz * ux * (1 - cos_theta) - uy * sin_theta,
         uz * uy * (1 - cos_theta) + ux * sin_theta,
         cos_theta + uz ** 2 * (1 - cos_theta)]
    ])

    return R


def rotate_velocity_offset_to_galactocentric(v_offset):
    """Rotate an offset velocity into Galactocentric velocity components."""

    # If input offset velocity is invalid
    if np.isnan(v_offset).any():
        return np.zeros(3)

    # Generate random rotation matrix
    rotation_matrix = random_rotation_matrix()

    # Transform velocity from pre-SN center-of-mass frame to Galactocentric cylindrical coordinates (vR/vT/vz)
    v_gal = rotation_matrix @ v_offset

    return v_gal


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def atomic_append_csv(target_file, df):
    """Append a DataFrame to a CSV file with an exclusive file lock.

    The file is locked immediately after low-level opening to avoid a race
    window between checking file size and writing rows.
    """
    # Open at the file-descriptor level.
    fd = os.open(str(target_file), os.O_WRONLY | os.O_CREAT)

    try:
        # Lock before wrapping the descriptor as a Python file object.
        fcntl.flock(fd, fcntl.LOCK_EX)

        # Read the true file size from the descriptor.
        file_size = os.fstat(fd).st_size

        # Move to the end for append semantics.
        os.lseek(fd, 0, os.SEEK_END)

        # Convert to a file object and write rows.
        with os.fdopen(fd, 'a') as f:
            if file_size == 0:
                df.to_csv(f, header=True, index=False)
            else:
                df.to_csv(f, header=False, index=False)

    except Exception:
        # Close manually only on error; normal completion is handled by fdopen.
        os.close(fd)
        raise


# ---------------------------------------------------------------------------
# Miscellaneous physical conversions
# ---------------------------------------------------------------------------


def calculate_metallicity_from_feh(fe_h):
    """Calculate metallicity Z from [Fe/H] following Bertelli et al. (1994)."""
    return 10 ** (0.977 * fe_h - 1.699)
