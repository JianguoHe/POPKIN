"""Microlensing observables for isolated compact objects."""

import numpy as np

MAS2_PER_DEG2 = (3600.0 * 1000.0) ** 2


def add_bh_microlensing_observables(
        data,
        source_distance=8.0,
        earth_velocity_y=230.0,
        earth_velocity_z=15.5,
        source_velocity_dispersion=80.0,
        source_velocity_y=None,
        source_velocity_z=None,
        random_state=None,
        source_surface_density=1.5e8,
        survey_area_deg2=31.0,
        distance_col="dist",
        mass_col="mass",
        lens_velocity_y_col="vT",
        lens_velocity_z_col="vz",
        weight_col="num",
        copy=True,
):
    """Add basic BH microlensing observables to a foreground-lens table.

    The calculation follows the simplified Galactic-bulge setup used in the
    POPKIN isolated-BH analysis: foreground BH lenses are paired statistically
    with bulge source stars at a fixed source distance.

    Args:
        data: Table-like object, usually a pandas DataFrame.
        source_distance: Source distance in kpc.
        earth_velocity_y: Solar/Earth transverse velocity along the rotation direction in km/s.
        earth_velocity_z: Solar/Earth vertical velocity in km/s.
        source_velocity_dispersion: One-dimensional bulge source velocity dispersion in km/s.
        source_velocity_y: Optional source velocities along y in km/s. If omitted, sampled from
            a normal distribution with ``source_velocity_dispersion``.
        source_velocity_z: Optional source velocities along z in km/s. If omitted, sampled from
            a normal distribution with ``source_velocity_dispersion``.
        random_state: Optional seed for source velocity sampling.
        source_surface_density: Surface density of source stars in the survey field.
        survey_area_deg2: Survey area in square degrees.
        distance_col: Lens distance column in kpc.
        mass_col: Lens mass column in solar masses.
        lens_velocity_y_col: Lens transverse velocity column along y in km/s.
        lens_velocity_z_col: Lens transverse velocity column along z in km/s.
        weight_col: Lens statistical weight column. Set to ``None`` to use unit weights.
        copy: Whether to return a copy.

    Returns:
        Table with added ``pm_rel``, ``theta_E``, ``tE``, and ``num_lens`` columns.
    """
    out = data.copy() if copy else data
    n_lenses = len(out)

    rng = np.random.default_rng(random_state)
    if source_velocity_y is None:
        source_velocity_y = rng.normal(loc=0.0, scale=source_velocity_dispersion, size=n_lenses)
    else:
        source_velocity_y = np.asarray(source_velocity_y, dtype=np.float64)

    if source_velocity_z is None:
        source_velocity_z = rng.normal(loc=0.0, scale=source_velocity_dispersion, size=n_lenses)
    else:
        source_velocity_z = np.asarray(source_velocity_z, dtype=np.float64)

    if len(source_velocity_y) != n_lenses or len(source_velocity_z) != n_lenses:
        raise ValueError("source_velocity_y and source_velocity_z must match the number of lenses")

    distance = np.asarray(out[distance_col], dtype=np.float64)
    mass = np.asarray(out[mass_col], dtype=np.float64)
    lens_velocity_y = np.asarray(out[lens_velocity_y_col], dtype=np.float64)
    lens_velocity_z = np.asarray(out[lens_velocity_z_col], dtype=np.float64)

    valid_distance = np.isfinite(distance) & (distance > 0.0)
    valid_lens = valid_distance & (distance < source_distance) & np.isfinite(mass) & (mass > 0.0)

    pm = np.full(n_lenses, np.nan, dtype=np.float64)
    theta_e = np.full(n_lenses, np.nan, dtype=np.float64)
    t_e = np.full(n_lenses, np.nan, dtype=np.float64)
    num_lens = np.full(n_lenses, np.nan, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        pm_y = (
            (lens_velocity_y[valid_distance] - earth_velocity_y) / distance[valid_distance]
            + (earth_velocity_y - source_velocity_y[valid_distance]) / source_distance
        )
        pm_z = (
            (lens_velocity_z[valid_distance] - earth_velocity_z) / distance[valid_distance]
            + (earth_velocity_z - source_velocity_z[valid_distance]) / source_distance
        )
        pm[valid_distance] = np.sqrt(pm_y ** 2 + pm_z ** 2) / 4.74

        theta_argument = 8.1 * mass[valid_lens] * (1.0 / distance[valid_lens] - 1.0 / source_distance)
        theta_e[valid_lens] = np.sqrt(theta_argument)

    valid_microlensing = valid_lens & np.isfinite(pm) & (pm > 0.0)
    t_e[valid_microlensing] = theta_e[valid_microlensing] / pm[valid_microlensing] * 365.0

    if weight_col is None:
        weights = np.ones(n_lenses, dtype=np.float64)
    else:
        weights = np.asarray(out[weight_col], dtype=np.float64)

    normalization = source_surface_density / (survey_area_deg2 * MAS2_PER_DEG2)
    num_lens[valid_microlensing] = (
        2.0 * theta_e[valid_microlensing] * pm[valid_microlensing]
        + np.pi * theta_e[valid_microlensing] ** 2
    ) * normalization * weights[valid_microlensing]

    out["pm_rel"] = pm
    out["theta_E"] = theta_e
    out["tE"] = t_e
    out["num_lens"] = num_lens

    return out


def estimate_bh_lens_fraction_by_timescale(t_e):
    """Estimate the BH-lens fraction from microlensing event timescale.

    This is an empirical approximation digitized from the right panel of
    Figure 7 in Lam et al. (2020), based on their Mock EWS simulation. The
    approximation is intended for rough comparison with OGLE-EWS-like
    microlensing samples, rather than as a universal microlensing relation.
    The fraction is set to zero below 30 days because the BH contribution is
    very small there and this regime is not used to infer the total event rate
    from BH-lensing events.

    Args:
        t_e: Event timescale in days.

    Returns:
        Estimated BH-lens fraction for each input timescale.
    """
    t_e = np.asarray(t_e, dtype=np.float64)

    t1, t2, t3, t4 = 30.0, 68.4, 146.3, 320.0
    f1, f2, f3, f4 = 9.1e-4, 0.0826, 0.311, 0.495

    conditions = [
        t_e < t1,
        (t_e >= t1) & (t_e < t2),
        (t_e >= t2) & (t_e < t3),
        (t_e >= t3) & (t_e <= t4),
        t_e > t4,
    ]

    slope1 = (f2 - f1) / (t2 - t1)
    slope2 = (f3 - f2) / (t3 - t2)
    slope3 = (f4 - f3) / (t4 - t3)

    functions = [
        lambda x: 0.0,
        lambda x: f1 + (x - t1) * slope1,
        lambda x: f2 + (x - t2) * slope2,
        lambda x: f3 + (x - t3) * slope3,
        lambda x: f4,
    ]

    return np.piecewise(t_e, conditions, functions)


__all__ = [
    "add_bh_microlensing_observables",
    "estimate_bh_lens_fraction_by_timescale",
]
