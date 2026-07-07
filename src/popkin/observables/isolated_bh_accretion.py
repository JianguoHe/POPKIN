"""Post-processing observables for isolated black holes accreting from the ISM."""

from pathlib import Path

import numpy as np
import pandas as pd

from popkin.config.logger import get_logger
from popkin.constants import c_light, pc
from popkin.galaxies import MilkyWay
from popkin.physics import m_dot_bh_hot_accretion, m_dot_edd

logger = get_logger(__name__)

REQUIRED_COLUMNS = ("mass", "v_pec", "num", "dist", "rho", "z")
DEFAULT_PHASES = ("MCs", "coldHI", "warmHI", "warmHII", "hotHII")
DEFAULT_DISCRETE_DENSITY_POINTS = {"MCs": 10, "coldHI": 10}
DEFAULT_DISTANCE_CUTS_KPC = (0.5, 1.0, 10.0)
PHASE_ATTRIBUTE_NAMES = {
    "MCs": "molecular_clouds",
    "coldHI": "cold_hi",
    "warmHI": "warm_hi",
    "warmHII": "warm_hii",
    "hotHII": "hot_hii",
}


def summarize_isolated_bh_accretion(
    data,
    output_path,
    *,
    galaxy=None,
    phases=None,
    discrete_density_points=None,
    mdot_range=None,
    flux_bol_range=None,
    candidate_flux_bol_threshold=1e-14,
    candidate_output_format="parquet",
    candidate_compression="zstd",
    distance_cuts_kpc=DEFAULT_DISTANCE_CUTS_KPC,
):
    """Write compact ISM-accretion summaries for isolated black holes.

    Args:
        data: A pandas DataFrame with columns ``mass``, ``v_pec``, ``num``,
            ``dist``, ``rho``, and ``z``. Here ``rho`` is the cylindrical
            Galactocentric radius, not gas density.
        output_path: Directory where the summary CSV files are written.
        galaxy: Optional galaxy model. If omitted, a default ``MilkyWay`` model
            is created inside the function.
        phases: ISM phase names to process.
        discrete_density_points: Mapping from phase name to the number of
            density-grid points used for phases with broad density PDFs.
        mdot_range: Thresholds for cumulative ``N(>mdot)``.
        flux_bol_range: Thresholds for cumulative ``N(>F_bol)``.
        candidate_flux_bol_threshold: Minimum bolometric flux for writing
            candidate rows. Set to ``None`` to skip the candidate table.
        candidate_output_format: Format for the candidate table. Use
            ``"parquet"`` for compact, fast binary output, or ``"csv"`` for
            legacy text output.
        candidate_compression: Compression codec used for Parquet candidate
            output.
        distance_cuts_kpc: Distance cuts used for extra cumulative flux columns.
    """
    _validate_columns(data)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    galaxy = galaxy or MilkyWay()
    phases = tuple(phases or DEFAULT_PHASES)
    discrete_density_points = dict(discrete_density_points or DEFAULT_DISCRETE_DENSITY_POINTS)
    mdot_range = np.asarray(mdot_range if mdot_range is not None else np.logspace(5, 18, num=66))
    flux_bol_range = np.asarray(
        flux_bol_range if flux_bol_range is not None else np.logspace(-25, -10, num=61)
    )

    mass = data["mass"].to_numpy(copy=False)
    v_pec = data["v_pec"].to_numpy(copy=False)
    num = data["num"].to_numpy(copy=False)
    dist = data["dist"].to_numpy(copy=False)
    rho = data["rho"].to_numpy(copy=False)
    z = data["z"].to_numpy(copy=False)

    mdot_results = {}
    flux_results = {}
    candidate_tables = []

    logger.info(
        f"Processing isolated-BH ISM accretion: rows={len(data)}, phases={len(phases)}",
        extra={"console": True},
    )

    for phase in phases:
        gas = _get_ism_phase(galaxy, phase)
        filling_fraction = gas.filling_fraction(rho, z)

        if phase in discrete_density_points:
            n_points = int(discrete_density_points[phase])
            density_grid, density_weights = gas.get_discrete_number_density(n_points=n_points)

            for density_index, (n_h, density_weight) in enumerate(zip(density_grid, density_weights), start=1):
                logger.info(
                    f"Processing {phase} density grid {density_index}/{n_points}: n_H={n_h:.3e}",
                    extra={"console": True},
                )
                cs = gas.cs(n_h) if callable(gas.cs) else gas.cs
                mdot = m_dot_bh_hot_accretion(m=mass, v=v_pec, n=n_h, mu=gas.mu, cs=cs)
                weights = num * filling_fraction * density_weight

                _accumulate_phase_results(
                    phase=phase,
                    n_h=n_h,
                    density_index=density_index,
                    mdot=mdot,
                    weights=weights,
                    dist=dist,
                    mass=mass,
                    v_pec=v_pec,
                    source_index=_optional_column(data, "index"),
                    mdot_range=mdot_range,
                    flux_bol_range=flux_bol_range,
                    distance_cuts_kpc=distance_cuts_kpc,
                    candidate_flux_bol_threshold=candidate_flux_bol_threshold,
                    mdot_results=mdot_results,
                    flux_results=flux_results,
                    candidate_tables=candidate_tables,
                )
        else:
            logger.info(f"Processing {phase}", extra={"console": True})
            mdot = m_dot_bh_hot_accretion(m=mass, v=v_pec, n=gas.n, mu=gas.mu, cs=gas.cs)
            weights = num * filling_fraction

            _accumulate_phase_results(
                phase=phase,
                n_h=gas.n,
                density_index=None,
                mdot=mdot,
                weights=weights,
                dist=dist,
                mass=mass,
                v_pec=v_pec,
                source_index=_optional_column(data, "index"),
                mdot_range=mdot_range,
                flux_bol_range=flux_bol_range,
                distance_cuts_kpc=distance_cuts_kpc,
                candidate_flux_bol_threshold=candidate_flux_bol_threshold,
                mdot_results=mdot_results,
                flux_results=flux_results,
                candidate_tables=candidate_tables,
            )

    mdot_df = _build_result_frame("mdot", mdot_range, mdot_results)
    mdot_path = output_dir / "IBH_N_mdot.csv"
    mdot_df.to_csv(mdot_path, index=False)
    logger.info(f"Saved mdot cumulative summary: {mdot_path}", extra={"console": True})

    flux_df = _build_result_frame("F_bol", flux_bol_range, flux_results)
    flux_path = output_dir / "IBH_N_F_bol.csv"
    flux_df.to_csv(flux_path, index=False)
    logger.info(f"Saved bolometric-flux cumulative summary: {flux_path}", extra={"console": True})

    if candidate_tables:
        candidates = pd.concat(candidate_tables, ignore_index=True)
        candidate_path = output_dir / _candidate_filename(
            candidate_flux_bol_threshold,
            candidate_output_format,
        )
        _write_candidate_table(
            candidates,
            candidate_path,
            output_format=candidate_output_format,
            compression=candidate_compression,
        )
        logger.info(f"Saved bright-candidate table: {candidate_path}", extra={"console": True})
    elif candidate_flux_bol_threshold is not None:
        candidate_path = output_dir / _candidate_filename(
            candidate_flux_bol_threshold,
            candidate_output_format,
        )
        if candidate_path.exists():
            candidate_path.unlink()
            logger.info(f"Removed stale bright-candidate table: {candidate_path}", extra={"console": True})


def _accumulate_phase_results(
    *,
    phase,
    n_h,
    density_index,
    mdot,
    weights,
    dist,
    mass,
    v_pec,
    source_index,
    mdot_range,
    flux_bol_range,
    distance_cuts_kpc,
    candidate_flux_bol_threshold,
    mdot_results,
    flux_results,
    candidate_tables,
):
    dist_array = np.asarray(dist)
    valid = (
        np.isfinite(mdot)
        & np.isfinite(weights)
        & (weights > 0)
        & np.isfinite(dist_array)
        & (dist_array > 0)
    )
    if not valid.any():
        return

    mdot = np.asarray(mdot)[valid].astype(np.float64, copy=False)
    weights = np.asarray(weights)[valid].astype(np.float64, copy=False)
    dist = dist_array[valid].astype(np.float64, copy=False)
    mass = np.asarray(mass)[valid].astype(np.float64, copy=False)
    v_pec = np.asarray(v_pec)[valid]
    source_index = np.asarray(source_index)[valid] if source_index is not None else None

    mdot_edd = m_dot_edd(m=mass)
    mdot_edd_ratio = mdot / mdot_edd
    eta = radiative_efficiency_xie_yuan_2012(mdot_edd_ratio)
    luminosity_bol = eta * mdot * c_light**2

    dist_cm = dist * (1000.0 * pc)
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_bol = luminosity_bol / (4.0 * np.pi * dist_cm**2)

    _add_result(mdot_results, phase, _weighted_cumulative(mdot, weights, mdot_range))
    _add_result(flux_results, phase, _weighted_cumulative(flux_bol, weights, flux_bol_range))

    for distance_cut in distance_cuts_kpc:
        within_distance = dist <= distance_cut
        label = _distance_label(distance_cut)
        _add_result(
            flux_results,
            f"{phase}_{label}",
            _weighted_cumulative(flux_bol[within_distance], weights[within_distance], flux_bol_range),
        )

    if candidate_flux_bol_threshold is not None:
        candidate_mask = np.isfinite(flux_bol) & (flux_bol > candidate_flux_bol_threshold)
        if candidate_mask.any():
            candidate_data = {
                "phase": phase,
                "density_index": density_index,
                "n_H": n_h,
                "mass": mass[candidate_mask],
                "v_pec": v_pec[candidate_mask],
                "dist": dist[candidate_mask],
                "num": weights[candidate_mask],
                "mdot": mdot[candidate_mask],
                "mdot_edd_ratio": mdot_edd_ratio[candidate_mask],
                "eta": eta[candidate_mask],
                "L_bol": luminosity_bol[candidate_mask],
                "F_bol": flux_bol[candidate_mask],
            }
            if source_index is not None:
                candidate_data = {"index": source_index[candidate_mask], **candidate_data}
            candidate_tables.append(pd.DataFrame(candidate_data))


def _weighted_cumulative(values, weights, thresholds):
    """Return weighted N(>threshold) without repeatedly scanning the array."""
    values = np.asarray(values)
    weights = np.asarray(weights)
    thresholds = np.asarray(thresholds)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.zeros(len(thresholds), dtype=np.float64)

    values = values[valid]
    weights = weights[valid]

    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cumulative = np.concatenate(([0.0], np.cumsum(weights_sorted)))
    idx = np.searchsorted(values_sorted, thresholds, side="right")

    return cumulative[-1] - cumulative[idx]


def _add_result(results, key, values):
    if key in results:
        results[key] += values
    else:
        results[key] = values.copy()


def _build_result_frame(threshold_name, thresholds, results):
    frame = pd.DataFrame({threshold_name: thresholds})
    for key, values in results.items():
        frame[key] = values
    return frame


def _validate_columns(data):
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns for isolated-BH accretion: {missing}")


def _get_ism_phase(galaxy, phase):
    attribute_name = PHASE_ATTRIBUTE_NAMES.get(phase, phase)
    if not hasattr(galaxy, attribute_name):
        raise AttributeError(f"Galaxy model has no ISM phase {phase!r} ({attribute_name!r})")
    return getattr(galaxy, attribute_name)


def _optional_column(data, column):
    if column not in data.columns:
        return None
    return data[column].to_numpy(copy=False)


def _distance_label(distance_kpc):
    if distance_kpc < 1:
        return f"{int(round(distance_kpc * 1000))}pc"
    if float(distance_kpc).is_integer():
        return f"{int(distance_kpc)}kpc"
    return f"{distance_kpc:g}kpc"


def _write_candidate_table(candidates, path, *, output_format, compression):
    output_format = output_format.lower()
    if output_format == "csv":
        candidates.to_csv(path, index=False)
    elif output_format == "parquet":
        candidates.to_parquet(path, index=False, compression=compression)
    else:
        raise ValueError("candidate_output_format must be 'parquet' or 'csv'")


def _candidate_filename(threshold, output_format="csv"):
    threshold_label = f"{threshold:.0e}".replace("+", "")
    suffix = "parquet" if output_format == "parquet" else "csv"
    return f"IBH_F_bol_gt_{threshold_label}.{suffix}"


def radiative_efficiency_xie_yuan_2012(f_mdot: float | np.ndarray) -> float | np.ndarray:
    """Calculate radiative efficiency following Xie & Yuan (2012).

    The efficiency is parameterized as a function of the dimensionless
    accretion rate f_mdot = m_dot / m_dot_edd.
    """
    f_mdot = np.asarray(f_mdot)
    eta = np.zeros_like(f_mdot)

    mask1 = f_mdot <= 2.9e-5
    eta[mask1] = 0.035 * (f_mdot[mask1] / 2.9e-5) ** 0.65

    mask2 = (f_mdot > 2.9e-5) & (f_mdot <= 3.3e-3)
    eta[mask2] = 0.035 * (f_mdot[mask2] / 2.9e-5) ** 0.076

    mask3 = (f_mdot > 3.3e-3) & (f_mdot <= 5.3e-3)
    eta[mask3] = 0.05 * (f_mdot[mask3] / 3.3e-3) ** 1.12

    mask4 = f_mdot > 5.3e-3
    eta[mask4] = 0.085

    return eta if eta.shape != () else eta.item()


__all__ = [
    "summarize_isolated_bh_accretion",
    "radiative_efficiency_xie_yuan_2012",
]
