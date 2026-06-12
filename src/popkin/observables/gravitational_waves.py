"""Gravitational-wave observables for compact binaries."""

import numpy as np

from popkin.constants import sec_per_day


def calc_gw_snr(data, orbit_integration):
    """Calculate LISA signal-to-noise ratios for compact binary systems."""
    import astropy.units as u
    from legwork import source

    gw_snr_list = ["gw_snr", "gw_snr_orbit"] if orbit_integration else ["gw_snr"]
    gw_snr_data = np.full(
        len(data),
        np.nan,
        dtype=[(col, "f8") for col in gw_snr_list],
    )

    f_orb_binary = np.full(len(data), np.nan, dtype=float)
    mask_period = data["period"] > 0
    f_orb_binary[mask_period] = 1 / (data["period"][mask_period] * sec_per_day)

    compact_star_types = ["HeWD", "COWD", "ONeWD", "NS", "BH"]
    mask = (
        np.isin(data["type1"], compact_star_types)
        & np.isin(data["type2"], compact_star_types)
        & (data["origin"] != "")
        & data["bound"]
        & (f_orb_binary > 8e-9)
        & (f_orb_binary < 1e0)
    )

    if mask.any():
        data_compact = data[mask]
        m_1 = data_compact["m1"] * u.Msun
        m_2 = data_compact["m2"] * u.Msun
        f_orb = f_orb_binary[mask] * u.Hz
        ecc = data_compact["ecc"]
        weights = data_compact["num"]

        dist_name_list = ["ini_dist", "dist"] if orbit_integration else ["ini_dist"]
        for dist_name, gw_snr in zip(dist_name_list, gw_snr_list):
            dist = data_compact[dist_name] * u.kpc

            sources = source.Source(
                m_1=m_1,
                m_2=m_2,
                f_orb=f_orb,
                dist=dist,
                ecc=ecc,
                weights=weights,
                interpolate_g=False,
                interpolate_sc=True,
                sc_params={
                    "instrument": "LISA",
                    "custom_psd": None,
                    "t_obs": 4 * u.yr,
                    "L": 2.5e9 * u.m,
                    "approximate_R": False,
                    "confusion_noise": "robson19",
                },
            )
            sources.get_snr()
            gw_snr_data[gw_snr][mask] = sources.snr

    return gw_snr_data


def _data_columns(data):
    if hasattr(data, "columns"):
        return set(data.columns)

    names = getattr(getattr(data, "dtype", None), "names", None)
    if names is not None:
        return set(names)

    return set()


def gw_snr_mask(data, threshold=5.0, snr_column="any"):
    """Return a mask for compact binaries above a gravitational-wave SNR threshold.

    Args:
        data: Structured array or DataFrame containing GW SNR columns.
        threshold: Minimum SNR required for selection.
        snr_column: SNR column to use. Accepted values are ``"gw_snr"``,
            ``"gw_snr_orbit"``, and ``"any"``. The ``"any"`` option selects
            sources above threshold in any available GW SNR column.

    Returns:
        Boolean mask with the same length as ``data``.
    """
    if snr_column == "any":
        data_columns = _data_columns(data)
        columns = [col for col in ("gw_snr", "gw_snr_orbit") if col in data_columns]
        if not columns:
            raise KeyError("No GW SNR columns found. Expected 'gw_snr' or 'gw_snr_orbit'.")

        mask = np.zeros(len(data), dtype=bool)
        for col in columns:
            values = np.asarray(data[col], dtype=float)
            mask |= np.isfinite(values) & (values >= threshold)
        return mask

    if snr_column not in ("gw_snr", "gw_snr_orbit"):
        raise ValueError("snr_column must be one of: 'gw_snr', 'gw_snr_orbit', 'any'.")

    if snr_column not in _data_columns(data):
        raise KeyError(f"GW SNR column not found: {snr_column}")

    values = np.asarray(data[snr_column], dtype=float)
    return np.isfinite(values) & (values >= threshold)


def select_gw_sources(data, threshold=5.0, snr_column="any"):
    """Select compact binaries above a gravitational-wave SNR threshold."""
    return data[gw_snr_mask(data, threshold=threshold, snr_column=snr_column)]
