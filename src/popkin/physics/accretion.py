"""Accretion-rate relations for compact objects."""

import numpy as np


def m_dot_edd(m, eta_std: float = 0.1, unit: str = "cgs"):
    """Calculate the Eddington accretion rate.

    Args:
        m: Compact-object mass [unit: M_sun]. Supports scalar or array-like input.
        eta_std: Standard thin-disk radiative efficiency.
        unit: Output unit. Must be ``"cgs"`` [g/s] or ``"msun"`` [M_sun/yr].

    Returns:
        Eddington accretion rate in the requested unit.
    """
    if unit == "cgs":
        return 1.4e18 * m * 0.1 / eta_std
    if unit == "msun":
        return 2.22e-8 * m * 0.1 / eta_std

    raise ValueError(f"Unsupported unit: '{unit}'. Available options: 'cgs', 'msun'")


def m_dot_bondi_hoyle(m, v, n=0.3, mu=1.36, cs=10):
    """Calculate the raw Bondi-Hoyle gas capture rate.

    This function applies no Eddington cap and no hot-accretion-flow
    correction. It is suitable for generic compact-object capture estimates.

    Args:
        m: Compact-object mass [unit: M_sun]. Supports scalar or array-like input.
        v: Relative velocity between object and gas [unit: km/s].
        n: Gas number density [unit: cm^-3].
        mu: Mean molecular weight.
        cs: Gas sound speed [unit: km/s].

    Returns:
        Raw Bondi-Hoyle capture rate [unit: g/s].
    """
    m_arr, v_arr, n_arr, mu_arr, cs_arr = np.broadcast_arrays(
        np.asarray(m, dtype=float),
        np.asarray(v, dtype=float),
        np.asarray(n, dtype=float),
        np.asarray(mu, dtype=float),
        np.asarray(cs, dtype=float),
    )

    scalar_output = m_arr.shape == ()

    m_dot = 3.702e14 * m_arr ** 2 * n_arr * mu_arr / (v_arr ** 2 + cs_arr ** 2) ** 1.5
    return m_dot.item() if scalar_output else m_dot


def m_dot_bh_hot_accretion(m, v, n=0.3, mu=1.36, cs=10, eddington_cap=True):
    """Calculate the inner black-hole accretion rate with hot-flow suppression.

    The raw Bondi-Hoyle capture rate is first calculated with
    :func:`m_dot_bondi_hoyle`. For black holes, low dimensionless accretion
    rates are then reduced by the hot-accretion-flow/outflow prescription used
    in the isolated-BH observable model.

    Args:
        m: Black-hole mass [unit: M_sun]. Supports scalar or array-like input.
        v: Relative velocity between black hole and gas [unit: km/s].
        n: Gas number density [unit: cm^-3].
        mu: Mean molecular weight.
        cs: Gas sound speed [unit: km/s].
        eddington_cap: If true, cap the rate at the Eddington accretion rate
            before applying the hot-flow correction.

    Returns:
        Inner black-hole accretion rate [unit: g/s].
    """
    m_arr, v_arr, n_arr, mu_arr, cs_arr = np.broadcast_arrays(
        np.asarray(m, dtype=float),
        np.asarray(v, dtype=float),
        np.asarray(n, dtype=float),
        np.asarray(mu, dtype=float),
        np.asarray(cs, dtype=float),
    )

    scalar_output = m_arr.shape == ()

    m_dot = np.asarray(m_dot_bondi_hoyle(m_arr, v_arr, n=n_arr, mu=mu_arr, cs=cs_arr), dtype=float)
    m_dot_eddington = m_dot_edd(m_arr)
    f_mdot = m_dot / m_dot_eddington

    if eddington_cap:
        result = np.array(np.minimum(m_dot, m_dot_eddington), dtype=float, copy=True)
    else:
        result = np.array(m_dot, dtype=float, copy=True)

    low_acc_mask = f_mdot <= 0.15

    if not np.any(low_acc_mask):
        return result.item() if scalar_output else result

    m_low = m_arr[low_acc_mask]
    v_low = v_arr[low_acc_mask]
    cs_low = cs_arr[low_acc_mask]
    f_mdot_low = f_mdot[low_acc_mask]
    m_dot_low = m_dot[low_acc_mask]
    low_indices = np.flatnonzero(low_acc_mask)
    result_flat = result.reshape(-1)

    r_disk = 1e4 * (m_low / 9) ** (2 / 3) * (np.sqrt(v_low ** 2 + cs_low ** 2) / 40) ** (-10 / 3)

    mask_big = r_disk > 1000
    if np.any(mask_big):
        r_crit_big = (0.15 / f_mdot_low[mask_big]) ** (1 / 0.47) * 1000
        r_out_big = np.minimum(r_disk[mask_big], r_crit_big)
        corrected_big = m_dot_low[mask_big] * (10 / r_out_big) ** 0.5

        result_flat[low_indices[mask_big]] = corrected_big

    mask_small = r_disk <= 1000
    if np.any(mask_small):
        r_crit_small = (f_mdot_low[mask_small] / 0.15) ** (1 / 0.33) * 1000
        mask_heat = r_crit_small <= r_disk[mask_small]

        if np.any(mask_heat):
            r_out_small = np.maximum(10, r_disk[mask_small][mask_heat])
            corrected_small = m_dot_low[mask_small][mask_heat] * (10 / r_out_small) ** 0.5

            small_indices = low_indices[mask_small]
            result_flat[small_indices[mask_heat]] = corrected_small

    return result.item() if scalar_output else result


__all__ = [
    "m_dot_edd",
    "m_dot_bondi_hoyle",
    "m_dot_bh_hot_accretion",
]
