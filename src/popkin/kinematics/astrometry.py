"""Astrometric coordinate and proper-motion utilities."""

import numpy as np
from astropy import units as u
from astropy.coordinates import Galactic, ICRS, SkyCoord


def convert_equatorial_pm_to_galactic(l, b, pm_ra, pm_dec):
    """Convert equatorial proper motions to Galactic proper motions.

    Args:
        l: Galactic longitude [deg]. Supports scalar or array-like input.
        b: Galactic latitude [deg]. Supports scalar or array-like input.
        pm_ra: Proper motion in right ascension, including cos(dec) [mas/yr].
        pm_dec: Proper motion in declination [mas/yr].

    Returns:
        Tuple ``(pm_l_cosb, pm_b)`` as NumPy arrays without astropy units.
    """
    l = np.asarray(l) * u.deg
    b = np.asarray(b) * u.deg
    pm_ra = np.asarray(pm_ra) * u.mas / u.yr
    pm_dec = np.asarray(pm_dec) * u.mas / u.yr

    galactic_coord = Galactic(l=l, b=b)
    icrs_coord = galactic_coord.transform_to(ICRS())
    final_coord = SkyCoord(
        ra=icrs_coord.ra,
        dec=icrs_coord.dec,
        pm_ra_cosdec=pm_ra,
        pm_dec=pm_dec,
        frame="icrs",
    ).transform_to(Galactic())

    return final_coord.pm_l_cosb.value, final_coord.pm_b.value


def propagate_equatorial_pm_errors_to_galactic(l, b, pm_ra, pm_dec, pm_ra_err, pm_dec_err):
    """Propagate equatorial proper-motion errors to Galactic coordinates.

    Args:
        l: Galactic longitude [deg].
        b: Galactic latitude [deg].
        pm_ra: Proper motion in right ascension, including cos(dec) [mas/yr].
        pm_dec: Proper motion in declination [mas/yr].
        pm_ra_err: Uncertainty in ``pm_ra`` [mas/yr].
        pm_dec_err: Uncertainty in ``pm_dec`` [mas/yr].

    Returns:
        Tuple ``(pm_l_cosb, pm_b, pm_l_cosb_err, pm_b_err)``.
    """
    pm_l_cosb, pm_b = convert_equatorial_pm_to_galactic(l, b, pm_ra, pm_dec)

    delta = 1e-6

    pm_l_ra1, _ = convert_equatorial_pm_to_galactic(l, b, pm_ra + delta, pm_dec)
    pm_l_ra2, _ = convert_equatorial_pm_to_galactic(l, b, pm_ra - delta, pm_dec)
    d_pm_l_d_ra = (pm_l_ra1 - pm_l_ra2) / (2 * delta)

    _, pm_b_ra1 = convert_equatorial_pm_to_galactic(l, b, pm_ra + delta, pm_dec)
    _, pm_b_ra2 = convert_equatorial_pm_to_galactic(l, b, pm_ra - delta, pm_dec)
    d_pm_b_d_ra = (pm_b_ra1 - pm_b_ra2) / (2 * delta)

    pm_l_dec1, _ = convert_equatorial_pm_to_galactic(l, b, pm_ra, pm_dec + delta)
    pm_l_dec2, _ = convert_equatorial_pm_to_galactic(l, b, pm_ra, pm_dec - delta)
    d_pm_l_d_dec = (pm_l_dec1 - pm_l_dec2) / (2 * delta)

    _, pm_b_dec1 = convert_equatorial_pm_to_galactic(l, b, pm_ra, pm_dec + delta)
    _, pm_b_dec2 = convert_equatorial_pm_to_galactic(l, b, pm_ra, pm_dec - delta)
    d_pm_b_d_dec = (pm_b_dec1 - pm_b_dec2) / (2 * delta)

    pm_l_cosb_err = np.sqrt((d_pm_l_d_ra * pm_ra_err) ** 2 + (d_pm_l_d_dec * pm_dec_err) ** 2)
    pm_b_err = np.sqrt((d_pm_b_d_ra * pm_ra_err) ** 2 + (d_pm_b_d_dec * pm_dec_err) ** 2)

    return pm_l_cosb, pm_b, pm_l_cosb_err, pm_b_err


__all__ = [
    "convert_equatorial_pm_to_galactic",
    "propagate_equatorial_pm_errors_to_galactic",
]
