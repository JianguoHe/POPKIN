"""Orbital response to supernovae in binary systems."""

import numpy as np
from popkin.utils import conditional_njit
from popkin.constants import G, M_sun, R_sun, km, sec_per_year


@conditional_njit
def _kepler_residual(x: float, e: float, M: float) -> float:
    """Return the residual of Kepler's equation."""
    if e < 1.0:
        return x - e * np.sin(x) - M  # Elliptic orbit: E - e*sin(E) = M
    elif e > 1.0:
        return e * np.sinh(x) - x - M  # Hyperbolic orbit: e*sinh(H) - H = M
    else:
        raise ValueError("Parabolic orbit (e=1) is not supported")


@conditional_njit
def _kepler_residual_derivative(x: float, e: float) -> float:
    """Return the derivative of Kepler's equation residual."""
    if e < 1.0:
        return 1.0 - e * np.cos(x)    # d/dE (E - e*sin(E) - M)
    elif e > 1.0:
        return e * np.cosh(x) - 1.0   # d/dH (e*sinh(H) - H - M)
    else:
        raise ValueError("Parabolic orbit (e=1) is not supported")


@conditional_njit
def _solve_kepler_anomaly(
        e: float,
        M: float,
        max_iterations: int = 1000000,
        tolerance: float = 1e-5
) -> float:
    """Solve Kepler's equation and return the orbital anomaly.

    Parameters
    ----------
    e : float
        Eccentricity (e != 1).
    M : float
        Mean anomaly.
    max_iterations : int
        Maximum number of Newton iterations.
    tolerance : float
        Residual tolerance.

    Returns
    -------
    x : float
        Eccentric anomaly for e < 1, or hyperbolic anomaly for e > 1.
    """
    # Initial guess.
    if e < 1.0:
        if M < np.pi:
            initial_guess = M + 0.85 * e
        else:
            initial_guess = M - 0.85 * e
    else:
        if M < 0.1:
            initial_guess = M / (e - 1.0)
        else:
            initial_guess = np.log(2.0 * M / e + 1.8)

    x = initial_guess

    for _ in range(max_iterations):
        f = _kepler_residual(x, e, M)
        df_dx = _kepler_residual_derivative(x, e)

        if abs(f) < tolerance:
            return x

        if abs(df_dx) < 1e-12:
            raise ValueError("Derivative too small")

        x = x - f / df_dx

    raise ValueError(f"Kepler equation did not converge (e={e}, M={M})")


@conditional_njit
def _sample_kick_velocity_cgs(kick, sigma):
    """Return the natal-kick velocity in cm/s."""
    if kick is not None:
        if kick.shape == (3,):
            return kick * km
        raise ValueError("'kick' is supposed to be a three-dimensional array.")

    if sigma is not None:
        if sigma >= 0.0:
            sigma_cgs = sigma * km
            return np.array([
                np.random.normal(0.0, sigma_cgs),
                np.random.normal(0.0, sigma_cgs),
                np.random.normal(0.0, sigma_cgs)
            ], dtype=np.float64)
        raise ValueError("'sigma' is supposed to be a positive real number.")

    return np.zeros(3, dtype=np.float64)


@conditional_njit
def _sample_initial_orbital_state_cgs(a_cgs, ecc, m_pre_cgs, reff):
    """Sample pre-supernova separation and relative velocity in CGS units."""
    c_cgs = a_cgs * ecc
    b_cgs = np.sqrt(abs(a_cgs ** 2 - c_cgs ** 2))

    if ecc == 1.0:
        raise ValueError("ecc = 1 is not supported")

    if ecc < 1.0:
        mean_anomaly = np.random.uniform(0.0, 2.0 * np.pi)
        elliptic_anomaly = _solve_kepler_anomaly(ecc, mean_anomaly)

        cos_ea = np.cos(elliptic_anomaly)
        sin_ea = np.sin(elliptic_anomaly)

        R_initial_cgs = np.sqrt((a_cgs * cos_ea - c_cgs) ** 2 + (b_cgs * sin_ea) ** 2)
        v_orb_rel_init = np.sqrt(G * m_pre_cgs * (2.0 / R_initial_cgs - 1.0 / a_cgs))

        ecc_cos = ecc * cos_ea
        denominator = 1.0 - ecc_cos * ecc_cos
        ratio_1 = np.sqrt(ecc * ecc / denominator) * sin_ea
        ratio_2 = np.sqrt((1.0 - ecc * ecc) / denominator)

        return R_initial_cgs, v_orb_rel_init * ratio_1, v_orb_rel_init * ratio_2, 0.0

    if reff is None or reff <= 0.0:
        raise ValueError("ecc > 1 requires a positive reff")

    r_effective_cgs = reff * R_sun
    r2 = r_effective_cgs * r_effective_cgs

    if r_effective_cgs >= a_cgs:
        exp_H_effective = (np.sqrt(r2 + b_cgs ** 2) + np.sqrt(r2 - a_cgs ** 2)) / c_cgs
    else:
        exp_H_effective = 1.0

    H_effective = np.log(exp_H_effective)
    mean_anomaly = np.random.uniform(-H_effective, H_effective)
    hyperbolic_anomaly = _solve_kepler_anomaly(ecc, mean_anomaly)

    cosh_ha = np.cosh(hyperbolic_anomaly)
    sinh_ha = np.sinh(hyperbolic_anomaly)

    R_initial_cgs = np.sqrt((c_cgs - a_cgs * cosh_ha) ** 2 + (b_cgs * sinh_ha) ** 2)
    v_orb_rel_init = np.sqrt(G * m_pre_cgs * (2.0 / R_initial_cgs + 1.0 / a_cgs))

    denominator = (ecc * cosh_ha) ** 2 - 1.0
    ratio_1 = np.sqrt(ecc ** 2 / denominator) * sinh_ha
    ratio_2 = np.sqrt((ecc ** 2 - 1.0) / denominator)

    return R_initial_cgs, v_orb_rel_init * ratio_1, v_orb_rel_init * ratio_2, 0.0


@conditional_njit
def post_supernova_orbit(
    a: float | int,
    ecc: float | int,
    m1_pre: float | int,
    m2_pre: float | int,
    m1_post: float | int | None = None,
    m2_post: float | int | None = None,
    kick: np.ndarray | None = None,
    sigma: float | int | None = None,
    impact: float | int = 0.0,
    reff: float | None = None,
):
    """Calculate post-supernova binary orbital parameters.

    Parameters
    ----------
    a : float | int
        Initial semi-major axis [R_sun].
    ecc : float | int
        Initial eccentricity.
    m1_pre : float | int
        Exploding-star mass before the supernova [M_sun].
    m2_pre : float | int
        Companion mass before the supernova [M_sun].
    m1_post : float | int | None, default=None
        Remnant mass after the supernova [M_sun]. If None, use m1_pre.
    m2_post : float | int | None, default=None
        Companion mass after the supernova [M_sun]. If None, use m2_pre.
    kick : np.ndarray | None, default=None
        Natal-kick velocity vector of the exploding star [km/s], shape (3,).
    sigma : float | int | None, default=None
        One-dimensional Gaussian sigma [km/s] used to draw a random kick when
        kick is not provided.
    impact : float | int, default=0.0
        Impact velocity received by the companion [km/s], following
        Tauris et al. (1998).
    reff : float | None, default=None
        Effective gravitational radius [R_sun]. Required for ecc > 1.

    Returns
    -------
    state : str | None
        Orbital state:
        - Bound binary: 'bound'
        - Disrupted binary: 'disrupted'
        - Ia SN: None
    a_output : float | None
        Post-supernova semi-major axis [R_sun].
        - Bound/disrupted systems: float
        - Ia SN: None
    ecc_output : float | None
        Post-supernova eccentricity.
        - Bound/disrupted systems: float
        - Ia SN: None
    h_orbit_output : float | None
        Scalar post-supernova orbital angular momentum [M_sun R_sun^2 / yr].
        - Bound/disrupted systems: float
        - Ia SN: None
    closest_approach : float
        Post-supernova closest approach [R_sun].
        - Bound/disrupted systems: float
        - Ia SN: None
    vc_offset : np.ndarray | None
        Center-of-mass velocity offset [km/s].
        - Bound binary: np.ndarray
        - Disrupted binary/Ia SN: None
    v1_runaway : np.ndarray | None
        Runaway velocity of the remnant [km/s].
        - Bound binary: None
        - Disrupted binary/Ia SN: np.ndarray
    v2_runaway : np.ndarray | None
        Runaway velocity of the companion [km/s].
        - Bound binary: None
        - Disrupted binary/Ia SN: np.ndarray
    radial_motion : str | None
        Post-supernova radial-motion trend ('closing' or 'leaving').
        - Bound binary/Ia SN: None
        - Disrupted binary: str

    Notes
    -----
    - Coordinate convention: +x points from the companion to the exploding
      star, and +z follows the orbital angular-momentum direction.
    - The input kick vector must use this same coordinate convention.
    - Parabolic orbits (ecc = 1) are not supported.
    """
    # Default parameter handling.
    if m1_post is None:
        m1_post = m1_pre
    if m2_post is None:
        m2_post = m2_pre

    # Convert to base CGS units.
    m1_pre_cgs: float  = m1_pre * M_sun
    m2_pre_cgs: float = m2_pre * M_sun
    m1_post_cgs: float = m1_post * M_sun
    m2_post_cgs: float = m2_post * M_sun
    m_pre_cgs: float = m1_pre_cgs + m2_pre_cgs
    m_post_cgs: float = m1_post_cgs + m2_post_cgs

    a_cgs: float = a * R_sun

    kick_cgs = _sample_kick_velocity_cgs(kick, sigma)

    # Impact velocity (km/s -> cm/s).
    impact_cgs: float = impact * km  # cm/s

    R_initial_cgs, v_orb_rel_init_x, v_orb_rel_init_y, v_orb_rel_init_z = (
        _sample_initial_orbital_state_cgs(a_cgs, ecc, m_pre_cgs, reff)
    )

    # Pre-supernova position, relative velocity, and component orbital velocity.
    R_vec_cgs = np.array([R_initial_cgs, 0.0, 0.0], dtype=np.float64)

    v_rel_pre_SN_cgs = np.array([v_orb_rel_init_x, v_orb_rel_init_y, v_orb_rel_init_z], dtype=np.float64)

    v_orb1_pre_SN_cgs = m2_pre_cgs / m_pre_cgs * v_rel_pre_SN_cgs
    v_orb2_pre_SN_cgs = -m1_pre_cgs / m_pre_cgs * v_rel_pre_SN_cgs

    # print("Relative position vector", R_vec_cgs)

    # Impact velocity handling.
    if impact_cgs >= 0.0:
        impact_vec_cgs = -(impact_cgs / R_initial_cgs) * R_vec_cgs
    else:
        raise ValueError("impact must be non-negative")

    # Check whether the event is a Type Ia supernova.
    if m1_post == 0.0:
        state = None
        a_output = None
        ecc_output = None
        h_orbit_output = None
        closest_approach = None
        vc_offset = None
        v1_runaway = None
        v2_runaway = (v_orb2_pre_SN_cgs + impact_vec_cgs) / km
        radial_motion = None
        return state, a_output, ecc_output, h_orbit_output, closest_approach, vc_offset, v1_runaway, v2_runaway, radial_motion

    v_rel_post_SN_cgs = v_rel_pre_SN_cgs + kick_cgs - impact_vec_cgs
    v_rel_norm_cgs = np.sqrt(v_rel_post_SN_cgs[0] ** 2 + v_rel_post_SN_cgs[1] ** 2 + v_rel_post_SN_cgs[2] ** 2)
    # print("Post-supernova relative velocity vector", v_rel_post_SN_cgs)

    # Angular momentum.
    # Specific angular momentum (cm^2/s).
    h_specific_cgs = np.cross(R_vec_cgs, v_rel_post_SN_cgs)
    h_norm_cgs = np.sqrt(h_specific_cgs[0] ** 2 + h_specific_cgs[1] ** 2 + h_specific_cgs[2] ** 2)

    # Total orbital angular momentum = reduced mass x specific angular momentum (g cm^2/s).
    mu_cgs = (m1_post_cgs * m2_post_cgs) / m_post_cgs  # Reduced mass (g).
    h_orbit_cgs = mu_cgs * h_specific_cgs

    # Convert to M_sun R_sun^2 / yr; only the scalar magnitude is returned.
    h_orbit_output = h_orbit_cgs / (M_sun * R_sun ** 2 / sec_per_year)
    h_orbit_output = np.sqrt(h_orbit_output[0] ** 2 + h_orbit_output[1] ** 2 + h_orbit_output[2] ** 2)

    # print("Specific angular momentum:", h_specific_cgs, h_norm_cgs)
    # print("Total orbital angular momentum (M_sun R_sun^2 / yr):", h_orbit_output)

    # Center-of-mass velocity.
    v_com_cgs = (
            (m1_post_cgs * (v_orb1_pre_SN_cgs + kick_cgs) +
             m2_post_cgs * (v_orb2_pre_SN_cgs + impact_vec_cgs)) / m_post_cgs
    )
    # print("Center-of-mass velocity:", v_com_cgs)

    # Orbital energy and eccentricity.
    E_orb_rel_cgs = 0.5 * v_rel_norm_cgs * v_rel_norm_cgs - G * m_post_cgs / R_initial_cgs
    h2_cgs = h_norm_cgs ** 2
    ecc2_kick = 2.0 * E_orb_rel_cgs * h2_cgs / (G * m_post_cgs) ** 2 + 1.0
    if ecc2_kick < 0.0:
        if ecc2_kick > -1e-10:
            ecc2_kick = 0.0
        else:
            raise ValueError(f"Invalid post-supernova eccentricity squared: {ecc2_kick}")
    ecc_kick = np.sqrt(ecc2_kick)
    
    # Determine whether the binary is disrupted.
    R_dot_v_cgs = np.dot(R_vec_cgs, v_rel_post_SN_cgs)

    # Helper constants.
    GM_h2 = G * m_post_cgs / h2_cgs
    A_const = 1.0 / R_initial_cgs - GM_h2
    B_const = -R_dot_v_cgs / (h_norm_cgs * R_initial_cgs)
    Delta = A_const * A_const + B_const * B_const - (GM_h2) ** 2

    if Delta > 0:  # Hyperbolic orbit -> disrupted binary.
        sin_phi = (-GM_h2 * B_const + np.sqrt(Delta) * A_const) / (A_const * A_const + B_const * B_const)
        cos_phi = (-GM_h2 * A_const - np.sqrt(Delta) * B_const) / (A_const * A_const + B_const * B_const)

        v_rel_final_cgs = np.sqrt(v_rel_norm_cgs * v_rel_norm_cgs - 2.0 * G * m_post_cgs / R_initial_cgs)

        k_const = v_rel_final_cgs * (cos_phi - R_dot_v_cgs * sin_phi / h_norm_cgs) / R_initial_cgs
        l_const = R_initial_cgs * v_rel_final_cgs * sin_phi / h_norm_cgs

        v_rel_final_vec_cgs = k_const * R_vec_cgs + l_const * v_rel_post_SN_cgs

        # Final velocities.
        v1_final_cgs = m2_post_cgs / m_post_cgs * v_rel_final_vec_cgs + v_com_cgs
        v2_final_cgs = -m1_post_cgs / m_post_cgs * v_rel_final_vec_cgs + v_com_cgs

        a_kick_cgs = 1.0 / (GM_h2 * (ecc_kick * ecc_kick - 1.0))

        state = 'disrupted'
        if R_dot_v_cgs >= 0.0:
            radial_motion = "leaving"
            closest_approach = R_initial_cgs / R_sun
        else:
            radial_motion = "closing"
            closest_approach = a_kick_cgs * (ecc_kick - 1.0) / R_sun

        ecc_output = ecc_kick
        a_output = a_kick_cgs / R_sun
        vc_offset = None
        v1_runaway = v1_final_cgs / km  # cm/s -> km/s
        v2_runaway = v2_final_cgs / km

    elif Delta < 0:  # Elliptic orbit -> bound binary.
        a_kick_cgs = 1.0 / (GM_h2 * (1.0 - ecc_kick * ecc_kick))

        state = 'bound'
        radial_motion = None
        closest_approach = a_kick_cgs * (1.0 - ecc_kick) / R_sun
        ecc_output = ecc_kick
        a_output = a_kick_cgs / R_sun
        vc_offset = v_com_cgs / km  # cm/s -> km/s
        v1_runaway = None
        v2_runaway = None

    else:  # Delta == 0, parabolic orbit (rare).
        raise ValueError("Parabolic orbit (ecc = 1) is not supported")

    return state, a_output, ecc_output, h_orbit_output, closest_approach, vc_offset, v1_runaway, v2_runaway, radial_motion

# @njit
# def run():
#     # result = post_supernova_orbit(a = 2000, ecc = 0, m1_pre = 10, m2_pre = 8, m1_post = 1.4, m2_post = 7, kick = np.array([100., 100., 200.]), impact=20)
#     result = post_supernova_orbit(a = 1000, ecc = 0, m1_pre = 10, m2_pre = 1.4, m1_post = 10, m2_post = 10, kick = np.array([0., 0., 100.]), )
#
#     print(result)
#     jorb = result[3]
#     print(jorb)
#
# run()
