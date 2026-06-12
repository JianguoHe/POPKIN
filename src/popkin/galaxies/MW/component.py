import numpy as np
from typing import Any
from scipy import integrate
from scipy.stats import beta
from scipy.interpolate import interp1d


# ==================== Galactic structure components ====================

class Component:
    """Base class for Milky Way structural components."""

    def __init__(self,
                 name: str,
                 mass: float,
                 tau_min: float,
                 tau_max: float,
                 Rd: float,
                 zd: float
    ):
        """Initialize a Galactic structural component.

        Args:
            name: Component name, such as 'thin disk', 'thick disk', 'bulge', or 'halo'.
            mass: Component mass [unit: M_sun].
            tau_min: Minimum lookback time [unit: Gyr].
            tau_max: Maximum lookback time [unit: Gyr].
            Rd: Radial scale length [unit: kpc], scalar or one-dimensional array.
            zd: Vertical scale height [unit: kpc].
        """
        self.name = name
        self.mass = mass
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.Rd = Rd
        self.zd = zd

    @staticmethod
    def sfr_function(tau: float | np.ndarray) -> float | np.ndarray:
        """Relative shape function for the star formation rate (SFR).

        Used by the thin- and thick-disk SFR evolution models. The actual SFR is:
            SFR_actual = sfr_scale × sfr_function(tau)

        Args:
            tau: Lookback time [unit: Gyr].
                 - tau = 0 is the present.
                 - tau > 0 is the past.

        Returns:
            Relative SFR shape, with the same shape as the input.
        """
        # Galactic age [unit: Gyr].
        tau_galaxy: float = 12.0

        # Star-formation timescale [unit: Gyr].
        tau_sfr_timescale: float = 6.8

        # Exponential SFR decay model: exp(-(galaxy_age - tau) / timescale).
        return np.exp(-(tau_galaxy - tau) / tau_sfr_timescale)

    def _calculate_sfr_scale(self) -> float:
        """Calculate the SFR normalization for the thin/thick disk.

        Formula: sfr_scale = mass / (∫ sfr_function dtau × 1e9)

        Returns:
            SFR normalization [unit: M_sun/yr].

        Raises:
            ValueError: If the integral is not positive.
        """
        n_points = 1000
        tau_grid = np.linspace(self.tau_min, self.tau_max, n_points)
        sfr_values = self.sfr_function(tau_grid)

        integral = integrate.trapezoid(sfr_values, tau_grid)

        if integral <= 0:
            raise ValueError(f"SFR integral must be positive, got {integral:.2e}")

        # 1 Gyr = 1e9 years
        return self.mass / (integral * 1e9)

    def sfr(self, tau: float | np.ndarray) -> float | np.ndarray:
        """Calculate the star formation rate.

        Args:
            tau: Lookback time [unit: Gyr]; 0 is the present.

        Returns:
            SFR [unit: M_sun/yr].
        """
        raise NotImplementedError

    def vertical_dist(self, z: float | np.ndarray) -> float | np.ndarray:
        """Vertical distribution function with an exponential profile.

        Args:
            z: Vertical height [unit: kpc].

        Returns:
            Vertical probability density [unit: kpc^-1].
        """
        return np.exp(-np.abs(z) / self.zd) / self.zd

    def vertical_cdf(self, z: float | np.ndarray) -> float | np.ndarray:
        """Vertical cumulative distribution function.

        Args:
            z: Vertical height [unit: kpc].

        Returns:
            Cumulative probability in [0, 1].
        """
        return 1 - np.exp(-np.abs(z) / self.zd)

    @property
    def info(self) -> dict[str, Any]:
        """Basic component information.

        Returns:
            Dictionary containing component parameters.
        """
        return {
            'name': self.name,
            'mass': f"{self.mass:.1e} M_sun",
            'formation time range': f"{self.tau_min:.2f} - {self.tau_max:.2f} Gyr",
            'radial scale length': f"{self.Rd:.2f} kpc",
            'vertical scale height': f"{self.zd:.2f} kpc",
            'type': self.__class__.__name__
        }


class ThinDisk(Component):
    """Milky Way thin-disk component.

    The thin disk is the younger, more metal-rich stellar component of the Milky Way.

    Common methods:
        sfr(tau)            - star formation rate [M_sun/yr]
        get_Rd(tau)         - time-dependent radial scale length
        radial_dist(R, tau) - radial probability density [kpc^-1]
        radial_cdf(R, tau)  - radial cumulative distribution [0-1]
        vertical_dist(z)    - vertical probability density [kpc^-1]
        vertical_cdf(z)     - vertical cumulative distribution [0-1]
    """

    def __init__(
            self,
            mass: float = 2.6e10,
            tau_min: float = 0.0,
            tau_max: float = 8.0,
            Rd: float = 4.0,
            zd: float = 0.3
    ):
        """Initialize the thin disk.

        Args:
            mass: Thin-disk mass [unit: M_sun].
            tau_min: Minimum lookback time [unit: Gyr].
            tau_max: Maximum lookback time [unit: Gyr].
            Rd: Radial scale length [unit: kpc].
            zd: Vertical scale height [unit: kpc].
        """
        super().__init__('thin disk', mass, tau_min, tau_max, Rd, zd)
        self.sfr_scale = self._calculate_sfr_scale()

    def sfr(self, tau: float | np.ndarray) -> float | np.ndarray:
        """Calculate the star formation rate.

        Args:
            tau: Lookback time [unit: Gyr]; 0 is the present.

        Returns:
            SFR [unit: M_sun/yr].
        """
        condition = (tau >= self.tau_min) & (tau < self.tau_max)
        return np.where(condition, self.sfr_scale * self.sfr_function(tau), 0)

    def get_Rd(self, tau: float | np.ndarray | None = None) -> float | np.ndarray:
        """Return the radial scale length, optionally with time evolution.

        The thin-disk radial scale length follows:
            Rd(tau) = Rd_max × (1 - α × tau / tau_max)
        where Rd_max = 4 kpc and α = 0.3.

        Args:
            tau: Lookback time [unit: Gyr].
                 - None: return the fixed value (4.0 kpc).
                 - float: return the time-dependent value.
                 - np.ndarray: return an array.

        Returns:
            Radial scale length [unit: kpc].

        Examples:
            >>> disk = ThinDisk()
            >>> disk.Rd                                 # 4.0 (fixed value)
            >>> disk.get_Rd()                           # 4.0 (present)
            >>> disk.get_Rd(4)                          # 3.4 (4 Gyr ago)
            >>> disk.get_Rd(np.array([0, 4, 8]))        # [4.0, 3.4, 2.8]
        """
        if tau is None:
            return self.Rd
        else:
            alpha = 0.3                                 # inside-out growth parameter
            return self.Rd * (1 - alpha * (tau / self.tau_max))

    def radial_dist(self, R: float | np.ndarray, tau: float | np.ndarray) -> float | np.ndarray:
        """Calculate the radial distribution function.

        Uses the exponential-disk model: p(R) = (R / Rd^2) × exp(-R / Rd).

        Args:
            R: Galactocentric radial distance [unit: kpc].
            tau: Lookback time [unit: Gyr].

        Returns:
            Radial probability density [unit: kpc^-1].
        """
        Rd_tau = self.get_Rd(tau)
        return np.exp(-R / Rd_tau) * R / Rd_tau ** 2

    def radial_cdf(self, R: float | np.ndarray, tau: float | np.ndarray) -> float | np.ndarray:
        """Calculate the radial cumulative distribution function.

        Args:
            R: Galactocentric radial distance [unit: kpc].
            tau: Lookback time [unit: Gyr].

        Returns:
            Cumulative probability in [0, 1].
        """
        Rd_tau = self.get_Rd(tau)  # Current-time radial scale length.
        return 1 - (1 + R / Rd_tau) * np.exp(-R / Rd_tau)


class ThickDisk(Component):
    """Milky Way thick-disk component.

    The thick disk is the older, more metal-poor stellar disk component.

    Common methods mirror the thin-disk interface for SFR and spatial distributions.
    """

    def __init__(
            self,
            mass: float = 2.6e10,
            tau_min: float = 8.0,
            tau_max: float = 12.0,
            Rd: float = 2.326,
            zd: float = 0.95
    ):
        """Initialize the thick disk.

        Args:
            mass: Thick-disk mass [unit: M_sun].
            tau_min: Minimum lookback time [unit: Gyr].
            tau_max: Maximum lookback time [unit: Gyr].
            Rd: Radial scale length [unit: kpc].
            zd: Vertical scale height [unit: kpc].
        """
        super().__init__('thick disk', mass, tau_min, tau_max, Rd, zd)
        self.sfr_scale = self._calculate_sfr_scale()

    def sfr(self, tau: float | np.ndarray) -> float | np.ndarray:
        """Calculate the star formation rate.

        Args:
            tau: Lookback time [unit: Gyr]; 0 is the present.

        Returns:
            SFR [unit: M_sun/yr].
        """
        condition = (tau >= self.tau_min) & (tau <= self.tau_max)
        return np.where(condition, self.sfr_scale * self.sfr_function(tau), 0)

    def radial_dist(self, R: float | np.ndarray) -> float | np.ndarray:
        """Calculate the radial distribution function.

        Uses the exponential-disk model: p(R) = (R / Rd^2) × exp(-R / Rd).

        Args:
            R: Galactocentric radial distance [unit: kpc].

        Returns:
            Radial probability density [unit: kpc^-1].
        """
        return np.exp(-R / self.Rd) * R / self.Rd ** 2

    def radial_cdf(self, R: float | np.ndarray) -> float | np.ndarray:
        """Calculate the radial cumulative distribution function.

        Args:
            R: Galactocentric radial distance [unit: kpc].

        Returns:
            Cumulative probability in [0, 1].
        """
        return 1 - (1 + R / self.Rd) * np.exp(-R / self.Rd)


class Bulge(Component):
    """Milky Way bulge component.

    The bulge is the central stellar component, dominated by old stars.

    Common methods mirror the disk components for SFR and spatial distributions.
    """

    def __init__(
            self,
            mass: float = 9e9,
            tau_min: float = 6.0,
            tau_max: float = 12.0,
            Rd: float = 1.5,
            zd: float = 0.2
    ):
        """Initialize the bulge.

        Args:
            mass: Bulge mass [unit: M_sun].
            tau_min: Minimum lookback time [unit: Gyr].
            tau_max: Maximum lookback time [unit: Gyr].
            Rd: Radial scale length [unit: kpc].
            zd: Vertical scale height [unit: kpc].
        """
        super().__init__('bulge', mass, tau_min, tau_max, Rd, zd)
        self._precompute_sfr_grid()

    def _precompute_sfr_grid(self) -> None:
        """Precompute the SFR grid once during initialization.

        The bulge star formation history is modeled with a Beta distribution:
            Beta(α=2, β=3) over [6, 12] Gyr.
        """
        n_points = 1000

        # Time grid.
        self._tau_grid = np.linspace(self.tau_min, self.tau_max, n_points)

        # Normalize time to [0, 1].
        tau_normalized = (self._tau_grid - self.tau_min) / (self.tau_max - self.tau_min)

        # Beta probability density (α=2, β=3).
        beta_pdf = beta(2, 3).pdf(tau_normalized)

        # Convert to SFR [unit: M_sun/yr].
        # Normalize via ∫ SFR dt = mass; dt is in Gyr and must be converted to years.
        integral = integrate.trapezoid(beta_pdf, self._tau_grid)
        self._sfr_grid = beta_pdf / integral * self.mass / 1e9

        # Linear interpolation.
        self._interp_sfr = self._create_interpolation()

    def _create_interpolation(self) -> interp1d:
        """Create the SFR interpolation function.

        Returns:
            Interpolation function mapping tau to SFR.
        """
        return interp1d(
            self._tau_grid,
            self._sfr_grid,
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )

    def sfr(self, tau: float | np.ndarray) -> float | np.ndarray:
        """Calculate the star formation rate.

        Uses the precomputed interpolation function.

        Args:
            tau: Lookback time [unit: Gyr]; 0 is the present.

        Returns:
            SFR [unit: M_sun/yr].
        """
        return self._interp_sfr(np.asarray(tau))

    def radial_dist(self, R: float | np.ndarray) -> float | np.ndarray:
        """Calculate the radial distribution function.

        Uses the exponential model: p(R) = (R / Rd^2) × exp(-R / Rd).

        Args:
            R: Galactocentric radial distance [unit: kpc].

        Returns:
            Radial probability density [unit: kpc^-1].
        """
        return np.exp(-R / self.Rd) * R / self.Rd ** 2

    def radial_cdf(self, R: float | np.ndarray) -> float | np.ndarray:
        """Calculate the radial cumulative distribution function.

        Args:
            R: Galactocentric radial distance [unit: kpc].

        Returns:
            Cumulative probability in [0, 1].
        """
        return 1 - (1 + R / self.Rd) * np.exp(-R / self.Rd)


