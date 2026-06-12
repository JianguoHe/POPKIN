import numpy as np
from typing import Any
from popkin.constants import M_sun, pc, kpc, m_p


# ==================== ISM phase classes ====================

class ISMPhase:
    """Base class for an interstellar-medium (ISM) phase.

    Represents one ISM phase, such as molecular clouds, cold/warm neutral media,
    or warm/hot ionized media.
    """

    def __init__(
            self,
            name: str,
            temperature: float,
            n_min: float | None = None,
            n_max: float | None = None,
            beta: float | None = None
    ):
        """Initialize an ISM phase.

        Args:
            name: Phase name, such as 'molecular clouds', 'cold HI', 'warm HI', 'warm HII', or 'hot HII'.
            temperature: Temperature [unit: K].
            n_min: Minimum number density [unit: cm^-3], used for molecular clouds and cold HI.
            n_max: Maximum number density [unit: cm^-3], used for molecular clouds and cold HI.
            beta: Power-law index of the density distribution, used to compute average density.
        """
        self.name = name
        self.temperature = temperature
        self.n_min = n_min
        self.n_max = n_max
        self.beta = beta

    def calculate_average_density(self) -> float:
        """Calculate average number density for molecular clouds or cold HI."""
        if self.n_min is None or self.n_max is None or self.beta is None:
            raise ValueError("n_min, n_max, and beta must be set before calculating average density")

        if self.beta <= 1:
            raise ValueError(f"beta must be > 1, got {self.beta}")
        elif self.beta == 2:
            return self.n_min * self.n_max * (np.log(self.n_max) - np.log(self.n_min)) / (self.n_max - self.n_min)
        else:
            beta = self.beta
            n_min = self.n_min
            n_max = self.n_max

            numerator = (beta - 1) / (beta - 2)
            integral_numerator = n_min ** (2 - beta) - n_max ** (2 - beta)
            integral_denominator = n_min ** (1 - beta) - n_max ** (1 - beta)

            return numerator * integral_numerator / integral_denominator

    def get_discrete_number_density(self, n_points: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Get discrete number density values and their weights for power-law distribution

        Args:
            n_points: Number of discrete points

        Returns:
            (n_values, weights): Arrays of density values and normalized weights

        Notes:
            The weight for each bin is proportional to ∫ n^(-beta) dn over the bin interval.
            Bins are equally spaced in log space.
        """
        if self.n_min is None or self.n_max is None or self.beta is None:
            raise ValueError("n_min, n_max, and beta must be set before discretizing number density")

        # Generate bin edges in log space
        log_n_edges = np.linspace(np.log(self.n_min), np.log(self.n_max), n_points + 1)
        n_edges = np.exp(log_n_edges)

        # Calculate bin centers (geometric mean)
        n_list = np.sqrt(n_edges[:-1] * n_edges[1:])

        # Calculate weights: ∫ n^(-beta) dn from n_i to n_{i+1}
        if self.beta == 1:
            # Special case: ∫ dn/n = ln(n)
            weights = np.log(n_edges[1:] / n_edges[:-1])
        else:
            # General case: ∫ n^(-beta) dn = n^(1-beta) / (1-beta)
            weights = (n_edges[1:] ** (1 - self.beta) - n_edges[:-1] ** (1 - self.beta)) / (1 - self.beta)

        # Normalize weights to sum to 1
        weights = weights / weights.sum()

        return n_list, weights

    def filling_fraction(self, R: float, z: float = 0.0) -> float:
        """Calculate the filling fraction."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement filling_fraction()"
        )

    @staticmethod
    def _check_shape_match(R: float | np.ndarray, z: float | np.ndarray) -> None:
        """Check whether R and z have compatible shapes.

        Args:
            R: Radial distance array.
            z: Vertical height array.

        Raises:
            ValueError: If R and z are both arrays with different lengths.
        """
        if isinstance(R, np.ndarray) and isinstance(z, np.ndarray) and len(R) != len(z):
            raise ValueError(
                f"R and z must have the same length: R={len(R)}, z={len(z)}"
            )

    @property
    def info(self) -> dict[str, Any]:
        """Basic ISM phase information."""
        return {
            'name': self.name,
            'phase': self.__class__.__name__,
            'temperature': f"{self.temperature:.0f} K"
        }


class MolecularClouds(ISMPhase):
    """Molecular-cloud ISM phase.

    Molecular clouds are the densest ISM phase and the main sites of star formation.

    Properties:
        - temperature: ~10 K
        - number density: 100 - 1e5 cm^-3
        - scale height: ~75 pc
        - molecular weight: mu = 2.72, dominated by H2 + He

    Common methods:
        surface_density(R)            - surface density [unit: M_sun/pc^2]
        filling_fraction(R, z=0)      - filling fraction, with z=0 at the Galactic midplane by default
        cs(n)                         - effective sound velocities [unit: km s⁻¹]
        get_discrete_number_density   - discrete number density distribution, returning (n, weight)

    Attributes:
        n                             - average number density [unit: cm^-3]
        mu                            - mean molecular weight
        H                             - scale height [unit: pc]
        info                          - basic information
    """

    def __init__(self):
        """Initialize molecular clouds.

        Notes:
            - Surface-density distribution is fitted from Galactic CO survey observations.
            - Vertical distribution follows an exponential profile.
        """
        super().__init__('molecular clouds', temperature=10.0)

        # Physical parameters.
        self.H: float = 75.0                    # Scale height [unit: pc].
        self.n_min: float = 100.0               # Minimum number density [unit: cm^-3].
        self.n_max: float = 1e5                 # Maximum number density [unit: cm^-3].
        self.beta: float = 2.8                  # Power-law index of the density distribution.
        self.mu: float = 2.72                   # Mean molecular weight (H2 + He).

        # Average density.
        self.n: float = self.calculate_average_density()

    @staticmethod
    def surface_density(R: float | np.ndarray) -> float | np.ndarray:
        """Molecular-cloud surface density for R < 12 kpc, fitted from Nakanishi & Sofue (2016).

        Piecewise function fitted from Galactic CO survey observations [unit: M_sun/pc^2].

        Args:
            R: Galactocentric radial distance [unit: kpc].

        Returns:
            Surface density.

        Notes:
            Piecewise model:
                - R <= 4.54 kpc: fourth-order polynomial fit for the central region.
                - 4.54 < R <= 12 kpc: fourth-order polynomial fit for the disk region.
                - R > 12 kpc: 0, no molecular-cloud component in the outer disk.
        """
        conditions = [
            R <= 4.54,
            (R > 4.54) & (R <= 12),
            R > 12
        ]

        results = [
            0.48961 * R ** 4 - 6.20856 * R ** 3 + 29.26366 * R ** 2 - 60.83331 * R + 50.58674,
            -0.00534192 * R ** 4 + 0.185167 * R ** 3 - 2.23903 * R ** 2 + 10.40043 * R - 11.52203,
            0.0
        ]

        result = np.select(conditions, results)

        # Enforce non-negative surface density.
        return np.maximum(result, 0.0)

    def filling_fraction(
            self,
            R: float | np.ndarray,
            z: float | np.ndarray = 0.0
    ) -> float | np.ndarray:
        """Calculate the molecular-cloud filling fraction.


        The filling fraction varies with radial distance and vertical height:
            f(R, z) = f₀(R) × exp(-|z| / H)

        where:
            f₀(R) = Σ(R) / (2 H n μ)

        Args:
            R: Galactocentric radial distance [unit: kpc].
            z: Vertical height [unit: kpc], with z=0 at the Galactic midplane by default.

        Returns:
            Filling fraction in [0, 1].
        """
        # Check shapes.
        self._check_shape_match(R, z)

        # Convert surface density from [M_sun/pc^2] to [g/cm^2].
        sigma = self.surface_density(R) * M_sun / pc ** 2

        # Vertical scale height [cm].
        H = self.H * pc

        # Mean molecular mass [g].
        mu = self.mu * m_p

        # Normalization factor.
        f0 = sigma / (2 * H * self.n * mu)

        # Vertical exponential decay.
        return f0 * np.exp(-np.abs(z * kpc) / H)

    @staticmethod
    def cs(n):
        return 3.7 * (100 / n) ** 0.35

    @property
    def density_info(self) -> dict[str, str]:
        """Density-related information."""
        return {
            'average_density': f"{self.n:.2f} cm⁻³",
            'density_range': f"{self.n_min:.0f} - {self.n_max:.0f} cm⁻³",
            'beta': f"{self.beta:.1f}",
            'mean_molecular_weight': f"{self.mu:.2f}"
        }

    @property
    def info(self) -> dict[str, Any]:
        """Complete molecular-cloud information."""
        base_info = super().info
        base_info.update({
            'description': 'Molecular clouds - the densest phase of ISM, site of star formation',
            'scale height': f"{self.H:.0f} pc",
        })
        base_info.update(self.density_info)
        return base_info


class HIBase(ISMPhase):
    """Base class for neutral hydrogen phases.

    HI is split into cold neutral medium (CNM) and warm neutral medium (WNM);
    this class contains their shared behavior.
    """

    def __init__(self, name: str, temperature: float):
        """Initialize an HI phase.

        Args:
            name: Phase name, either 'cold HI' or 'warm HI'.
            temperature: Temperature [unit: K].
        """
        super().__init__(name, temperature)

        # Shared parameters.
        self.H: float | None = None         # Scale height [unit: pc].
        self.n: float | None = None         # Average number density [unit: cm^-3].
        self.mu: float = 1.36               # Mean molecular weight in proton-mass units.
        self.cs: float = 10.                # effective sound velocities [unit: km s⁻¹]

    @staticmethod
    def total_surface_density(R: float | np.ndarray) -> float | np.ndarray:
        """Total HI surface density, cold plus warm, for R < 31 kpc.

        Piecewise function fitted from Nakanishi & Sofue (2016) Galactic HI survey data
        [unit: M_sun/pc^2].

        Args:
            R: Galactocentric radial distance [unit: kpc].

        Returns:
            Total HI surface density.

        Notes:
            Piecewise model:
                - R <= 7.5 kpc: cubic polynomial for the central region.
                - 7.5 < R <= 12.5 kpc: cubic polynomial for the inner disk.
                - 12.5 < R <= 30 kpc: fourth-order polynomial for the outer disk.
                - R > 30 kpc: 0
        """
        conditions = [
            R <= 7.5,
            (R > 7.5) & (R <= 12.5),
            (R > 12.5) & (R <= 30),
            R > 30
        ]

        results = [
            -0.0124129 * R ** 3 + 0.283338 * R ** 2 - 1.01765 * R + 2.59463,
            0.0632278 * R ** 3 - 2.35709 * R ** 2 + 28.58025 * R - 102.72103,
            0.000263413 * R ** 4 - 0.0274939 * R ** 3 + 1.07473 * R ** 2 - 18.70121 * R + 122.90123,
            0.0
        ]

        return np.select(conditions, results)

    @staticmethod
    def cold_fraction(R: float | np.ndarray) -> float | np.ndarray:
        """Fraction of cold HI in total HI.

        Args:
            R: Galactocentric radial distance [unit: kpc].

        Returns:
            Cold-HI fraction in [0, 1].

        Notes:
            Piecewise model:
                - R ≤ 0.5 kpc: 50%
                - 0.5 < R ≤ 1 kpc: linear decline, 0.8 - 0.6R
                - 1 < R ≤ 11 kpc: 20%, constant
                - 11 < R ≤ 13 kpc: linear decline, 1.3 - 0.1R
                - R > 13 kpc: 0%
        """
        conditions = [
            R <= 0.5,
            (R > 0.5) & (R <= 1),
            (R > 1) & (R <= 11),
            (R > 11) & (R <= 13),
            R > 13
        ]

        results = [
            0.5,
            0.8 - 0.6 * R,
            0.2,
            1.3 - 0.1 * R,
            0.0
        ]

        return np.select(conditions, results)

    def phase_factor(self, R: float | np.ndarray) -> float | np.ndarray:
        """Phase factor, i.e. the fraction of total HI in the current phase.

        Subclasses must implement:
            - ColdHI: return cold_fraction(R)
            - WarmHI: return 1 - cold_fraction(R)

        Args:
            R: Galactocentric radial distance [unit: kpc].

        Returns:
            Current-phase fraction in [0, 1].
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement phase_factor()"
        )

    def surface_density(self, R: float | np.ndarray) -> float | np.ndarray:
        """Surface density of the current phase [unit: M_sun/pc^2].

        Args:
            R: Galactocentric radial distance [unit: kpc].

        Returns:
            Current-phase surface density.
        """
        total = self.total_surface_density(R)
        factor = self.phase_factor(R)
        return total * factor

    def filling_fraction(
            self,
            R: float | np.ndarray,
            z: float | np.ndarray = 0.0
    ) -> float | np.ndarray:
        """Calculate the filling fraction.

        Model: f(R, z) = f0(R) * exp(-|z| / H),
        where f0(R) = Sigma(R) / (2 H n mu).

        Args:
            R: Galactocentric radial distance [unit: kpc].
            z: Vertical height [unit: kpc], with z=0 at the Galactic midplane by default.

        Returns:
            Filling fraction in [0, 1].
        """
        if self.H is None:
            raise ValueError(f"{self.__class__.__name__} has no H value")
        if self.n is None:
            raise ValueError(f"{self.__class__.__name__} has no n value")

        # Check shapes.
        self._check_shape_match(R, z)

        # Convert surface density from [M_sun/pc^2] to [g/cm^2].
        sigma = self.surface_density(R) * M_sun / pc ** 2

        # Vertical scale height [cm].
        H = self.H * pc

        # Mean molecular mass [g].
        mu = self.mu * m_p

        # Radial normalization factor.
        f0 = sigma / (2 * H * self.n * mu)

        # Vertical exponential decay.
        return f0 * np.exp(-np.abs(z * kpc) / H)


class ColdHI(HIBase):
    """Cold HI, or the Cold Neutral Medium (CNM).

    Properties:
        - temperature: ~100 K
        - number density: 10 - 100 cm^-3
        - scale height: ~150 pc
        - density-distribution power-law index: beta ~= 3.8

    Common methods:
        surface_density(R)            - surface density [unit: M_sun/pc^2]
        total_surface_density(R)      - total HI surface density [unit: M_sun/pc^2]
        filling_fraction(R, z=0)      - filling fraction, with z=0 at the Galactic midplane by default
        get_discrete_number_density   - discrete number density distribution, returning (n, weight)

    Attributes:
        n                             - average number density [unit: cm^-3]
        mu                            - mean molecular weight
        H                             - scale height [unit: pc]
        cs                            - effective sound velocities [unit: km s⁻¹]
        info                          - basic information
    """

    def __init__(self):
        """Initialize cold HI."""
        super().__init__('cold HI', temperature=100.0)

        # Physical parameters.
        self.H = 150.0                  # Scale height [unit: pc].
        self.n_min = 10.0               # Minimum number density [unit: cm^-3].
        self.n_max = 100.0              # Maximum number density [unit: cm^-3].
        self.beta = 3.8                 # Density-distribution power-law index.
        self.mu = 1.36                  # Mean molecular weight.

        # Average density.
        self.n = self.calculate_average_density()

    def phase_factor(self, R: float | np.ndarray) -> float | np.ndarray:
        """Fraction of total HI in cold HI."""
        return self.cold_fraction(R)

    @property
    def density_info(self) -> dict[str, str]:
        """Density-related information."""
        return {
            'average density': f"{self.n:.2f} cm⁻³",
            'density range': f"{self.n_min:.0f} - {self.n_max:.0f} cm⁻³",
            'beta': f"{self.beta:.1f}",
        }

    @property
    def info(self) -> dict[str, Any]:
        """Complete cold-HI information."""
        base_info = super().info
        base_info.update({
            'description': 'Cold Neutral Medium (CNM) - the denser phase of HI',
            'scale_height': f"{self.H:.0f} pc",
        })
        base_info.update(self.density_info)
        return base_info


class WarmHI(HIBase):
    """Warm HI, or the Warm Neutral Medium (WNM).

    Properties:
        - temperature: ~8000 K
        - number density: ~0.3 cm^-3, constant-density model
        - scale height: ~500 pc

    Common methods:
        surface_density(R)            - surface density [unit: M_sun/pc^2]
        total_surface_density(R)      - total HI surface density [unit: M_sun/pc^2]
        filling_fraction(R, z=0)      - filling fraction, with z=0 at the Galactic midplane by default

    Attributes:
        n                             - average number density [unit: cm^-3]
        mu                            - mean molecular weight
        H                             - scale height [unit: pc]
        cs                            - effective sound velocities [unit: km s⁻¹]
        info                          - basic information
    """

    def __init__(self):
        """Initialize warm HI."""
        super().__init__('warm HI', temperature=8000.0)

        # Physical parameters.
        self.H = 500.0                  # Scale height [unit: pc].
        self.n = 0.3                    # Average number density [unit: cm^-3].
        self.mu = 1.36                  # Mean molecular weight.

    def phase_factor(self, R: float | np.ndarray) -> float | np.ndarray:
        """Fraction of total HI in warm HI."""
        return 1 - self.cold_fraction(R)

    @property
    def density_info(self) -> dict[str, str]:
        """Density-related information."""
        return {
            'average_density': f"{self.n:.2f} cm⁻³",
        }

    @property
    def info(self) -> dict[str, Any]:
        """Complete warm-HI information."""
        base_info = super().info
        base_info.update({
            'description': 'Warm Neutral Medium (WNM) - the diffuse phase of HI',
            'scale_height': f"{self.H:.0f} pc",
        })
        base_info.update(self.density_info)
        return base_info


class WarmHII(ISMPhase):
    """Warm HII, or the Warm Ionized Medium (WIM).

    Properties:
        - temperature: ~8000 K
        - number density: ~0.15 cm^-3
        - scale height: ~1000 pc
        - filling factor: ~0.2 at the Galactic midplane

    Common methods:
        filling_fraction(R=8, z=0)     - filling fraction, with z=0 at the Galactic midplane by default

    Attributes:
        n                             - average number density [unit: cm^-3]
        mu                            - mean molecular weight
        H                             - scale height [unit: pc]
        cs                            - effective sound velocities [unit: km s⁻¹]
        info                          - basic information
    """

    def __init__(self):
        """Initialize warm HII."""
        super().__init__('warm HII', temperature=8000.0)

        # Physical parameters.
        self.H = 1000.0                 # Scale height [unit: pc].
        self.f0 = 0.2                   # Midplane filling factor.
        self.n = 0.15                   # Average number density [unit: cm^-3].
        self.mu = 1.36                  # Mean molecular weight.
        self.cs = 10.                   # effective sound velocities [unit: km s⁻¹]

    def filling_fraction(
            self,
            R: float | np.ndarray = 8.0,
            z: float | np.ndarray = 0.0
    ) -> float | np.ndarray:
        """Calculate the filling fraction.

        Model: f(z) = f0 × exp(-|z| / H).

        Args:
            R: Galactocentric radial distance [unit: kpc]. This model does not depend on R.
            z: Vertical height [unit: kpc], with z=0 at the Galactic midplane by default.

        Returns:
            Filling fraction in [0, 1].
        """
        # Check shapes.
        self._check_shape_match(R, z)

        # Calculate filling fraction.
        vertical = self.f0 * np.exp(-np.abs(z * kpc) / (self.H * pc))

        if isinstance(R, np.ndarray) and not isinstance(z, np.ndarray):
            return np.full_like(R, vertical, dtype=float)
        else:
            return vertical

    @property
    def density_info(self) -> dict[str, str]:
        """Density-related information."""
        return {
            'average_density': f"{self.n:.2f} cm⁻³",
            'filling_factor': f"{self.f0:.1f} (at z=0)"
        }

    @property
    def info(self) -> dict[str, Any]:
        """Complete warm-HII information."""
        base_info = super().info
        base_info.update({
            'description': 'Warm Ionized Medium (WIM) - the diffuse ionized phase of ISM',
            'scale_height': f"{self.H:.0f} pc",
        })
        base_info.update(self.density_info)
        return base_info


class HotHII(ISMPhase):
    """Hot HII, or the Hot Ionized Medium (HIM).

    Properties:
        - temperature: ~1e6 K
        - number density: ~0.002 cm^-3
        - scale height: ~3000 pc
        - filling factor: ~0.4 at the Galactic midplane

    Common methods:
        filling_fraction(R=8, z=0)     - filling fraction, with z=0 at the Galactic midplane by default

    Attributes:
        n                             - average number density [unit: cm^-3]
        mu                            - mean molecular weight
        H                             - scale height [unit: pc]
        cs                            - effective sound velocities [unit: km s⁻¹]
        info                          - basic information
    """

    def __init__(self):
        """Initialize hot HII."""
        super().__init__('hot HII', temperature=1e6)

        # Physical parameters.
        self.H = 3000.0                 # Scale height [unit: pc].
        self.f0 = 0.4                   # Midplane filling factor.
        self.n = 0.002                  # Average number density [unit: cm^-3].
        self.mu = 1.36                  # Mean molecular weight.
        self.cs = 150.                  # effective sound velocities [unit: km s⁻¹]


    def filling_fraction(
            self,
            R: float | np.ndarray = 8.0,
            z: float | np.ndarray = 0.0
    ) -> float | np.ndarray:
        """Calculate the filling fraction.

        Model: f(z) = f0 × exp(-|z| / H).

        Args:
            R: Galactocentric radial distance [unit: kpc]. This model does not depend on R.
            z: Vertical height [unit: kpc], with z=0 at the Galactic midplane by default.

        Returns:
            Filling fraction in [0, 1].
        """
        # Check shapes.
        self._check_shape_match(R, z)

        # Calculate filling fraction.
        vertical = self.f0 * np.exp(-np.abs(z * kpc) / (self.H * pc))

        if isinstance(R, np.ndarray) and not isinstance(z, np.ndarray):
            return np.full_like(R, vertical, dtype=float)
        else:
            return vertical

    @property
    def density_info(self) -> dict[str, str]:
        """Density-related information."""
        return {
            'average_density': f"{self.n:.3f} cm⁻³",
            'filling_factor': f"{self.f0:.1f} (at z=0)"
        }

    @property
    def info(self) -> dict[str, Any]:
        """Complete hot-HII information."""
        base_info = super().info
        base_info.update({
            'description': 'Hot Ionized Medium (HIM) - the hot coronal phase of ISM',
            'scale_height': f"{self.H:.0f} pc",
        })
        base_info.update(self.density_info)
        return base_info


    # @staticmethod
    # def surface_density(R: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    #     """Surface density (M_sun/pc^2)."""
    #     raise NotImplementedError
