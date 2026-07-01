# src/popkin/kinematics/orbit.py - single/binary orbit evolution with galpy.
# In current tests, galpy's dop853_c integrator is the fastest option, so it is used by default.
# The Galactic Center supermassive black hole is disabled by default and can be enabled by configuration.

import numpy as np
import astropy.units as u
from typing import Literal
from galpy.orbit import Orbit
from galpy.util import conversion
from galpy.potential import MWPotential2014, KeplerPotential
from popkin.utils import rotate_velocity_offset_to_galactocentric


class OrbitIntegrator:
    """Orbit integrator."""

    def __init__(
            self,
            data: np.ndarray,
            obj_type: Literal['single', 'binary'],
            info_orbit: dict,
            include_GC_SMBH: bool = False,
            base_seed: int = 42,
    ):
        self.data = data
        self.obj_type = obj_type
        self.info_orbit = info_orbit
        self.base_columns = sum(self.info_orbit.values(), [])
        self.integration_indices = self._get_integration_indices()
        self._set_galactic_potential(include_GC_SMBH)
        self._set_orbit_data()
        self.base_seed = base_seed

    def _set_galactic_potential(self, include_GC_SMBH: bool):
        """Initialize the Galactic potential."""
        if include_GC_SMBH:
            self.gp = MWPotential2014 + KeplerPotential(amp=4 * 10**6.0 / conversion.mass_in_msol(220.0,8.0))
        else:
            self.gp = MWPotential2014

    def _set_orbit_data(self):
        """Initialize orbit output arrays."""
        if self.obj_type == 'single':
            self.cols_orbit = self.base_columns
            cols_pos_vel = ['R', 'z', 'phi', 'vR', 'vT', 'vz']
            cols_v_pec_i = ['v_pec_i']
        elif self.obj_type == 'binary':
            star1_cols = [f'star1_{col}' for col in self.base_columns]
            star2_cols = [f'star2_{col}' for col in self.base_columns]
            self.cols_orbit = self.base_columns + star1_cols + star2_cols
            cols_pos_vel = ['R', 'z', 'phi', 'vR', 'vT', 'vz']
            cols_pos_vel = cols_pos_vel + [f'star1_{col}' for col in cols_pos_vel] + [f'star2_{col}' for col in cols_pos_vel]
            cols_v_pec_i = ['v_pec_i', 'star1_v_pec_i', 'star2_v_pec_i']
        else:
            raise ValueError(f"Unsupported obj_type: '{self.obj_type}'. Expected one of: 'single', 'binary'.")

        self.orbit_data = np.full(
            len(self.data),
            np.nan,
            dtype=[(col, 'f8') for col in self.cols_orbit]
        )
        self.orbit_pos_vel = np.full(
            len(self.data),
            np.nan,
            dtype=[(col, 'f8') for col in cols_pos_vel],
        )
        self.orbit_pos_vel_temp = self.orbit_pos_vel.copy()
        self.orbit_v_pec_i = np.full(
            len(self.data),
            np.nan,
            dtype=[(col, 'f8') for col in cols_v_pec_i]
        )

    def _get_integration_indices(self):
        """Return indices where orbit integration should be evaluated."""
        evol_time = self.data['time']

        # Include every positive 1000 Myr boundary.
        indices = np.where((evol_time % 1000 == 0) & (evol_time > 0))[0].tolist()

        # Always include the first row.
        indices.insert(0, 0)

        # Always include the final row.
        last_idx = len(evol_time) - 1
        if last_idx not in indices:
            indices.append(last_idx)

        return indices

    # @timer
    def integrate(self):
        """Integrate the orbit once per 1 Gyr segment."""
        for i in range(len(self.integration_indices) - 1):
            start = self.integration_indices[i]
            end = self.integration_indices[i + 1]
            # Skip invalid sources.
            if not self.data['origin'][start]:
                continue
            # Use one birth position for each 1 Gyr segment.
            ini_pos_cols = ['ini_x', 'ini_y', 'ini_z', 'ini_rho', 'ini_phi', 'ini_dist']
            for col in ini_pos_cols:
                self.data[col][start:end] = self.data[col][start]
            # Initial Orbit instance.
            ini_R = self.data['ini_rho'][start] / 8
            ini_z = self.data['ini_z'][start] / 8
            ini_phi = self.data['ini_phi'][start]
            ini_vR = np.random.normal(0, 10) / 220
            ini_vz = np.random.normal(0, 10) / 220
            ini_vT = np.random.normal(0, 10) / 220 + self.gp.vcirc(R=ini_R)

            ini_o = Orbit([ini_R, ini_vR, ini_vT, ini_z, ini_vz, ini_phi], ro=8., vo=220.)

            # Orbit integration.
            if self.obj_type == 'binary':
                self.integrate_binary(ini_o)
                if 'v_pec_i' in self.base_columns:
                    for col in ['v_pec_i', 'star1_v_pec_i', 'star2_v_pec_i']:
                        self.orbit_data[col][start:end] = self.orbit_v_pec_i[col][start:end]
            else:
                self.integrate_segments(ini_o, integration_target='single')
                if 'v_pec_i' in self.base_columns:
                    self.orbit_data['v_pec_i'][start:end] = self.orbit_v_pec_i['v_pec_i'][start:end]

            # Store base position/velocity coordinates.
            for col in list(self.orbit_pos_vel.dtype.names):
                self.orbit_pos_vel[col][start:end] = self.orbit_pos_vel_temp[col][start:end]

        # Compute derived orbit attributes after integration.
        mask = ~np.isnan(self.orbit_pos_vel['R'])
        if np.any(mask):
            o = Orbit([
                self.orbit_pos_vel['R'][mask],
                self.orbit_pos_vel['vR'][mask],
                self.orbit_pos_vel['vT'][mask],
                self.orbit_pos_vel['z'][mask],
                self.orbit_pos_vel['vz'][mask],
                self.orbit_pos_vel['phi'][mask],
            ], ro=8., vo=220.)
            analyzer = OrbitAnalyzer(o, self.info_orbit, self.gp)
            for col in self.base_columns:
                if col == 'v_pec_i':
                    continue
                self.orbit_data[col][mask] = getattr(analyzer, col)

        if self.obj_type == 'binary':
            mask1 = ~np.isnan(self.orbit_pos_vel['star1_R'])
            mask2 = ~np.isnan(self.orbit_pos_vel['star2_R'])
            for prefix, mask in zip(['star1_', 'star2_'], [mask1, mask2]):
                if np.any(mask):
                    o = Orbit([
                        self.orbit_pos_vel[f'{prefix}R'][mask],
                        self.orbit_pos_vel[f'{prefix}vR'][mask],
                        self.orbit_pos_vel[f'{prefix}vT'][mask],
                        self.orbit_pos_vel[f'{prefix}z'][mask],
                        self.orbit_pos_vel[f'{prefix}vz'][mask],
                        self.orbit_pos_vel[f'{prefix}phi'][mask],
                    ], ro=8., vo=220.)
                    analyzer = OrbitAnalyzer(o, self.info_orbit, self.gp)
                    for col in self.base_columns:
                        if col == 'v_pec_i':
                            continue
                        self.orbit_data[f'{prefix}{col}'][mask] = getattr(analyzer, col)

    # @timer
    def integrate_binary(
            self,
            ini_o: Orbit
    ) -> None:
        """Integrate a binary-system orbit."""

        # Binary remains bound throughout evolution.
        if np.all(self.data['bound']):
            self.integrate_segments(ini_o=ini_o, integration_target='binary')
        # Binary is already unbound or merged at the first timestep.
        elif np.any((self.data['time'] == 0) & (~self.data['bound'])):
            # Handle the disrupted state directly.
            mask = ~self.data['bound']
            # Star 2 is gone; integrate only star 1.
            if np.all(self.data['type2'][mask] == 'massless'):
                self.integrate_segments(ini_o=ini_o, integration_target='star1')
            # Star 1 is gone; integrate only star 2.
            elif np.all(self.data['type1'][mask] == 'massless'):
                self.integrate_segments(ini_o=ini_o, integration_target='star2')
            else:
                raise ValueError(
                    "Invalid disrupted-binary state. If the binary is unbound at the first timestep, "
                    "exactly one component should be massless."
                )
        # Binary starts bound and later becomes disrupted or merged.
        else:
            # First integrate the bound phase.
            o = self.integrate_segments(ini_o=ini_o, integration_target='binary')

            # Then integrate the disrupted components.
            mask = ~self.data['bound']
            random_seed = self.base_seed * 1_000_000 + np.where(mask)[0][0]
            if self.data['type1'][mask][0] != 'massless':
                offset = np.array([
                    self.data['v1_offset_x'][mask][0],
                    self.data['v1_offset_y'][mask][0],
                    self.data['v1_offset_z'][mask][0]
                ])
                v_off = rotate_velocity_offset_to_galactocentric(v_offset=offset, random_seed=random_seed) / 220
                o1 = Orbit(o.vxvv.flatten() + np.array([0, v_off[0], v_off[1], 0, v_off[2], 0]), ro=8., vo=220.)
                self.integrate_segments(ini_o=o1, integration_target='star1')
            if self.data['type2'][mask][0] != 'massless':
                offset = np.array([
                    self.data['v2_offset_x'][mask][0],
                    self.data['v2_offset_y'][mask][0],
                    self.data['v2_offset_z'][mask][0]
                ])
                v_off = rotate_velocity_offset_to_galactocentric(v_offset=offset, random_seed=random_seed) / 220
                o2 = Orbit(o.vxvv.flatten() + np.array([0, v_off[0], v_off[1], 0, v_off[2], 0]), ro=8., vo=220.)
                self.integrate_segments(ini_o=o2, integration_target='star2')

    # @timer
    def integrate_segments(
            self,
            ini_o: Orbit,
            integration_target: str
    ) -> Orbit:
        """Integrate one or more orbit segments."""
        # Initial orbit.
        o = ini_o

        # Select velocity-offset columns for the requested integration target.
        if integration_target == 'binary':
            check_columns = ['v_offset_x', 'v_offset_y', 'v_offset_z']  # Bound-binary offset velocity.
            mask = self.data['bound']
            orbit_cols_prefix = ''
        elif integration_target == 'star1':
            check_columns = ['v1_offset_x', 'v1_offset_y', 'v1_offset_z']  # Star-1 offset velocity.
            mask = ~self.data['bound']
            orbit_cols_prefix = 'star1_'
        elif integration_target == 'star2':
            check_columns = ['v2_offset_x', 'v2_offset_y', 'v2_offset_z']  # Star-2 offset velocity.
            mask = ~self.data['bound']
            orbit_cols_prefix = 'star2_'
        elif integration_target == 'single':
            check_columns = ['v_kick_x', 'v_kick_y', 'v_kick_z']  # Single-star kick velocity.
            mask = np.ones(len(self.data), dtype=bool)
            orbit_cols_prefix = ''
        else:
            raise ValueError(
                "Unsupported integration_target. Expected one of: 'binary', 'star1', 'star2', 'single'."
            )

        # Locate rows with a velocity offset.
        filtered_data = self.data[mask]
        has_offset = ~np.isnan(filtered_data[check_columns[0]])
        non_zero_indices = np.where(has_offset)[0]
        non_zero_indices = non_zero_indices[non_zero_indices != 0]

        # Build integration time segments from velocity-offset rows.
        if len(non_zero_indices) == 0:
            # Single time segment.
            time_segments = [(0, len(filtered_data) - 1)]
        elif len(non_zero_indices) <= 2:
            # Initial segment.
            time_segments = [(0, non_zero_indices[0])]
            # Middle segments.
            for i in range(len(non_zero_indices) - 1):
                time_segments.append((non_zero_indices[i], non_zero_indices[i + 1]))
            # Final segment.
            time_segments.append((non_zero_indices[-1], len(filtered_data) - 1))
        else:
            raise ValueError("Orbit integration supports at most two velocity-offset events per segment.")
        # print('time_segments:', time_segments)
        # print(integration_target, non_zero_indices, time_segments)
        # print(f"integrate_segment_{integration_target} time segments: {time_segments}")

        # Segment integration.
        filtered_indices = np.where(mask)[0]
        for integrate_start, integrate_end in time_segments:
            orig_idx = filtered_indices[integrate_start:integrate_end + 1]
            ts = self.data['time'][orig_idx] * u.Myr
            # start = time.time()
            o.integrate(ts, pot=self.gp, method='dop853_c')
            # end = time.time()
            # print(o.vxvv)
            # print(ts)
            # print(f"integrate_segment_{integration_target} elapsed: {(end - start) * 1000:.2f} ms")
            # Temporarily store intermediate states.
            self.orbit_pos_vel_temp[f'{orbit_cols_prefix}R'][orig_idx] = o.R(ts) / 8
            self.orbit_pos_vel_temp[f'{orbit_cols_prefix}z'][orig_idx] = o.z(ts) / 8
            self.orbit_pos_vel_temp[f'{orbit_cols_prefix}phi'][orig_idx] = o.phi(ts)
            self.orbit_pos_vel_temp[f'{orbit_cols_prefix}vR'][orig_idx] = o.vR(ts) / 220
            self.orbit_pos_vel_temp[f'{orbit_cols_prefix}vz'][orig_idx] = o.vz(ts) / 220
            self.orbit_pos_vel_temp[f'{orbit_cols_prefix}vT'][orig_idx] = o.vT(ts) / 220

            # Current orbit position
            last_time = ts[-1]
            last_R = o.R(t=last_time) / 8
            last_z = o.z(t=last_time) / 8
            last_phi = o.phi(t=last_time)

            # Rotate the pre-SN center-of-mass-frame offset into the local (vR, vT, vz) basis.
            random_seed = self.base_seed * 1_000_000 + filtered_indices[integrate_end]
            offset = np.array([filtered_data[col][integrate_end] for col in check_columns])
            v_off = rotate_velocity_offset_to_galactocentric(v_offset=offset, random_seed=random_seed) / 220

            # Current orbit velocity.
            last_vR = o.vR(t=last_time) / 220 + v_off[0]
            last_vT = o.vT(t=last_time) / 220 + v_off[1]
            last_vz = o.vz(t=last_time) / 220 + v_off[2]

            # Update Orbit state.
            o = Orbit([last_R, last_vR, last_vT, last_z, last_vz, last_phi], ro=8., vo=220.)


        # Get the initial characteristic peculiar velocity for each system type.
        if 'v_pec_i' in self.base_columns:
            R = self.orbit_pos_vel_temp[f'{orbit_cols_prefix}R'][filtered_indices]
            vR = self.orbit_pos_vel_temp[f'{orbit_cols_prefix}vR'][filtered_indices]
            vz = self.orbit_pos_vel_temp[f'{orbit_cols_prefix}vz'][filtered_indices]
            vT = self.orbit_pos_vel_temp[f'{orbit_cols_prefix}vT'][filtered_indices]

            v_circ = self.gp.vcirc(R=R)

            v_pec = np.sqrt(vR ** 2 + vz ** 2 + (vT - v_circ) ** 2) * 220

            if integration_target == 'binary':
                stellar_type = np.array(
                    [f"{t1}_{t2}" for t1, t2 in zip(filtered_data['type1'], filtered_data['type2'])])
            elif integration_target == 'star1':
                stellar_type = filtered_data['type1']
            elif integration_target == 'star2':
                stellar_type = filtered_data['type2']
            else:
                stellar_type = filtered_data['type']

            unique_types, first_occurrence_idx = np.unique(stellar_type, return_index=True)
            first_v_pec_values = v_pec[first_occurrence_idx]
            type_to_idx = np.searchsorted(unique_types, stellar_type)
            v_pec_i = first_v_pec_values[type_to_idx]
            self.orbit_v_pec_i[f'{orbit_cols_prefix}v_pec_i'][filtered_indices] = v_pec_i


        # Return the final phase-space state.
        return o


class OrbitAnalyzer:
    """
    Orbit analyzer for extracting quantities in several coordinate systems.

    Parameters
    ----------
    orbit : galpy.orbit.Orbit
        galpy Orbit object.
    info_orbit : dict
        Position/velocity attributes to extract.
    gp : Galpy object
        Galactic potential.
    """

    def __init__(
            self,
            orbit: Orbit,
            info_orbit: dict,
            gp,
    ):
        self.x = orbit.x()
        self.y = orbit.y()
        self.z = orbit.z()
        self.rho = orbit.R()
        self.phi = orbit.phi()
        self.vx = orbit.vx()
        self.vy = orbit.vy()
        self.vz = orbit.vz()
        self.vR = orbit.vR()
        self.vT = orbit.vT()

        self.dist = orbit.dist()
        self.v_radial = orbit.vlos()

        if 'velocity' in info_orbit:
            self.v = np.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)
            self.v_esc = gp.vesc(R=self.rho / 8) * 220
            self.v_circ = gp.vcirc(R=self.rho / 8) * 220
            self.v_pec = np.sqrt(self.vR ** 2 + self.vz ** 2 + (self.vT - self.v_circ) ** 2)

        if 'Galactic' in info_orbit:
            # Spherical quantities.
            self.l = orbit.ll()
            self.b = orbit.bb()
            self.pm_l = orbit.pmll()
            self.pm_b = orbit.pmbb()

            # Cartesian quantities.
            self.helioX = orbit.helioX()
            self.helioY = orbit.helioY()
            self.helioZ = orbit.helioZ()
            self.U = orbit.U()
            self.V = orbit.V()
            self.W = orbit.W()

        if 'ICRS' in info_orbit:
            self.ra = orbit.ra()
            self.dec = orbit.dec()
            self.pm_ra = orbit.pmra()
            self.pm_dec = orbit.pmdec()
