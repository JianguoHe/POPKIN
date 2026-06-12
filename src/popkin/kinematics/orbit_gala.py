"""Experimental gala-based orbit integration implementation.

This file is kept as a commented reference implementation. It can be used by
users who explicitly install gala and want to experiment with a gala backend.
The production orbit integrator remains ``orbit.py`` because gala was less
stable for the non-uniform timesteps used by the current POPKIN workflow.

To experiment with this backend, uncomment the code below and switch the
driver/import path manually.
"""


# import numpy as np
# import astropy.units as u
# from typing import Literal
#
# import astropy.coordinates as coord
# import gala.dynamics as gd
# import gala.potential as gp
#
# from popkin.utils import rotate_velocity_offset_to_galactocentric
#
#
# mw = gp.MilkyWayPotential(version="latest")
#
#
# class OrbitIntegrator:
#     """Orbit integrator based on gala."""
#
#     def __init__(
#             self,
#             data: np.ndarray,
#             obj_type: Literal['single', 'binary'],
#             info_orbit: dict,
#     ):
#         self.data = data
#         self.obj_type = obj_type
#         self.info_orbit = info_orbit
#         self.base_columns = sum(self.info_orbit.values(), [])
#         self.integration_indices = self._get_integration_indices()
#         self._set_orbit_data()
#
#     def _set_orbit_data(self):
#         """Initialize orbit output arrays."""
#         if self.obj_type == 'single':
#             self.cols_orbit = self.base_columns
#             cols_pos_vel = ['pos', 'vel']
#             cols_v_pec_i = ['v_pec_i']
#         elif self.obj_type == 'binary':
#             star1_cols = [f'star1_{col}' for col in self.base_columns]
#             star2_cols = [f'star2_{col}' for col in self.base_columns]
#             self.cols_orbit = self.base_columns + star1_cols + star2_cols
#             cols_pos_vel = ['pos', 'vel', 'star1_pos', 'star1_vel', 'star2_pos', 'star2_vel']
#             cols_v_pec_i = ['v_pec_i', 'star1_v_pec_i', 'star2_v_pec_i']
#         else:
#             raise ValueError(f"Unknown obj_type: '{self.obj_type}'. Must be either 'single' or 'binary'.")
#
#         self.orbit_data = np.full(
#             len(self.data),
#             np.nan,
#             dtype=[(col, 'f8') for col in self.cols_orbit]
#         )
#         self.orbit_pos_vel = np.full(
#             len(self.data),
#             np.nan,
#             dtype=[(col, 'f8', (3,)) for col in cols_pos_vel],
#         )
#         self.orbit_pos_vel_temp = self.orbit_pos_vel.copy()
#         self.orbit_v_pec_i = np.full(
#             len(self.data),
#             np.nan,
#             dtype=[(col, 'f8') for col in cols_v_pec_i]
#         )
#
#     def _get_integration_indices(self):
#         """Return indices where orbit integration should be evaluated."""
#         time = self.data['time']
#
#         # Include every positive 1000 Myr boundary.
#         indices = np.where((time % 1000 == 0) & (time > 0))[0].tolist()
#
#         # Always include the first row.
#         indices.insert(0, 0)
#
#         # Always include the final row.
#         last_idx = len(time) - 1
#         if last_idx not in indices:
#             indices.append(last_idx)
#
#         return indices
#
#     def integrate(self):
#         """Integrate the orbit once per 1 Gyr segment."""
#         for i in range(len(self.integration_indices) - 1):
#             start = self.integration_indices[i]
#             end = self.integration_indices[i + 1]
#
#             # Skip invalid sources.
#             if not self.data['origin'][start]:
#                 continue
#
#             # Use one birth position for each 1 Gyr segment.
#             ini_pos_cols = ['ini_x', 'ini_y', 'ini_z', 'ini_rho', 'ini_phi', 'ini_dist']
#             for col in ini_pos_cols:
#                 self.data[col][start:end] = self.data[col][start]
#
#             # Initial phase-space position.
#             ini_x = self.data['ini_x'][start]
#             ini_y = self.data['ini_y'][start]
#             ini_z = self.data['ini_z'][start]
#             ini_phi = self.data['ini_phi'][start]
#             ini_pos = [ini_x, ini_y, ini_z] * u.kpc
#
#             ini_vz = np.random.normal(0, 10)
#             ini_vR = np.random.normal(0, 10)
#             ini_vT = np.random.normal(0, 10) + mw.circular_velocity(ini_pos)[0].value
#             ini_vx = ini_vR * np.cos(ini_phi) + ini_vT * np.sin(ini_phi)
#             ini_vy = ini_vR * np.sin(ini_phi) - ini_vT * np.cos(ini_phi)
#             ini_velocity = [ini_vx, ini_vy, ini_vz] * u.km / u.s
#
#             w0 = gd.PhaseSpacePosition(
#                 pos=ini_pos,
#                 vel=ini_velocity,
#             )
#
#             # Orbit integration.
#             if self.obj_type == 'binary':
#                 self.integrate_binary(w0)
#                 if 'v_pec_i' in self.base_columns:
#                     for col in ['v_pec_i', 'star1_v_pec_i', 'star2_v_pec_i']:
#                         self.orbit_data[col][start:end] = self.orbit_v_pec_i[col][start:end]
#             else:
#                 self.integrate_segments(w0, integration_target='single')
#                 if 'v_pec_i' in self.base_columns:
#                     self.orbit_data['v_pec_i'][start:end] = self.orbit_v_pec_i['v_pec_i'][start:end]
#
#             # Store base position/velocity coordinates.
#             for col in list(self.orbit_pos_vel.dtype.names):
#                 self.orbit_pos_vel[col][start:end] = self.orbit_pos_vel_temp[col][start:end]
#
#         # Compute derived orbit attributes after integration.
#         mask = ~np.isnan(self.orbit_pos_vel['pos']).any(axis=1)
#         if np.any(mask):
#             orbit = gd.Orbit(
#                 pos=self.orbit_pos_vel['pos'][mask].T * u.kpc,
#                 vel=self.orbit_pos_vel['vel'][mask].T * u.km / u.s,
#             )
#             analyzer = OrbitAnalyzer(orbit, self.info_orbit)
#             for col in self.base_columns:
#                 if col == 'v_pec_i':
#                     continue
#                 self.orbit_data[col][mask] = getattr(analyzer, col)
#
#         if self.obj_type == 'binary':
#             mask1 = ~np.isnan(self.orbit_pos_vel['star1_pos']).any(axis=1)
#             mask2 = ~np.isnan(self.orbit_pos_vel['star2_pos']).any(axis=1)
#             for prefix, mask in zip(['star1_', 'star2_'], [mask1, mask2]):
#                 if np.any(mask):
#                     orbit = gd.Orbit(
#                         pos=self.orbit_pos_vel[f'{prefix}pos'][mask].T * u.kpc,
#                         vel=self.orbit_pos_vel[f'{prefix}vel'][mask].T * u.km / u.s,
#                     )
#                     analyzer = OrbitAnalyzer(orbit, self.info_orbit)
#                     for col in self.base_columns:
#                         if col == 'v_pec_i':
#                             continue
#                         self.orbit_data[f'{prefix}{col}'][mask] = getattr(analyzer, col)
#
#     def integrate_binary(self, w0):
#         """Integrate a binary-system orbit."""
#
#         # Binary remains bound throughout evolution.
#         if np.all(self.data['bound']):
#             self.integrate_segments(w0=w0, integration_target='binary')
#
#         # Binary is already unbound or merged at the first timestep.
#         elif np.any((self.data['time'] == 0) & (~self.data['bound'])):
#             # Handle the disrupted state directly.
#             mask = ~self.data['bound']
#
#             # Star 2 is gone; integrate only star 1.
#             if np.all(self.data['type2'][mask] == 'massless'):
#                 self.integrate_segments(w0=w0, integration_target='star1')
#
#             # Star 1 is gone; integrate only star 2.
#             elif np.all(self.data['type1'][mask] == 'massless'):
#                 self.integrate_segments(w0=w0, integration_target='star2')
#             else:
#                 raise ValueError('Please check the evolution of binary.')
#
#         # Binary starts bound and later becomes disrupted or merged.
#         else:
#             # First integrate the bound phase.
#             w = self.integrate_segments(w0=w0, integration_target='binary')
#
#             # Then integrate the disrupted components.
#             mask = ~self.data['bound']
#             if self.data['type1'][mask][0] != 'massless':
#                 offset = np.array([
#                     self.data['v1_offset_x'][mask][0],
#                     self.data['v1_offset_y'][mask][0],
#                     self.data['v1_offset_z'][mask][0],
#                 ])
#                 offset_gc = rotate_velocity_offset_to_galactocentric(v_offset=offset)
#                 w1 = gd.PhaseSpacePosition(
#                     pos=w.xyz,
#                     vel=w.v_xyz + offset_gc * u.km / u.s,
#                 )
#                 self.integrate_segments(w0=w1, integration_target='star1')
#
#             if self.data['type2'][mask][0] != 'massless':
#                 offset = np.array([
#                     self.data['v2_offset_x'][mask][0],
#                     self.data['v2_offset_y'][mask][0],
#                     self.data['v2_offset_z'][mask][0],
#                 ])
#                 offset_gc = rotate_velocity_offset_to_galactocentric(v_offset=offset)
#                 w2 = gd.PhaseSpacePosition(
#                     pos=w.xyz,
#                     vel=w.v_xyz + offset_gc * u.km / u.s,
#                 )
#                 self.integrate_segments(w0=w2, integration_target='star2')
#
#     def integrate_segments(self, w0, integration_target):
#         """Integrate one or more orbit segments."""
#         w = w0
#
#         # Select velocity-offset columns for the requested integration target.
#         if integration_target == 'binary':
#             check_columns = ['v_offset_x', 'v_offset_y', 'v_offset_z']            # Bound-binary offset velocity.
#             mask = self.data['bound']
#             orbit_cols_prefix = ''
#         elif integration_target == 'star1':
#             check_columns = ['v1_offset_x', 'v1_offset_y', 'v1_offset_z']         # Star-1 offset velocity.
#             mask = ~self.data['bound']
#             orbit_cols_prefix = 'star1_'
#         elif integration_target == 'star2':
#             check_columns = ['v2_offset_x', 'v2_offset_y', 'v2_offset_z']         # Star-2 offset velocity.
#             mask = ~self.data['bound']
#             orbit_cols_prefix = 'star2_'
#         elif integration_target == 'single':
#             check_columns = ['v_kick_x', 'v_kick_y', 'v_kick_z']                  # Single-star kick velocity.
#             mask = np.ones(len(self.data), dtype=bool)
#             orbit_cols_prefix = ''
#         else:
#             raise ValueError('Please set the proper component of binary to integrate.')
#
#         # Locate rows with a velocity offset.
#         filtered_data = self.data[mask]
#         has_offset = ~np.isnan(filtered_data[check_columns[0]])
#         non_zero_indices = np.where(has_offset)[0]
#         non_zero_indices = non_zero_indices[non_zero_indices != 0]
#
#         # Build integration time segments from velocity-offset rows.
#         if len(non_zero_indices) == 0:
#             # Single time segment.
#             time_segments = [(0, len(filtered_data) - 1)]
#         elif len(non_zero_indices) <= 2:
#             # Initial segment.
#             time_segments = [(0, non_zero_indices[0])]
#             # Middle segments.
#             for i in range(len(non_zero_indices) - 1):
#                 time_segments.append((non_zero_indices[i], non_zero_indices[i + 1]))
#             # Final segment.
#             time_segments.append((non_zero_indices[-1], len(filtered_data) - 1))
#         else:
#             raise ValueError('The number of velocity offsets is above 2 unphysically.')
#
#         # Segment integration.
#         filtered_indices = np.where(mask)[0]
#         for integrate_start, integrate_end in time_segments:
#             orig_idx = filtered_indices[integrate_start:integrate_end + 1]
#             ts = self.data['time'][orig_idx] * u.Myr
#             orbit = mw.integrate_orbit(w, t=ts, Integrator='dop853', cython_if_possible=True)
#
#             # Retry with tighter tolerances if the integration appears to diverge.
#             final_pos = orbit[-1].xyz.value
#             final_vel = orbit[-1].v_xyz.to_value(u.km / u.s)
#             if (np.all(np.abs(final_pos) < 0.1) or
#                     np.std(final_pos) < 1e-3 or
#                     np.all(np.abs(final_vel) < 1) or
#                     np.std(final_vel) < 1 or
#                     np.isnan(final_pos).any() or
#                     np.isnan(final_vel).any()):
#                 orbit = mw.integrate_orbit(
#                     w,
#                     t=ts,
#                     Integrator='dop853',
#                     cython_if_possible=True,
#                     Integrator_kwargs={'atol': 1e-18, 'rtol': 1e-18},
#                 )
#
#             self.orbit_pos_vel_temp[f'{orbit_cols_prefix}pos'][orig_idx] = orbit.xyz.value.T
#             self.orbit_pos_vel_temp[f'{orbit_cols_prefix}vel'][orig_idx] = orbit.v_xyz.to_value(u.km / u.s).T
#
#             # Translate the pre-SN center-of-mass velocity offset to Galactocentric Cartesian coordinates.
#             offset = np.array([filtered_data[col][integrate_end] for col in check_columns])
#             offset_gc = rotate_velocity_offset_to_galactocentric(v_offset=offset)
#
#             # Update phase-space state.
#             w = gd.PhaseSpacePosition(
#                 pos=orbit[-1].xyz,
#                 vel=orbit[-1].v_xyz.to(u.km / u.s) + offset_gc * u.km / u.s,
#             )
#             print(f'after integration w: {w}')
#
#         # Get the initial characteristic peculiar velocity for each system type.
#         if 'v_pec_i' in self.base_columns:
#             orbit = gd.Orbit(
#                 pos=self.orbit_pos_vel_temp[f'{orbit_cols_prefix}pos'][filtered_indices].T * u.kpc,
#                 vel=self.orbit_pos_vel_temp[f'{orbit_cols_prefix}vel'][filtered_indices].T * u.km / u.s,
#             )
#             cyl = orbit.cylindrical
#             vz = cyl.v_z.to_value(u.km / u.s)
#             vR = cyl.v_rho.to_value(u.km / u.s)
#             vT = -(cyl.rho * cyl.pm_phi).to_value(u.km / u.s, u.dimensionless_angles())
#             v_circ = mw.circular_velocity(orbit.xyz).value
#             v_pec = np.sqrt(vR ** 2 + vz ** 2 + (vT - v_circ) ** 2)
#
#             if integration_target == 'binary':
#                 stellar_type = np.array([f"{t1}_{t2}" for t1, t2 in zip(filtered_data['type1'], filtered_data['type2'])])
#             elif integration_target == 'star1':
#                 stellar_type = filtered_data['type1']
#             elif integration_target == 'star2':
#                 stellar_type = filtered_data['type2']
#             else:
#                 stellar_type = filtered_data['type']
#
#             unique_types, first_occurrence_idx = np.unique(stellar_type, return_index=True)
#             first_v_pec_values = v_pec[first_occurrence_idx]
#             type_to_idx = np.searchsorted(unique_types, stellar_type)
#             v_pec_i = first_v_pec_values[type_to_idx]
#             self.orbit_v_pec_i[f'{orbit_cols_prefix}v_pec_i'][filtered_indices] = v_pec_i
#
#         # Return the final phase-space state.
#         return w
#
#
# class OrbitAnalyzer:
#     """Orbit analyzer for extracting quantities in several coordinate systems.
#
#     Parameters
#     ----------
#     orbit : gala.dynamics.Orbit
#         Integrated orbit object.
#     info_orbit : dict
#         Position/velocity attributes to extract.
#     """
#
#     def __init__(self, orbit, info_orbit):
#         cyl = orbit.cylindrical
#
#         self.x, self.y, self.z = orbit.xyz.value
#         self.vx, self.vy, self.vz = orbit.v_xyz.to_value(u.km / u.s)
#         self.rho = cyl.rho.value
#         self.phi = cyl.phi.value
#         self.vR = cyl.v_rho.to_value(u.km / u.s)
#         self.vT = -(cyl.rho * cyl.pm_phi).to_value(u.km / u.s, u.dimensionless_angles())
#
#         if 'velocity' in info_orbit:
#             self.v = np.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)
#             self.v_circ = mw.circular_velocity(orbit.xyz).value
#             self.v_pec = np.sqrt(self.vR ** 2 + self.vz ** 2 + (self.vT - self.v_circ) ** 2)
#
#         if 'Galactic' in info_orbit:
#             gal_keys = set(info_orbit['Galactic'])
#             gal_coords = orbit.to_coord_frame(coord.Galactic())
#
#             # Spherical quantities.
#             if not gal_keys.isdisjoint({'l', 'b', 'pm_l', 'pm_b', 'dist', 'v_radial'}):
#                 self.l = gal_coords.l.value
#                 self.b = gal_coords.b.value
#                 self.pm_l = gal_coords.pm_l_cosb.value
#                 self.pm_b = gal_coords.pm_b.value
#                 self.dist = gal_coords.distance.value
#                 self.v_radial = gal_coords.radial_velocity.value
#
#             # Cartesian quantities.
#             if not gal_keys.isdisjoint({'helioX', 'helioY', 'helioZ', 'U', 'V', 'W'}):
#                 self.helioX, self.helioY, self.helioZ = gal_coords.cartesian.xyz.value
#                 self.U, self.V, self.W = gal_coords.velocity.d_xyz.value
#
#         if 'ICRS' in info_orbit:
#             icrs_keys = set(info_orbit['ICRS'])
#             icrs_coords = orbit.to_coord_frame(coord.ICRS())
#
#             if not icrs_keys.isdisjoint({'ra', 'dec', 'pm_ra', 'pm_dec'}):
#                 self.ra = icrs_coords.ra.to_value(u.deg)
#                 self.dec = icrs_coords.dec.to_value(u.deg)
#                 self.pm_ra = icrs_coords.pm_ra_cosdec.value
#                 self.pm_dec = icrs_coords.pm_dec.value
