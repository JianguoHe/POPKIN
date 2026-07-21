from galpy.orbit import Orbit
from galpy.potential import MWPotential2014 as mwp14
from galpy.potential import calcRotcurve, calcEscapecurve
import astropy.units as u
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd

info_orbit: list = [
    'x', 'y', 'z', 'R', 'phi', 'dist', 'helioX', 'helioY', 'helioZ',
    'vx', 'vy', 'vz', 'vR', 'vT', 'vlos', 'U', 'V', 'W',
    'ra', 'dec', 'll', 'bb',
    'pmra', 'pmdec', 'pmll', 'pmbb'
]
# print(calcRotcurve(mwp14, Rs=1))
Rs = np.linspace(0.001, 1, 1000)
# start = time.time()
# # v_esc = calcEscapecurve(mwp14, Rs=Rs)
# v_esc = mwp14.vesc(R=Rs)
v_cir = mwp14.vcirc(R=Rs)
# # v_cir = calcRotcurve(mwp14, Rs=Rs)
# end = time.time()
#
# print(f"Elapsed time: {(end - start) * 1000:.2f} ms")
print(v_cir[:5])


o = Orbit(vxvv=[0.08666025, 0.13144172, 0.6886768,  0.06907211, 0.13380396, 1.58268559])

ts = np.array([12.37727595, 12.37727595, 12.47727595, 13.47727595,
                23.47727595, 123.47727595, 623.47727595, 1000.,
                1500., 2000., 2500., 3000., 3500., 4000., 4500.,
                5000., 5500., 6000., 6500., 7000., 7500., 8000.,
                8500., 9000., 9500., 10000., 10500., 11000.,
                11500., 12000.]) * u.Myr

# ts = np.sort(np.random.choice(12001, size=200, replace=False)) * u.Myr
ts_1 = np.linspace(0, 12000, 1000) * u.Myr
start = time.time()
for _ in range(1):
    o.integrate(ts, pot=mwp14, method='dop853_c')
    # for info in info_orbit:
    #     result = getattr(o, info)(ts)
    # print(o.z(ts))
    # print(o.vx(ts))
    # print(o.vy(ts))
    # print(o.vz(ts))
    # ra, dec, pm_ra, pm_dec = o.ra, o.dec, o.pmra, o.pmdec

end = time.time()
print(f"Runtime: {end - start:.4f} s")
plt.plot(o.x(ts), o.y(ts))
plt.show()
# print(o.pmll(ts))
# start = time.time()
# for _ in range(100):
#     # R = o.R(ts)
#     R = o.vxvv
#
# end = time.time()
# print(R)
# print(f"Runtime R: {end - start:.4f} s")
print('Final state:', o.vxvv)
R = o.R(ts)
z = o.z(ts)
phi = o.phi(ts)
vR = o.vR(ts)
vz = o.vz(ts)
vT = o.vT(ts)
df = pd.DataFrame({
    'time_Gyr': ts.value,
    'R_kpc': R,
    'vR_kms': vR,
    'vT_kms': vT,
    'z_kpc': z,
    'vz_kms': vz,
    'phi_rad': phi,
})
print(df.head(10))
# ini_R = 8 / 8
# ini_z = 0.0208 / 8
# ini_phi = 0.0
# ini_vR = 220 / 220
# ini_vz = 10 / 220
# ini_vT = calcRotcurve(mwp14, Rs=ini_R)[0] + 10 / 220

# Create the initial Orbit object.

o = Orbit([R/8, vR/220, vT/220, z/8, vz/220, phi], ro=8., vo=220.)

# for info in info_orbit:
#     result = getattr(o, info)()

# start = time.time()
# for _ in range(100):
#     # result = o.R
#     result = o.vxvv[:, 0] * 8
#
# end = time.time()
# print(result)
# print(f"Runtime R: {end - start:.4f} s")

df = pd.DataFrame({
    'time_Gyr': ts.value,
    'R_kpc': o.vxvv[:, 0] * 8,
    'vR_kms': o.vxvv[:, 1] * 220,
    'vT_kms': o.vxvv[:, 2] * 220,
    'z_kpc': o.vxvv[:, 3] * 8,
    'vz_kms': o.vxvv[:, 4] * 220,
    'phi_rad': o.vxvv[:, 5],
})
print(df.head(10))
print(o.R())
# print(o.vxvv[-1])






