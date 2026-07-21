from pathlib import Path

from popkin.stars.single_star import SingleStar
from popkin.stars.binary_star import BinaryStar
from popkin.kinematics.orbit import OrbitIntegrator
from popkin.config.controls_default import info_orbit
from popkin.galaxies import MilkyWay
from popkin.utils import process_binary_star_data, merge_structured_data
import numpy as np
import pandas as pd
import time

SCRIPT_DIR = Path(__file__).resolve().parent



star = SingleStar(type=1, mass=25, Z=0.02, index=1)
star.evolve()
star_data = star.data

from popkin.utils import create_popbin_parameter_space
# index_list = [542, 550, 582, 606, 610, 670, 676, 707, 1518, 1564, 1577, 1713, 1722, 1750, 1772, 1774, 2571, 2577]
# index_list =  [
#     435, 1524, 2413, 3436, 3466, 4251, 4321, 4459, 4476, 5153, 5151, 5150,
#     6050, 6049, 7853, 9655, 21325
# ][:]
index_list = [2331, 6299, 10384, 10517, 11258, 11529, 12543, 13225,
              13435, 14252, 16962, 17099, 18622, 20405, 23124, 23109, 24002, 25075][:1]

for index in index_list:
    print('*' * 65)
    print(index)
    print('*' * 65)

    m1, m2, period, ecc, weight = create_popbin_parameter_space()[index]

    star1 = SingleStar(type=1, mass=m1, Z=0.02, index=index)
    star2 = SingleStar(type=1, mass=m2, Z=0.02, index=index)
    binary = BinaryStar(star1=star1, star2=star2, period=period, ecc=0, index=index)
    binary.evolve()
    binary_data = process_binary_star_data(binary, star1, star2)

    galaxy = MilkyWay(metallicity_model='enrichment', Z=0.02)
    stars = galaxy.generate_star(tau=binary_data['time'] / 1000)
    binary_birth = stars[['origin', 'ini_x', 'ini_y', 'ini_z', 'ini_rho', 'ini_phi', 'ini_dist']]
    binary_data = merge_structured_data([binary_data, binary_birth])

    # df = pd.DataFrame(binary_data)
    # df.to_csv(SCRIPT_DIR / 'test_orbit.csv', index=False, float_format='%.20g')

    # o = Orbit(data=star_data, info_orbit=info_orbit, obj_type='single')
    start = time.time()
    o = OrbitIntegrator(data=binary_data, info_orbit=info_orbit, obj_type='binary')
    for _ in range(1):
        # print(o.info_orbit)
        # print(o.orbit_columns)
        # # print(o.orbit_data.dtype)
        # print(o.data['time'][o.integration_indices])
        o.integrate()
    # time.sleep(3)
end = time.time()
print(f"Elapsed time: {(end - start)*1000:.2f} ms")

df = pd.DataFrame(merge_structured_data([o.data, o.orbit_data]))
df.to_csv(SCRIPT_DIR / 'test_orbit.csv', index=False, float_format='%.4g')
