import numpy as np
from pathlib import Path

from popkin.config.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------------------------------------------------------
#                      Binding-energy parameter fitting data (WJL2016: doi: 10.1088/1674-4527/16/8/126)
# ------------------------------------------------------------------------------------------------------------------

# Data directory for WJL2016 binding-energy lambda tables.
lambda_dir = Path(__file__).resolve().parent / 'wjl2016_data'

# Table grids used to calculate common-envelope binding-energy lambda.
metallicities = ['z=0.02', 'z=0.001', 'z=0.0001']
masses = ['M1', 'M2', 'M4', 'M6', 'M8', 'M10', 'M20', 'M30', 'M40', 'M60']

lambda_data = {}

for z in metallicities:
    lambda_data[z] = {}

    for mass in masses:
        file_path = lambda_dir / z / f'{mass}.dat'

        if file_path.exists():
            lambda_data[z][mass] = np.loadtxt(file_path)
        else:
            logger.warning("WJL2016 table file not found: %s", file_path)
            lambda_data[z][mass] = None

z002 = np.array([lambda_data['z=0.02'][mass] for mass in masses])
z0001 = np.array([lambda_data['z=0.001'][mass] for mass in masses])
z00001 = np.array([lambda_data['z=0.0001'][mass] for mass in masses])
