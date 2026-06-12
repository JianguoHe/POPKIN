# POPKIN: Population Synthesis and Stellar Kinematics

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue)](https://doi.org/10.xxxx/xxxxx)

POPKIN is a Python framework for rapid stellar evolution, binary evolution,
population synthesis, and Galactic kinematics. It is built from the
first-generation rapid binary-evolution framework of BSE and reorganized into a
modular Python codebase for transparent model development and large population
calculations.

POPKIN currently supports single-star evolution, binary-star evolution,
single-star population synthesis, and binary population synthesis. Population
synthesis can be coupled to a Milky Way model, metallicity evolution, Galactic
orbit integration, and selected observable calculations such as compact-binary
gravitational-wave signal-to-noise ratios and isolated black-hole accretion from
the interstellar medium.

> Development status: POPKIN is under active development. Interfaces and
> default physical prescriptions may change before a stable public release.

## ✨ Features

### 🔭 Core Capabilities

- **Single-star evolution**: evolve stars from the zero-age main sequence to
  compact remnants with configurable stellar-wind, remnant, and supernova
  prescriptions.
- **Binary-star evolution**: model Roche-lobe overflow, mass transfer, tides,
  common-envelope evolution, supernova mass loss, natal kicks, binary
  disruption, and mergers.
- **Population synthesis**: run single-star and binary population synthesis
  with configurable initial grids, metallicity models, target-source criteria,
  and output columns.
- **Galactic modelling**: couple populations to Milky Way star-formation
  history, chemical enrichment, Galactic structure, and interstellar-medium
  phases.
- **Stellar kinematics**: integrate Galactic orbits with `galpy` and record
  present-day positions, velocities, and observable astrometric quantities.

### 🛠️ Technical Highlights

- **Configuration-driven workflow**: control physical prescriptions, runtime
  options, source selection, and outputs through workspace-level Python
  configuration files.
- **Parallel population synthesis**: use multiprocessing for large parameter
  grids and asynchronous output writing for large binary-population catalogues.
- **Optional JIT acceleration**: use Numba to accelerate repeated stellar and
  binary evolution calculations.
- **Structured output management**: write selected output columns, intermediate
  batches, merged catalogues, and runtime logs in a reproducible workspace.
- **Modular source tree**: separate stellar physics, galaxy models, kinematics,
  physics utilities, observables, and drivers for easier extension.

### 📊 Physics and Observable Modules

- **Stellar and binary physics**: winds, magnetic braking, mass-transfer
  stability, common-envelope evolution, compact-remnant formation, and natal
  kicks.
- **Supernova prescriptions**: rapid, delayed, and stochastic explosion models,
  with configurable CCSN, ECSN, and AIC kick prescriptions.
- **Metallicity models**: constant-metallicity calculations and Galactic
  chemical-enrichment based population synthesis.
- **Gravitational waves**: LISA signal-to-noise estimates for compact binaries,
  with source-selection helpers for precomputed SNR columns.
- **Isolated black-hole accretion**: post-processing summaries for black holes
  accreting from different ISM phases.

### ⚡ Application Areas

- **Compact-object populations**: white dwarfs, neutron stars, stellar-mass
  black holes, and their binary combinations.
- **Galactic compact binaries**: NS-NS, BH-BH, NS-BH, NS-WD, and WD-WD systems,
  including potential Galactic gravitational-wave sources.
- **Runaway and disrupted systems**: kinematic outcomes of supernova kicks,
  binary disruption, and post-supernova orbital evolution.
- **Population-level observables**: catalogue-level analysis of source numbers,
  spatial distributions, accretion properties, and gravitational-wave
  detectability.
- **Future extensions**: X-ray binary observables, electromagnetic transients,
  microlensing, and survey-selection tools are planned extension points.

## 🚀 Installation

POPKIN is recommended to be installed in a dedicated Conda environment.

Run the following commands in the directory where you want to place the POPKIN
source tree:

```bash
git clone https://github.com/JianguoHe/POPKIN.git POPKIN
cd POPKIN
conda env create -f environment.yml
conda activate popkin
```

Install `galpy`, which is required by POPKIN's Galactic kinematics module:

```bash
# macOS
conda install -c conda-forge gsl galpy

# Linux
python -m pip install --only-binary galpy galpy
```

Verify the installation:

```bash
python -c "import popkin; print(popkin.__version__)"
```

### Optional Dependencies

The experimental `gala`-based orbit module is kept in the source tree for users
who want to test it, but the default orbit integration path uses `galpy`.

## 📖 Quick Start

POPKIN is designed to run from a user workspace. A workspace contains the main
configuration file, program-specific configuration files, a `run.py` entry
script, and output directories. Start a new calculation by copying the template
workspace:

```bash
cp -r work_template my_popkin_run
cd my_popkin_run
python run.py
```

The selected program is controlled by `program` in `inlist.py`:

```python
program = "popbin"  # "sse", "bse", "popsin", or "popbin"
```

Program-specific settings are placed in the corresponding configuration file:

- `inlist_sse.py` for single-star evolution.
- `inlist_bse.py` for binary-star evolution.
- `inlist_popsin.py` for single-star population synthesis.
- `inlist_popbin.py` for binary population synthesis.

Runtime logs are written to `logs/`, and simulation outputs are written under
`data/` unless a custom output directory is configured.

## ⚙️ Configuration Model

Default parameters are defined in
`src/popkin/config/controls_default.py`. User workspaces override these defaults
through local configuration files.

### Main Configuration

`inlist.py` controls global runtime and physical settings, including:

- selected program (`sse`, `bse`, `popsin`, or `popbin`);
- metallicity model;
- multiprocessing and JIT options;
- Galactic orbit integration and gravitational-wave SNR switches;
- common-envelope, mass-transfer, stellar-wind, magnetic-braking, and
  supernova prescriptions;
- natal-kick models, including the default Hobbs et al. Maxwellian model and an
  optional Disberg & Mandel lognormal CCSN kick model.

### Program-Specific Configuration

Program-specific files control settings that depend on the selected calculation:

- initial stellar or binary parameters;
- population-synthesis grids and initial distributions;
- source-selection criteria;
- output columns;
- output format and precision;
- optional custom data directories for population-synthesis runs.

Because these files are Python modules, users can define variables, lists,
dictionaries, and expressions directly.

## 🧭 Running Modes

### Single-Star Evolution

Use `program = "sse"` in `inlist.py` and configure the stellar mass,
metallicity, and output options in `inlist_sse.py`.

The underlying class can also be used directly:

```python
from popkin.stars.single_star import SingleStar

star = SingleStar(type=1, mass=10.0, Z=0.02, index=0)
star.evolve()
star_track = star.data
```

### Binary-Star Evolution

Use `program = "bse"` in `inlist.py` and configure the initial binary in
`inlist_bse.py`.

The core binary class can be used directly:

```python
from popkin.stars.single_star import SingleStar
from popkin.stars.binary_star import BinaryStar

star1 = SingleStar(type=1, mass=20.0, Z=0.02, index=0)
star2 = SingleStar(type=1, mass=10.0, Z=0.02, index=0)
binary = BinaryStar(star1, star2, ecc=0.0, sep=1000.0, index=0)
binary.evolve()
binary_track = binary.data
```

### Population Synthesis

Use `program = "popsin"` or `program = "popbin"` in `inlist.py`.

`popsin` evolves a one-dimensional grid of single-star initial masses.
`popbin` evolves a multi-dimensional binary initial-parameter grid. For each
metallicity, POPKIN applies the configured source-selection criteria, writes
batch outputs asynchronously, and merges the final catalogue.

Population-synthesis runs are best launched from the workspace:

```bash
python run.py
```

## 📊 Outputs

The output structure is workspace based. A new workspace copied from
`work_template/` has the following layout:

```text
my_popkin_run/
├── data/             # simulation catalogues and intermediate batches
├── logs/             # runtime logs
├── inlist.py
├── inlist_sse.py
├── inlist_bse.py
├── inlist_popsin.py
├── inlist_popbin.py
└── run.py
```

Output columns are configured in the program-specific inlist files. Large
binary-population outputs are handled by `OutputManager`, which writes
intermediate batches and merges the final catalogue after the worker pool
finishes.

Supported output formats are controlled by the relevant inlist file and the
current `OutputManager` implementation.

## 🏗️ Project Structure

```text
POPKIN/
├── src/popkin/
│   ├── config/              # user configuration, paths, logging, output manager
│   ├── drivers/             # sse, bse, popsin, and popbin entry points
│   ├── stars/               # single-star and binary-star evolution physics
│   ├── galaxies/            # Milky Way model and planned external-galaxy modules
│   ├── kinematics/          # orbit integration and astrometry utilities
│   ├── binding_energy/      # common-envelope binding-energy prescriptions
│   ├── metallicity/         # metallicity-dependent stellar fitting coefficients
│   ├── physics/             # reusable physics relations
│   ├── observables/         # GW, isolated-BH accretion, and planned observables
│   ├── constants.py         # physical constants and structured output dtypes
│   └── utils.py             # shared utility functions
├── tests/                   # development tests and validation scripts
├── work_template/           # user workspace template
├── environment.yml
├── pyproject.toml
└── README.md
```

## 🔭 Observables and Post-processing

Current observable and post-processing modules include:

- `observables/gravitational_waves.py`: LISA SNR estimates for compact binaries
  and helpers for selecting sources above a precomputed SNR threshold.
- `observables/isolated_bh_accretion.py`: post-processing summaries for
  isolated black holes accreting from different ISM phases.

The following modules are present as early-stage extension points and will be
expanded in future versions:

- X-ray binary observables.
- Electromagnetic transient utilities.
- Microlensing observables.
- Survey-selection utilities.

## 🛠️ Development Notes

POPKIN uses Numba conditionally. If `jit_enabled = True`, the stellar-evolution
classes are compiled before large population-synthesis calculations. On
Unix-like systems, the population-synthesis drivers use the `fork`
multiprocessing start method so worker processes can inherit warmed-up compiled
state through copy-on-write memory sharing.

Binary population synthesis can generate very large catalogues. Prefer selecting
only the columns needed for a scientific application and use program-specific
source-selection criteria to keep outputs manageable.

## 🧪 Testing

Run the available tests from the repository root:

```bash
pytest tests/
```

Some scripts under `tests/` are exploratory validation or plotting scripts
rather than formal unit tests, so the test suite should be treated as a
development aid rather than a complete release validation suite.

## 📝 Citation

If you use POPKIN in a publication, please cite the POPKIN paper when available.
For the underlying BSE framework, cite:

```bibtex
@article{Hurley2002,
  author  = {Hurley, J. R. and Tout, C. A. and Pols, O. R.},
  title   = {Evolution of binary stars and the effect of tides on binary populations},
  journal = {Monthly Notices of the Royal Astronomical Society},
  year    = {2002},
  volume  = {329},
  pages   = {897--928}
}
```

Please also cite the relevant physical prescriptions and third-party packages
used in a given calculation, such as `NumPy`, `SciPy`, `Astropy`, `Numba`,
`galpy`, and `LEGWORK`, where appropriate.

## 📄 License

POPKIN is intended to be released under the MIT License. See `LICENSE` for the
license text.

## 📬 Contact

Maintainer: Jianguo He (`hejg@smail.nju.edu.cn`)

Repository: https://github.com/JianguoHe/POPKIN

Issue tracker: https://github.com/JianguoHe/POPKIN/issues

The DOI link will be updated before the public release.
