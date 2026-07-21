# POPKIN: Population Synthesis and Stellar Kinematics

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue)](https://doi.org/10.xxxx/xxxxx)

POPKIN is a modular Python framework for rapid single-star evolution, 
binary-star evolution, population synthesis, and stellar kinematics. 
It was developed to connect compact-object formation physics with the
present-day positions, velocities, and observable signatures of stellar
populations in the Milky Way.

The code builds on the classic Binary Star Evolution (BSE) framework and
reorganizes the calculation into an inspectable, object-oriented Python
codebase. In addition to intrinsic stellar and binary evolution, POPKIN couples
synthetic populations to Galactic birth environments, star-formation and
metallicity histories, orbit integration in the Galactic potential,
interstellar-medium properties, and observable post-processing such as
compact-binary gravitational-wave signal-to-noise ratios, isolated black-hole
accretion, and microlensing observables.

> Development status: POPKIN is under active development. Interfaces, default
> prescriptions, and example workspaces may change before a stable public
> release.

## 💡 Why POPKIN
Modern compact-object surveys increasingly require models that do more than
evolve binaries in isolation. For Galactic isolated black holes (IBHs), neutron
stars, compact binaries, and runaway systems, the observable population is
shaped jointly by stellar evolution, binary disruption, natal kicks, Galactic
star-formation history, chemical enrichment, orbital motion, and survey
selection.

POPKIN is designed for this coupled problem. It provides:

- **Star evolution**: rapid single-star and binary-star evolution 
  from the zero-age main sequence to remnants.
- **Population synthesis**: population synthesis for single and binary 
  stellar systems with statistical weights.
- **Galactic evolution history**: a Milky Way model with thin-disk, thick-disk, 
  and bulge star formation, chemical enrichment, and interstellar-medium phases.
- **Orbital motion tracking**: piecewise Galactic orbit integration for systems 
  whose velocities change after supernovae, binary disruption, or mergers.
- **Post-processing**: tools for turning synthetic populations into observable
  quantities.

These components can be used to study compact-object populations, Galactic
compact binaries, runaway and disrupted systems, isolated black-hole accretion,
X-ray binaries, electromagnetic transients, microlensing, compact-binary
gravitational-wave sources, and related survey predictions.


## 🏗️ Architecture

The architecture of POPKIN is shown below:

![POPKIN structure](/POPKIN_structure.png)

POPKIN is organized around four connected layers.

- **Galactic model**: star-formation history, chemical enrichment, Galactic
   components, birth positions, and interstellar-medium phases.
- **Stellar physics**: single-star evolution, binary-star evolution, winds,
   magnetic braking, mass transfer, common-envelope evolution, remnant
   formation, and natal kicks.
- **Kinematic evolution**: orbit integration in the Galactic potential using
   `galpy`, including velocity changes caused by supernovae and disrupted
   binaries.
- **Observable post-processing**: compact-binary gravitational-wave
   signal-to-noise ratios, accretion from the ISM, and microlensing quantities
   for compact lenses.

The current framework treats non-single stellar systems as binaries. Multiple
systems, star clusters, time-dependent Galactic potentials, and more detailed
survey models are planned extension points.


## ⚡ Main Features

### ✨ Stellar and Binary Evolution

- Single-star evolution from the ZAMS to white dwarfs, neutron stars, and black
  holes.
- Binary evolution with Roche-lobe overflow, tides, stable and unstable mass
  transfer, common-envelope evolution, supernova mass loss, natal kicks, binary
  disruption, and mergers.
- Configurable prescriptions for stellar winds, remnant masses, natal kicks,
  common-envelope binding energy, magnetic braking, and accretion efficiency.
- Core `SingleStar` and `BinaryStar` classes that can be used directly for
  detailed evolutionary-track studies.

### 📊 Population Synthesis

- Single-star population synthesis over stellar initial mass.
- Binary population synthesis over primary mass, secondary mass, and orbital
  period or separation.
- Mass-dependent binary fractions, configurable initial distributions, and
  statistical weights for synthetic populations.
- Constant-metallicity calculations and Galactic chemical-enrichment based
  calculations.
- User-defined source-selection criteria and output columns.

### 💫 Galactic Kinematics

- Birth positions and formation rates tied to the Milky Way star-formation and
  metallicity model.
- Orbit integration with `galpy` and `MWPotential2014`; an optional Galactic
  center supermassive black hole contribution can be added for inner-Galaxy studies.
- Piecewise orbit integration when supernovae, natal kicks, or binary
  disruption change a system's velocity state.
- Present-day Galactocentric and sky-coordinate outputs for synthetic sources.

### 🔭 Observable Modules

- Compact-binary GW SNR estimates, currently configured for LISA-style
  calculations through `LEGWORK`.
- Accreting IBH post-processing using ISM phase information, Bondi-Hoyle gas
  capture, radiatively inefficient accretion-flow corrections, luminosities,
  and fluxes.
- Compact-object microlensing quantities, including relative proper motion,
  angular Einstein radius, Einstein crossing time, and event-rate weights for
  user-selected lens catalogues.


## 🚀 Installation

We recommend installing POPKIN in a dedicated Conda environment.

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
program = "sse"  # "sse", "bse", "popsin", or "popbin"
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
  optional Disberg & Mandel 2025 lognormal CCSN kick model.

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

For population-synthesis runs (`popsin` and `popbin`), merged catalogue outputs
support `parquet`, `csv`, `hdf5`, and `npy` formats, controlled by the relevant
inlist file. `parquet` is the default and is recommended for large catalogues
because it is compact and fast to read, while `csv` is mainly useful for small,
human-readable outputs. Single-track `sse` and `bse` runs write CSV evolution
tracks.

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
│   ├── observables/         # GW, accretion, microlensing, and survey utilities
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
- `observables/microlensing.py`: compact-lens microlensing quantities, including
  relative proper motion, angular Einstein radius, Einstein crossing time, and
  event-rate weights.
- `observables/survey_selection.py`: shared flux, luminosity, and threshold
  selection utilities.
- `observables/xray_binaries.py`: basic luminosity, flux, and candidate-selection
  helpers for X-ray binaries and related accreting systems.
- `observables/electromagnetic_transients.py`: lightweight event-selection and
  compact-merger classification utilities.

The X-ray binary and electromagnetic-transient modules are intentionally
lightweight at this stage. More detailed prescriptions for duty cycles, spectral
states, beaming, ejecta properties, light curves, and survey-specific selection
functions can be added in future versions.

## 🛠️ Development Notes

POPKIN uses Numba conditionally. If `jit_enabled = True`, the stellar-evolution
classes are compiled before large population-synthesis calculations. On
Unix-like systems, the population-synthesis drivers use the `fork`
multiprocessing start method so worker processes can inherit warmed-up compiled
state through copy-on-write memory sharing.

Binary population synthesis can generate very large catalogues. Prefer selecting
only the columns needed for a scientific application and use program-specific
source-selection criteria to keep outputs manageable.

## 🧪 Testing and Validation

The repository currently includes development and validation scripts under
`tests/`. These scripts are mainly used to check individual physics modules,
galaxy-model utilities, orbit integration, and plotting diagnostics during
development.

They are not yet organized as a complete automated `pytest` test suite. Users
who wish to run them should inspect each script first, since some scripts may
generate figures, write diagnostic files, or require optional dependencies.

## 📝 Citation

If you use POPKIN in a publication, please cite the POPKIN paper when available.
For the underlying BSE framework, cite:

```bibtex
@ARTICLE{Hurley2002,
       author = {{Hurley}, Jarrod R. and {Tout}, Christopher A. and {Pols}, Onno R.},
        title = "{Evolution of binary stars and the effect of tides on binary populations}",
      journal = {\mnras},
     keywords = {METHODS: ANALYTICAL, METHODS: STATISTICAL, BINARIES: GENERAL, STARS: EVOLUTION, STARS: VARIABLES: OTHER, GALAXIES: STELLAR CONTENT, Astrophysics},
         year = 2002,
        month = feb,
       volume = {329},
       number = {4},
        pages = {897-928},
          doi = {10.1046/j.1365-8711.2002.05038.x},
archivePrefix = {arXiv},
       eprint = {astro-ph/0201220},
 primaryClass = {astro-ph},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2002MNRAS.329..897H},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

Please also cite the relevant physical prescriptions and third-party packages
used in a given calculation, such as `NumPy`, `SciPy`, `Astropy`, `Numba`,
`galpy`, and `LEGWORK`, where appropriate.

## 📄 License

POPKIN is released under the MIT License. See `LICENSE` for the license text.

## 📬 Contact

Maintainer: Jianguo He (`hejg@smail.nju.edu.cn`)

Repository: https://github.com/JianguoHe/POPKIN

Issue tracker: https://github.com/JianguoHe/POPKIN/issues

The DOI link will be updated before the public release.
