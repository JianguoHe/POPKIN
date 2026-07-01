"""Single-star population synthesis driver."""

import multiprocessing as mp

import numpy as np
import tqdm

from popkin.config.controls_default import (
    metallicity_model, Z, Z_list, parallel,
    target_popsin, popsin_output_columns, simplify_output,
    output_format, output_precision,
    enable_orbital_integration, include_GC_SMBH, info_orbit,
    jit_enabled, m_range, n_grid_popsin, IMF_scheme, binary_fraction,
)
from popkin.config.logger import get_logger, timer
from popkin.config.output_manager import OutputManager
from popkin.config.paths import paths
from popkin.config.user_config import apply_user_config, log_loaded_configs
from popkin.galaxies import MilkyWay
from popkin.kinematics.orbit import OrbitIntegrator
from popkin.stars.single_star import SingleStar
from popkin.utils import (
    process_single_star_data,
    merge_structured_data, select_structured_data,
    create_popsin_parameter_space, search_popsin_source, get_metallicity_str,
)

logger = get_logger(__name__)

loaded_configs = apply_user_config(globals(), "inlist", "inlist_popsin")
log_loaded_configs(logger, loaded_configs)


_WORKER_GALAXY: MilkyWay | None = None
_WORKER_OUTPUT_QUEUES: dict[str, list] | None = None


def init_popsin_worker(galaxy, output_queues):
    global _WORKER_GALAXY, _WORKER_OUTPUT_QUEUES

    _WORKER_GALAXY = galaxy
    _WORKER_OUTPUT_QUEUES = output_queues


def put_output_data(filename, data):
    if _WORKER_OUTPUT_QUEUES is None:
        raise RuntimeError("popsin worker output queues have not been initialized")

    _WORKER_OUTPUT_QUEUES[filename][0].put(data)


def create_simplify_mask(star_data):
    mask = np.zeros(len(star_data), dtype=bool)

    if len(star_data) == 0:
        return mask

    mask[0] = True
    mask[1:] = star_data["type"][1:] != star_data["type"][:-1]
    mask |= star_data["event"] != ""
    return mask


def clean_old_outputs(filenames):
    removed = 0

    for filename in filenames:
        for path in paths.data.glob(f"{filename}*"):
            if path.is_file():
                path.unlink()
                removed += 1

    if removed:
        logger.info(f"[Cleanup] Deleted old output files: {removed}", extra={"console": True})


def popsin_main(args) -> None:
    galaxy = _WORKER_GALAXY

    if galaxy is None:
        raise RuntimeError("popsin worker has not been initialized")

    index, row = args
    mass, weight = row

    star = SingleStar(type=1, mass=mass, Z=galaxy.Z, index=index)

    try:
        star.evolve()
    except Exception as e:
        logger.error(
            f"Evolution failed: {type(e).__name__}: {e} | "
            f"Initial params: index={index}, mass={mass:.2f}",
            extra={"console": True},
        )
        return

    star_data = process_single_star_data(star)

    files = []
    conditions = []

    for target in target_popsin:
        condition = search_popsin_source(star_data, target)
        if np.any(condition):
            files.append(f"{target['filename']}_{get_metallicity_str(galaxy.Z)}")
            conditions.append(condition)

    if not files:
        return

    try:
        stars = galaxy.generate_star(tau=star_data["time"] / 1000, weight=weight)
    except Exception as e:
        logger.error(
            f"Failed to generate stellar samples from galaxy model: {type(e).__name__}: {e} | "
            f"Initial params: index={index}, mass={mass:.2f}",
            extra={"console": True},
        )
        return

    star_birth = stars[["origin", "ini_x", "ini_y", "ini_z", "ini_rho", "ini_phi", "ini_dist", "rate"]]

    dt = np.zeros(len(star_data))
    dt[:-1] = np.diff(star_data["time"])

    star_extra = np.zeros(len(star_data), dtype=[
        ("index", "i8"),
        ("Z", "f8"),
        ("weight", "f8"),
        ("dt", "f8"),
        ("num", "f8"),
    ])
    star_extra["index"] = index
    star_extra["Z"] = galaxy.Z
    star_extra["weight"] = weight
    star_extra["dt"] = dt
    star_extra["num"] = dt * star_birth["rate"] * 1e6

    star_data = merge_structured_data([star_data, star_birth, star_extra])
    cols_output = list(popsin_output_columns)

    if enable_orbital_integration:
        orbit = OrbitIntegrator(
            data=star_data,
            obj_type="single",
            info_orbit=info_orbit,
            include_GC_SMBH=include_GC_SMBH,
            base_seed=index,
        )

        try:
            orbit.integrate()
        except Exception as e:
            logger.error(
                f"Orbit integration failed: {type(e).__name__}: {e} | "
                f"Initial params: index={index}, mass={mass:.2f}",
                extra={"console": True},
            )
            return

        star_data = merge_structured_data([orbit.data, orbit.orbit_data])
        cols_output += orbit.cols_orbit

    if simplify_output:
        mask = create_simplify_mask(star_data)

        for filename, condition in zip(files, conditions):
            put_output_data(
                filename,
                select_structured_data(star_data, cols_output, mask | condition),
            )
    else:
        data_out = select_structured_data(star_data, cols_output)
        for filename in files:
            put_output_data(filename, data_out)


@timer("Single Population Synthesis")
def popsin() -> None:
    if metallicity_model not in ("constant", "enrichment"):
        raise ValueError(
            f"Unsupported metallicity model: '{metallicity_model}'. "
            f"Expected 'constant' or 'enrichment'."
        )

    if not target_popsin:
        raise ValueError(
            "Empty target list. Please provide at least one target source.\n"
            "Documentation reference: /src/popkin/config/controls_default.py"
        )

    ini_parameter_space = create_popsin_parameter_space(
        m_range=m_range,
        n_grid_popsin=n_grid_popsin,
        IMF_scheme=IMF_scheme,
        binary_fraction=binary_fraction,
    )

    tasks = list(enumerate(ini_parameter_space))
    worker_parallel = min(max(1, int(parallel)), len(tasks))

    mp.set_start_method("fork", force=True)

    if jit_enabled:
        logger.info("Warming up JIT compilation ...", extra={"console": True})
        SingleStar(type=1, mass=2.0, Z=0.02, index=1).evolve()

    z_values = [Z] if metallicity_model == "constant" else Z_list
    model_name = "Constant metallicity" if metallicity_model == "constant" else "Enrichment metallicity"
    logger.info(f"Using {model_name} model, Z values: {z_values}", extra={"console": True})

    for z in z_values:
        logger.info(f"Starting evolution for Z={z}", extra={"console": True})

        galaxy = MilkyWay(
            metallicity_model=metallicity_model,
            Z=z,
            IMF_scheme=IMF_scheme,
            binary_fraction=binary_fraction,
        )
        logger.info("Galaxy model created successfully", extra={"console": True})

        filenames = [
            f"{target['filename']}_{get_metallicity_str(z)}"
            for target in target_popsin
        ]
        clean_old_outputs(filenames)

        manager = OutputManager(
            parallel=worker_parallel,
            data_dir=paths.data,
            output_format=output_format,
            output_precision=output_precision,
            writer_count=1,
            queue_maxsize=4,
            batch_max_bytes=16 * 1024 ** 2,
            merge_workers=1,
        )
        queues = manager.start(targets=target_popsin, Z=z)

        logger.info(f"Parallel configuration: workers={worker_parallel}, chunksize=10", extra={"console": True})

        with mp.Pool(
            processes=worker_parallel,
            initializer=init_popsin_worker,
            initargs=(galaxy, queues),
        ) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(popsin_main, tasks, chunksize=10),
                total=len(tasks),
                desc=f"Population synthesis progress (Z={z})",
                unit="sys",
            ):
                pass

        logger.info("Stopping OutputManager...", extra={"console": True})
        manager.stop()

        logger.info("Merging final files...", extra={"console": True})
        manager.merge_all(keep_batches=False)


if __name__ == "__main__":
    popsin()
