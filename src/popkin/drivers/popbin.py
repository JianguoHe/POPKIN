"""Binary star population synthesis driver."""

import multiprocessing as mp
import os
import queue
import time

import numpy as np
import tqdm

from popkin.config.controls_default import (
    parallel, max_time, metallicity_model, Z, Z_list, IMF_scheme, binary_fraction,
    target_popbin, popbin_output_columns, simplify_output, output_format, output_precision,
    enable_orbital_integration, include_GC_SMBH, info_orbit, calculate_gw_snr, jit_enabled,
    ini_orbit_scheme, ini_ecc_scheme, n_grid_popbin, m1_range, m2_range, log10_P_range, sep_range,
)
from popkin.config.logger import get_logger, timer
from popkin.config.output_manager import OutputManager
from popkin.config.paths import paths
from popkin.config.user_config import apply_user_config, log_loaded_configs
from popkin.galaxies import MilkyWay
from popkin.kinematics.orbit import OrbitIntegrator
from popkin.observables.gravitational_waves import calc_gw_snr
from popkin.stars.binary_star import BinaryStar
from popkin.stars.single_star import SingleStar
from popkin.utils import (
    process_single_star_data, process_binary_star_data,
    merge_structured_data, select_structured_data,
    create_popbin_parameter_space, search_popbin_source, get_metallicity_str,
)

logger = get_logger(__name__)

POPBIN_PROGRESS_LOG_STEP = 0.05

loaded_configs = apply_user_config(globals(), "inlist", "inlist_popbin")
log_loaded_configs(logger, loaded_configs)


_WORKER_GALAXY: MilkyWay | None = None
_WORKER_OUTPUT_QUEUES: dict[str, list] | None = None
_WORKER_QUEUE_CURSOR = 0


def init_popbin_worker(galaxy, output_queues):
    global _WORKER_GALAXY, _WORKER_OUTPUT_QUEUES, _WORKER_QUEUE_CURSOR

    _WORKER_GALAXY = galaxy
    _WORKER_OUTPUT_QUEUES = output_queues
    _WORKER_QUEUE_CURSOR = os.getpid()


def put_output_data(filename, data):
    """Put data into any available output queue; poll with wait when all queues are full."""
    global _WORKER_QUEUE_CURSOR

    if _WORKER_OUTPUT_QUEUES is None:
        raise RuntimeError("popbin worker output queues have not been initialized")

    queues = _WORKER_OUTPUT_QUEUES[filename]
    n_queues = len(queues)

    start = _WORKER_QUEUE_CURSOR % n_queues
    _WORKER_QUEUE_CURSOR += 1

    for offset in range(n_queues):
        try:
            queues[(start + offset) % n_queues].put_nowait(data)
            return
        except queue.Full:
            pass

    while True:
        for offset in range(n_queues):
            try:
                queues[(start + offset) % n_queues].put(data, block=True, timeout=0.05)
                return
            except queue.Full:
                continue


def create_binary(star1, star2, orbit_param, ecc, index):
    if ini_orbit_scheme == "Sana2012":
        return BinaryStar(star1=star1, star2=star2, period=orbit_param, ecc=ecc, index=index)

    if ini_orbit_scheme == "Hurley2002":
        return BinaryStar(star1=star1, star2=star2, sep=orbit_param, ecc=ecc, index=index)

    raise ValueError(f"Unsupported orbit model: {ini_orbit_scheme}")


def create_prefixed_star_data(star_name, star_data, cols_star):
    dtype = [(f"{star_name}_{col}", star_data.dtype[col]) for col in cols_star]
    out = np.empty(star_data.shape, dtype=dtype)

    for col, (new_name, _) in zip(cols_star, dtype):
        out[new_name] = star_data[col]

    return out


def create_simplify_mask(binary_data):
    mask = np.zeros(len(binary_data), dtype=bool)
    if len(binary_data) == 0:
        return mask

    mask[0] = True
    mask[1:] = (
        (binary_data["type1"][1:] != binary_data["type1"][:-1]) |
        (binary_data["type2"][1:] != binary_data["type2"][:-1])
    )
    mask |= binary_data["event"] != ""
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


def log_population_progress(done, total, start_time, z, next_fraction):
    """Log coarse-grained population-synthesis progress for batch/server runs."""
    if total <= 0:
        return next_fraction

    fraction = done / total
    if done < total and fraction < next_fraction:
        return next_fraction

    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else float("nan")

    logger.info(
        f"Population progress (Z={z}): {done}/{total} "
        f"({fraction:.1%}), rate={rate:.1f} sys/s, "
        f"elapsed={elapsed:.0f}s, eta={remaining:.0f}s",
        extra={"console": True},
    )

    while next_fraction <= fraction:
        next_fraction += POPBIN_PROGRESS_LOG_STEP
    return next_fraction


def popbin_main(args) -> None:
    galaxy = _WORKER_GALAXY

    if galaxy is None or _WORKER_OUTPUT_QUEUES is None:
        raise RuntimeError("popbin worker has not been initialized")

    index, row = args
    m1, m2, orbit_param, ecc, weight = row

    if m1 < m2:
        return

    star1 = SingleStar(type=1, mass=m1, Z=galaxy.Z, index=index)
    star2 = SingleStar(type=1, mass=m2, Z=galaxy.Z, index=index)
    binary = create_binary(star1, star2, orbit_param, ecc, index)

    try:
        binary.evolve()
    except Exception as e:
        logger.error(
            f"Evolution failed: {type(e).__name__}: {e} | "
            f"Initial params: index={index}, m1={m1:.2f}, m2={m2:.2f}, orbit={orbit_param:.2f}, ecc={ecc:.3f}",
            extra={"console": True},
        )
        return

    star1_data = process_single_star_data(star1)
    star2_data = process_single_star_data(star2)
    binary_data = process_binary_star_data(binary, star1, star2)

    if binary_data["time"][-1] < max_time:
        return

    files = []
    conditions = []

    for target in target_popbin:
        condition = search_popbin_source(star1_data, star2_data, binary_data, target)
        if np.any(condition):
            files.append(f"{target['filename']}_{get_metallicity_str(galaxy.Z)}")
            conditions.append(condition)

    if not files:
        return

    try:
        np.random.seed(index)
        stars = galaxy.generate_star(tau=binary_data["time"] / 1000, weight=weight)
    except Exception as e:
        logger.error(
            f"Failed to generate stellar samples from galaxy model: {type(e).__name__}: {e} | "
            f"Initial params: index={index}, m1={m1:.2f}, m2={m2:.2f}, orbit={orbit_param:.2f}, ecc={ecc:.3f}",
            extra={"console": True},
        )
        return

    binary_birth = stars[["origin", "ini_x", "ini_y", "ini_z", "ini_rho", "ini_phi", "ini_dist", "rate"]]

    dt = np.zeros(len(binary_data))
    dt[:-1] = np.diff(binary_data["time"])

    binary_extra = np.zeros(len(binary_data), dtype=[
        ("index", "i8"),
        ("Z", "f8"),
        ("weight", "f8"),
        ("dt", "f8"),
        ("num", "f8"),
    ])
    binary_extra["index"] = index
    binary_extra["Z"] = galaxy.Z
    binary_extra["weight"] = weight
    binary_extra["dt"] = dt
    binary_extra["num"] = dt * binary_birth["rate"] * 1e6

    data_parts = [binary_data, binary_birth, binary_extra]
    cols_output = list(popbin_output_columns["binary"])

    for star_name, star_data in [("star1", star1_data), ("star2", star2_data)]:
        cols_star = popbin_output_columns.get(star_name)
        if not cols_star:
            continue

        data_parts.append(create_prefixed_star_data(star_name, star_data, cols_star))
        cols_output += [f"{star_name}_{col}" for col in cols_star]

    binary_data = merge_structured_data(data_parts)

    if enable_orbital_integration:
        orbit = OrbitIntegrator(
            data=binary_data,
            obj_type="binary",
            info_orbit=info_orbit,
            include_GC_SMBH=include_GC_SMBH,
        )

        try:
            orbit.integrate()
        except Exception as e:
            logger.error(
                f"Orbit integration failed: {type(e).__name__}: {e} | "
                f"Initial params: index={index}, m1={m1:.2f}, m2={m2:.2f}, orbit={orbit_param:.2f}, ecc={ecc:.3f}",
                extra={"console": True},
            )
            return

        binary_data = merge_structured_data([orbit.data, orbit.orbit_data])
        cols_output += orbit.cols_orbit

    if calculate_gw_snr:
        gw_snr_data = calc_gw_snr(data=binary_data, orbit_integration=enable_orbital_integration)
        binary_data = merge_structured_data([binary_data, gw_snr_data])
        cols_output += list(gw_snr_data.dtype.names)

    if simplify_output:
        mask = create_simplify_mask(binary_data)

        for filename, condition in zip(files, conditions):
            put_output_data(
                filename,
                select_structured_data(binary_data, cols_output, mask | condition),
            )
    else:
        data_out = select_structured_data(binary_data, cols_output)
        for filename in files:
            put_output_data(filename, data_out)


@timer("Binary Population Synthesis")
def popbin() -> None:
    if metallicity_model not in ("constant", "enrichment"):
        raise ValueError(
            f"Unsupported metallicity model: '{metallicity_model}'. "
            f"Expected 'constant' or 'enrichment'."
        )

    if not target_popbin:
        raise ValueError(
            "Empty target list. Please provide at least one target source.\n"
            "Documentation reference: /src/popkin/config/controls_default.py"
        )

    if ini_orbit_scheme == "Sana2012":
        orbit_param_range = (10 ** log10_P_range[0], 10 ** log10_P_range[1])
    elif ini_orbit_scheme == "Hurley2002":
        orbit_param_range = sep_range
    else:
        raise ValueError(f"Unsupported orbit model: {ini_orbit_scheme}")

    ini_parameter_space = create_popbin_parameter_space(
        m1_range=m1_range,
        m2_range=m2_range,
        orbit_param_range=orbit_param_range,
        n_grid_popbin=n_grid_popbin,
        ini_orbit_scheme=ini_orbit_scheme,
        ini_ecc_scheme=ini_ecc_scheme,
        IMF_scheme=IMF_scheme,
        binary_fraction=binary_fraction,
    )

    valid_tasks = [
        (index, row)
        for index, row in enumerate(ini_parameter_space)
        if row[0] >= row[1]
    ]

    logger.info(
        f"Parameter space filter: total={len(ini_parameter_space)}, "
        f"valid={len(valid_tasks)}, "
        f"skipped_m1_lt_m2={len(ini_parameter_space) - len(valid_tasks)}",
        extra={"console": True},
    )

    mp.set_start_method("fork", force=True)

    if jit_enabled:
        logger.info("Warming up JIT compilation ...", extra={"console": True})
        star = SingleStar(type=1, mass=2.0, Z=0.02, index=1)
        create_binary(star, star, 1000.0, 0.0, 1).evolve()

    z_values = [Z] if metallicity_model == "constant" else Z_list
    model_name = "Constant metallicity" if metallicity_model == "constant" else "Enrichment metallicity"
    logger.info(f"Using {model_name} model, Z values: {z_values}", extra={"console": True})

    worker_parallel = max(1, int(parallel))

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
            for target in target_popbin
        ]
        clean_old_outputs(filenames)

        manager = OutputManager(
            parallel=worker_parallel,
            data_dir=paths.data,
            output_format=output_format,
            output_precision=output_precision,
        )
        queues = manager.start(targets=target_popbin, Z=z)

        logger.info(f"Parallel configuration: workers={worker_parallel}, chunksize=50", extra={"console": True})

        progress_start = time.time()
        next_progress_fraction = POPBIN_PROGRESS_LOG_STEP
        completed_tasks = 0

        with mp.Pool(
            processes=worker_parallel,
            initializer=init_popbin_worker,
            initargs=(galaxy, queues),
        ) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(popbin_main, valid_tasks, chunksize=50),
                total=len(valid_tasks),
                desc=f"Population synthesis progress (Z={z})",
                unit="sys",
            ):
                completed_tasks += 1
                next_progress_fraction = log_population_progress(
                    completed_tasks,
                    len(valid_tasks),
                    progress_start,
                    z,
                    next_progress_fraction,
                )

        logger.info("Stopping OutputManager...", extra={"console": True})
        manager.stop()

        logger.info("Merging final files...", extra={"console": True})
        manager.merge_all(keep_batches=False)


if __name__ == "__main__":
    popbin()
