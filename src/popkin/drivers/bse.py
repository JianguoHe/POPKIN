"""Binary-star evolution driver."""

import pandas as pd

from popkin.config.controls_default import Z, m1, m2, period, sep, ecc, type1, type2, index_bse
from popkin.config.controls_default import sse_output_columns, bse_output_columns, output_precision
from popkin.config.logger import get_logger, timer
from popkin.config.paths import paths
from popkin.config.user_config import apply_user_config, log_loaded_configs
from popkin.stars.single_star import SingleStar
from popkin.stars.binary_star import BinaryStar
from popkin.utils import process_single_star_data, process_binary_star_data

logger = get_logger(__name__)

loaded_configs = apply_user_config(globals(), "inlist", "inlist_bse")
log_loaded_configs(logger, loaded_configs)


@timer("Binary Star Evolution")
def bse() -> None:
    """Run binary-star evolution and save results to CSV."""
    logger.info(
        f"Initializing binary: m1={m1:.2f} M_sun, m2={m2:.2f} M_sun, "
        f"Z={Z:.4f}, ecc={ecc:.3f}",
        extra={"console": True},
    )

    star1 = SingleStar(type=type1, mass=m1, Z=Z, index=index_bse)
    star2 = SingleStar(type=type2, mass=m2, Z=Z, index=index_bse)
    logger.info("Single-star instances created")

    if period is not None:
        binary = BinaryStar(star1=star1, star2=star2, period=period, ecc=ecc, index=index_bse)
        logger.info(f"Initialized from orbital period: period={period:.2f} days", extra={"console": True})
    elif sep is not None:
        binary = BinaryStar(star1=star1, star2=star2, sep=sep, ecc=ecc, index=index_bse)
        logger.info(f"Initialized from semi-major axis: sep={sep:.2f} R_sun", extra={"console": True})
    else:
        logger.error("Either period or sep must be provided", extra={"console": True})
        raise ValueError("Either period or sep must be provided")

    logger.info("Running evolution...", extra={"console": True})
    try:
        binary.evolve()
        logger.info(f"Evolution completed: steps={len(binary.data)}", extra={"console": True})
    except Exception as e:
        logger.error(f"Evolution failed: {type(e).__name__}: {e}", extra={"console": True})
        raise

    star1_data = process_single_star_data(star1)
    star2_data = process_single_star_data(star2)
    binary_data = process_binary_star_data(binary, star1, star2)

    df_star1 = pd.DataFrame(star1_data)
    df_star2 = pd.DataFrame(star2_data)
    df_binary = pd.DataFrame(binary_data)

    with open(paths.data / "binary.csv", "w") as file:
        df_binary[bse_output_columns].to_csv(file, header=True, index=False, float_format=f'%.{output_precision}g')

    with open(paths.data / "star1.csv", "w") as file:
        df_star1[sse_output_columns].to_csv(file, header=True, index=False, float_format=f'%.{output_precision}g')

    with open(paths.data / "star2.csv", "w") as file:
        df_star2[sse_output_columns].to_csv(file, header=True, index=False, float_format=f'%.{output_precision}g')

    logger.info(f"Data saved: {paths.data}", extra={"console": True})

    _log_evolution_summary(df_binary)


def _log_evolution_summary(data: pd.DataFrame) -> None:
    """Log a compact evolution summary.

    Args:
        data: Binary evolution data.
    """
    df = data[
        (data["type1"] != data["type1"].shift()) |
        (data["type2"] != data["type2"].shift()) |
        (data["event"] != "")
    ]

    columns_show = [
        "time", "ecc", "period", "sep", "type1", "type2",
        "m1", "m2", "mc1", "mc2", "R1_div_RL1", "R2_div_RL2", "event",
    ]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)

    logger.info(
        "\n%s\n%s\n%s",
        f"bse parameters: m1={m1:.2f} M_sun, m2={m2:.2f} M_sun, Z={Z:.4f}, ecc={ecc:.3f}",
        "bse result:",
        df[columns_show].to_string(index=False),
        extra={"console": True},
    )


if __name__ == "__main__":
    bse()
