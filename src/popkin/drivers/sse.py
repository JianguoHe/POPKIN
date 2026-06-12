"""Single-star evolution driver."""

import pandas as pd

from popkin.config.controls_default import Z, mass, star_type, index_sse
from popkin.config.controls_default import sse_output_columns, output_precision
from popkin.config.logger import get_logger, timer
from popkin.config.paths import paths
from popkin.config.user_config import apply_user_config, log_loaded_configs
from popkin.stars.single_star import SingleStar
from popkin.utils import process_single_star_data

logger = get_logger(__name__)

loaded_configs = apply_user_config(globals(), "inlist", "inlist_sse")
log_loaded_configs(logger, loaded_configs)


@timer("Single Star Evolution")
def sse() -> None:
    """Run single-star evolution and save the result to CSV."""
    logger.info(
        f"Initializing single star: mass={mass:.2f} M_sun, Z={Z:.4f}, type={star_type}",
        extra={"console": True},
    )

    star = SingleStar(type=star_type, mass=mass, Z=Z, index=index_sse)

    logger.info("Running evolution...", extra={"console": True})
    star.evolve()
    logger.info(f"Evolution completed: steps={len(star.data)}", extra={"console": True})

    star_data = process_single_star_data(star)
    df_star = pd.DataFrame(star_data)

    output_path = paths.data / "star.csv"
    df_star[sse_output_columns].to_csv(output_path, header=True, index=False, float_format=f'%.{output_precision}g')
    logger.info(f"Data saved: {output_path}", extra={"console": True})

    _log_evolution_summary(df_star)


def _log_evolution_summary(data: pd.DataFrame) -> None:
    """Log a compact evolution summary.

    Args:
        data: Evolution data.
    """
    df = data[
        (data["type"] != data["type"].shift()) | (data["event"] != "")
    ]

    columns_show = ["time", "type", "mass", "M_core", "R", "R_core", "L", "event"]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)

    logger.info(
        "\n%s\n%s\n%s",
        f"sse parameters: mass={mass:.2f} M_sun, Z={Z:.4f}, type={star_type}",
        "sse result:",
        df[columns_show].to_string(index=False),
        extra={"console": True},
    )


if __name__ == "__main__":
    sse()
