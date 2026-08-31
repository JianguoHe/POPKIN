"""Plot compact-remnant outcomes as a function of the pre-SN CO-core mass.

The F12 panels show the standard CO-core-mass intervals of the rapid and
delayed prescriptions.  The exact F12 remnant mass also depends on the total
pre-SN mass, so these two rows should be interpreted as a schematic summary.
The M25 rows use the thresholds implemented in ``SingleStar`` at Z_sun and
0.1 Z_sun, with the default fallback model B.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR / "figures"
FIGURE_PDF = FIGURE_DIR / "remnant_type_vs_co_core.pdf"

MCO_MIN = 2.0
MCO_MAX = 20.0
ROW_HEIGHT = 0.76

REMNANT_COLORS = {
    "NS": "#ff7f0e",
    "fallback BH": "#2ca02c",
    "direct-collapse BH": "#1f77b4",
}


def _clip_interval(x0, x1):
    """Clip an interval to the plotted CO-core-mass range."""
    return max(MCO_MIN, x0), min(MCO_MAX, x1)


def _add_deterministic_interval(ax, y, x0, x1, outcome):
    """Draw one deterministic remnant-outcome block."""
    x0, x1 = _clip_interval(x0, x1)
    if x1 <= x0:
        return
    ax.fill_between(
        [x0, x1],
        y - ROW_HEIGHT / 2,
        y + ROW_HEIGHT / 2,
        step="post",
        color=REMNANT_COLORS[outcome],
        edgecolor="white",
        linewidth=0.7,
        zorder=2,
    )


def _add_probability_interval(ax, y, x0, x1, probabilities):
    """Draw a vertically stacked probabilistic outcome interval.

    The listed probabilities determine the vertical fractions of the row,
    while the CO-core mass remains on the horizontal axis.  Outcomes are
    stacked from the bottom upward in the order given by ``probabilities``.
    """
    x0, x1 = _clip_interval(x0, x1)
    if x1 <= x0:
        return

    total = sum(probabilities.values())
    if total <= 0:
        return

    y_bottom = y - ROW_HEIGHT / 2
    for outcome, probability in probabilities.items():
        height = ROW_HEIGHT * probability / total
        ax.fill_between(
            [x0, x1],
            y_bottom,
            y_bottom + height,
            step="post",
            color=REMNANT_COLORS[outcome],
            edgecolor="white",
            linewidth=0.7,
            zorder=2,
        )
        y_bottom += height


def _add_mm20_row(ax, y):
    """Draw the stochastic MM20 outcome probabilities from the implementation."""
    x_1 = np.linspace(2.0, 7.0, 301)
    p_bh = (x_1 - 2.0) / 5.0
    p_direct_given_bh = (x_1 - 2.0) / 6.0
    p_ns = 1.0 - p_bh
    p_fallback = p_bh * (1.0 - p_direct_given_bh)
    p_direct = p_bh * p_direct_given_bh

    y_bottom = y - ROW_HEIGHT / 2
    for outcome, probability in (
        ("fallback BH", p_fallback),
        ("direct-collapse BH", p_direct),
        ("NS", p_ns),
    ):
        y_top = y_bottom + ROW_HEIGHT * probability
        ax.fill_between(
            x_1,
            y_bottom,
            y_top,
            color=REMNANT_COLORS[outcome],
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        y_bottom = y_top

    # Between M_CO=7 and 8 every remnant is a BH, with a stochastic split
    # between incomplete and complete fallback.
    x_2 = np.linspace(7.0, 8.0, 101)
    p_direct_2 = (x_2 - 2.0) / 6.0
    p_fallback_2 = 1.0 - p_direct_2
    y_bottom = y - ROW_HEIGHT / 2
    for outcome, probability in (
        ("fallback BH", p_fallback_2),
        ("direct-collapse BH", p_direct_2),
    ):
        y_top = y_bottom + ROW_HEIGHT * probability
        ax.fill_between(
            x_2,
            y_bottom,
            y_top,
            color=REMNANT_COLORS[outcome],
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
        y_bottom = y_top

    _add_deterministic_interval(ax, y, 8.0, MCO_MAX, "direct-collapse BH")


def _add_m25_row(ax, y, thresholds):
    """Draw one M25 row using the four CO-core thresholds for one MT case."""
    mco_1, mco_2, mco_3 = thresholds
    _add_deterministic_interval(ax, y, MCO_MIN, mco_1, "NS")
    _add_deterministic_interval(ax, y, mco_1, mco_2, "direct-collapse BH")

    # Default M25 fallback model B: a 10% fallback-BH probability in the
    # intermediate successful-SN region and a 90% NS probability.
    _add_probability_interval(
        ax,
        y,
        mco_2,
        mco_3,
        {"fallback BH": 0.10, "NS": 0.90},
    )
    _add_deterministic_interval(ax, y, mco_3, MCO_MAX, "direct-collapse BH")


def plot_remnant_type():
    """Generate the compact-remnant outcome diagram."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    rows = [
        ("F12(R)", "f12_rapid", None),
        ("F12(D)", "f12_delayed", None),
        ("MM20", "mm20", None),
        (r"$\mathrm{M25}(Z=Z_{\odot})$" + "\n" + "Single", "m25", (6.6, 7.2, 13.0)),
        ("Case A", "m25", (7.4, 8.4, 15.4)),
        ("Case B", "m25", (7.7, 8.3, 15.2)),
        ("Case C", "m25", (6.6, 7.1, 13.2)),
        (r"$\mathrm{M25}(Z=0.1Z_{\odot})$" + "\n" + "Single", "m25", (6.1, 6.6, 12.9)),
        ("Case A", "m25", (6.9, 7.4, 13.7)),
        ("Case B", "m25", (6.9, 7.9, 13.7)),
        ("Case C", "m25", (6.3, 7.1, 12.3)),
    ]

    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    y_positions = np.arange(len(rows), dtype=float)[::-1]

    for y, (_, model_type, thresholds) in zip(y_positions, rows):
        if model_type == "f12_rapid":
            # Rapid F12: the compact-remnant outcome follows the familiar
            # rapid-explosion CO-core intervals; the 2--5 M_sun BH gap is a
            # gap in remnant mass, not an empty CO-core interval.
            _add_deterministic_interval(ax, y, 2.0, 6.0, "NS")
            _add_deterministic_interval(ax, y, 6.0, 7.0, "direct-collapse BH")
            _add_deterministic_interval(ax, y, 7.0, 11.0, "fallback BH")
            _add_deterministic_interval(ax, y, 11.0, 20.0, "direct-collapse BH")
        elif model_type == "f12_delayed":
            # Delayed F12 has a continuous fallback branch before complete
            # fallback at M_CO=11 M_sun.
            _add_deterministic_interval(ax, y, 2.0, 3.5, "NS")
            _add_deterministic_interval(ax, y, 3.5, 11.0, "fallback BH")
            _add_deterministic_interval(ax, y, 11.0, 20.0, "direct-collapse BH")
        elif model_type == "mm20":
            _add_mm20_row(ax, y)
        else:
            _add_m25_row(ax, y, thresholds)

    ax.set(
        xlim=(MCO_MIN, MCO_MAX),
        ylim=(-0.7, len(rows) - 0.3),
        xlabel=r"Pre-SN carbon-oxygen core mass $(M_\odot)$",
        # ylabel="Supernova prescription and mass-transfer case",
        ylabel="",
        yticks=y_positions,
        yticklabels=[label for label, _, _ in rows],
    )
    ax.set_xticks(np.arange(2, 21, 2))
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labeltop=True,
        labelbottom=True,
    )

    # Separate the three SN prescriptions from the M25 cases, and the two
    # metallicity groups within M25.
    ax.axhline(7.5, color="0.25", linewidth=0.9)
    ax.axhline(3.5, color="0.55", linewidth=0.7, linestyle="--")

    handles = [
        Patch(facecolor=REMNANT_COLORS[outcome], edgecolor="none", label=outcome)
        for outcome in ("NS", "fallback BH", "direct-collapse BH")
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.58, 1.01),
        ncol=3,
        frameon=False,
    )
    # fig.text(
    #     0.58,
    #     0.015,
    #     "M25 intermediate regions: default fallback model B (90% NS, 10% fallback BH)",
    #     ha="center",
    #     va="bottom",
    #     fontsize=9.5,
    #     color="0.25",
    # )

    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.12, top=0.90)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {FIGURE_PDF}")


if __name__ == "__main__":
    plot_remnant_type()
