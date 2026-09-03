"""Plot pre-supernova and compact-remnant masses for single stars."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

from popkin.stars import single_star as single_star_module
from popkin.stars.single_star import SingleStar


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = SCRIPT_DIR / "figures"
FIGURE_PATH = FIGURE_DIR / "stellar_mass_relation.pdf"

TEST_INLIST = {
    "ccsn_kick_prescription": "zero",
    "ccsn_remnant_maltsev_fallback": 0.5,
    "ccsn_remnant_maltsev_fallback_model": "B",
}

MODEL_CONFIGS = {
    "Fryer et al. (2012), rapid": {
        "ccsn_remnant_prescription": "fryer2012_rapid",
    },
    "Mandel et al. (2020)": {
        "ccsn_remnant_prescription": "mandel2020",
    },
    "Maltsev et al. (2025)": {
        "ccsn_remnant_prescription": "maltsev2025",
    },
}

REMNANT_LABELS = {
    "Fryer et al. (2012), rapid": "F12(R)",
    "Mandel et al. (2020)": "MM20",
    "Maltsev et al. (2025)": "M25(0.50)",
}

MASS_GRID = np.linspace(0.1, 100.0, 501)
REMNANT_COLORS = ("#4D4D4D", "#7B61A8", "#D62728")


def _record_single_star_model(model_config, metallicity, wind_model):
    """Evolve the mass grid and return the properties at CCSN formation."""
    parameter_values = {
        **TEST_INLIST,
        "wind_model": wind_model,
        **model_config,
    }
    parameter_names = tuple(parameter_values)
    previous_parameters = {
        name: getattr(single_star_module, name) for name in parameter_names
    }
    original_sn_remnant = SingleStar.SN_remnant
    ccsn_records = {}

    def record_sn_remnant(star, mcbagb):
        # M_core has two meanings in the AGB treatment. During EAGB,
        # M_co_core stores the CO-core mass; during TPAGB and He-star
        # evolution, M_core already denotes the CO-core mass.
        if star.type == 5:
            m_co = star.M_co_core
        else:
            m_co = star.M_core

        ccsn_records[id(star)] = {
            "mass_pre_sn": star.mass,
            "mass_he": mcbagb,
            "mass_co": m_co,
        }
        original_sn_remnant(star, mcbagb)
        ccsn_records[id(star)]["mass_remnant"] = star.mass

    for name, value in parameter_values.items():
        setattr(single_star_module, name, value)

    SingleStar.SN_remnant = record_sn_remnant
    try:
        data = {
            "mass0": MASS_GRID.copy(),
            "mass_pre_sn": np.full(MASS_GRID.size, np.nan),
            "mass_he": np.full(MASS_GRID.size, np.nan),
            "mass_co": np.full(MASS_GRID.size, np.nan),
            "mass_remnant": np.full(MASS_GRID.size, np.nan),
        }

        for index, mass0 in enumerate(MASS_GRID):
            star = SingleStar(type=1, mass=float(mass0), Z=metallicity, index=index)
            star.evolve()
            record = ccsn_records.get(id(star))
            if record is not None:
                for name, value in record.items():
                    data[name][index] = value

        return data
    finally:
        SingleStar.SN_remnant = original_sn_remnant
        for name, value in previous_parameters.items():
            setattr(single_star_module, name, value)


def _plot_pre_sn_masses(ax, data, colors):
    """Plot the progenitor mass and core masses shared by both SN models."""
    quantities = (
        ("mass_pre_sn", "total"),
        ("mass_he", "helium core"),
        ("mass_co", "carbon oxygen core"),
    )
    for color, (name, label) in zip(colors, quantities):
        ax.plot(
            data["mass0"],
            data[name],
            color=color,
            linewidth=4.0,
            label=label,
        )


def _plot_remnant_masses(ax, model_data, colors):
    """Plot the model-dependent compact-remnant masses without connecting lines."""
    for color, (model_name, data) in zip(colors, model_data.items()):
        ax.plot(
            data["mass0"],
            data["mass_remnant"],
            color=color,
            marker="o",
            markersize=1.5,
            markerfacecolor="none",
            markeredgewidth=0.7,
            alpha=1,
            linestyle="None",
            label=model_name,
        )



def test_stars():
    """Generate the single-star mass relation figure used for diagnostics."""
    sns.set_theme(style="ticks", context="notebook")
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 12,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    })
    colors = sns.color_palette()

    panel_configs = [
        (0.01, "merritt2026"),
        (0.01, "belczynski2010"),
        (0.0001, "merritt2026"),
        (0.0001, "belczynski2010"),
    ]
    wind_labels = {
        "merritt2026": "Merritt et al. (2026)",
        "belczynski2010": "Belczynski et al. (2010)",
    }
    panel_data = [
        {
            model_name: _record_single_star_model(model_config, metallicity, wind_model)
            for model_name, model_config in MODEL_CONFIGS.items()
        }
        for metallicity, wind_model in panel_configs
    ]

    finite_values = np.concatenate([
        values[np.isfinite(values)]
        for model_data in panel_data
        for data in model_data.values()
        for name, values in data.items()
        if name != "mass0"
    ])
    y_max = max(20.0, 1.05 * np.max(finite_values))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True, sharey=True)
    axes = axes.ravel()
    progenitor_labels = ("total", "helium core", "carbon oxygen core")

    for panel_index, (ax, (metallicity, wind_model), model_data) in enumerate(
        zip(axes, panel_configs, panel_data)
    ):
        print(f'Processing panel {panel_index + 1}/{len(axes)}: Z={metallicity}, wind_model={wind_model}')
        # if panel_index > 0:
        #     continue
        row, column = divmod(panel_index, 2)
        ax.axhspan(7.0, 11.0, color="lightgray", alpha=0.35, zorder=0)

        reference_data = next(iter(model_data.values()))
        _plot_pre_sn_masses(ax, reference_data, colors[:3])
        _plot_remnant_masses(ax, model_data, REMNANT_COLORS)

        ax.set_xlim(0.0, 100.0)
        ax.set_ylim(0.0, y_max)
        ax.set_xlabel(r"$M_{\rm ZAMS}\,(M_\odot)$")
        ax.set_ylabel(r"$M_{\rm pre-SN}\,(M_\odot)$")

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            bottom=True,
            left=True,
            right=True,
            labeltop=False,
            labelbottom=True,
            labelleft=True,
            labelright=False,
        )

        ax.text(
            0.97,
            0.97,
            f"$Z={metallicity:g}$\n{wind_labels[wind_model]} wind",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )

        remnant_handles = [
            Line2D(
                [0], [0], color=color, marker="o", markersize=2.0,
                markerfacecolor="none", linestyle="None",
                label=REMNANT_LABELS[model_name],
            )
            for color, model_name in zip(REMNANT_COLORS, MODEL_CONFIGS)
        ]
        progenitor_handles = [
            Line2D([0], [0], color=color, linewidth=3.0, label=label)
            for color, label in zip(colors[:3], progenitor_labels)
        ]
        ax.legend(
            handles=progenitor_handles + remnant_handles,
            loc="upper left",
            frameon=True,
        )

    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    test_stars()
