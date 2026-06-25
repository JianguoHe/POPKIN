"""Observable calculators for POPKIN post-processing and survey predictions."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from popkin.observables.electromagnetic_transients import (
        classify_compact_merger,
        select_compact_merger_candidates,
        select_event_rows,
    )
    from popkin.observables.gravitational_waves import (
        calc_gw_snr,
        gw_snr_mask,
        select_gw_sources,
    )
    from popkin.observables.isolated_bh_accretion import (
        radiative_efficiency_xie_yuan_2012,
        summarize_isolated_bh_accretion,
    )
    from popkin.observables.microlensing import (
        add_microlensing_observables,
        estimate_bh_lens_fraction_by_timescale,
    )
    from popkin.observables.survey_selection import (
        flux_limited_mask,
        flux_to_luminosity,
        luminosity_to_flux,
        threshold_mask,
    )
    from popkin.observables.xray_binaries import (
        accretion_luminosity,
        capped_accretion_luminosity,
        eddington_luminosity,
        select_xray_binary_candidates,
        xray_flux,
    )

__all__ = [
    "calc_gw_snr",
    "gw_snr_mask",
    "select_gw_sources",
    "summarize_isolated_bh_accretion",
    "radiative_efficiency_xie_yuan_2012",
    "add_microlensing_observables",
    "estimate_bh_lens_fraction_by_timescale",
    "luminosity_to_flux",
    "flux_to_luminosity",
    "flux_limited_mask",
    "threshold_mask",
    "eddington_luminosity",
    "accretion_luminosity",
    "capped_accretion_luminosity",
    "xray_flux",
    "select_xray_binary_candidates",
    "select_event_rows",
    "classify_compact_merger",
    "select_compact_merger_candidates",
]


def __getattr__(name):
    if name in {"calc_gw_snr", "gw_snr_mask", "select_gw_sources"}:
        from popkin.observables.gravitational_waves import (
            calc_gw_snr,
            gw_snr_mask,
            select_gw_sources,
        )

        exports = {
            "calc_gw_snr": calc_gw_snr,
            "gw_snr_mask": gw_snr_mask,
            "select_gw_sources": select_gw_sources,
        }
        return exports[name]

    if name in {
        "summarize_isolated_bh_accretion",
        "radiative_efficiency_xie_yuan_2012",
    }:
        from popkin.observables.isolated_bh_accretion import (
            radiative_efficiency_xie_yuan_2012,
            summarize_isolated_bh_accretion,
        )

        exports = {
            "summarize_isolated_bh_accretion": summarize_isolated_bh_accretion,
            "radiative_efficiency_xie_yuan_2012": radiative_efficiency_xie_yuan_2012,
        }
        return exports[name]

    if name in {
        "add_microlensing_observables",
        "estimate_bh_lens_fraction_by_timescale",
    }:
        from popkin.observables.microlensing import (
            add_microlensing_observables,
            estimate_bh_lens_fraction_by_timescale,
        )

        exports = {
            "add_microlensing_observables": add_microlensing_observables,
            "estimate_bh_lens_fraction_by_timescale": estimate_bh_lens_fraction_by_timescale,
        }
        return exports[name]

    if name in {
        "luminosity_to_flux",
        "flux_to_luminosity",
        "flux_limited_mask",
        "threshold_mask",
    }:
        from popkin.observables.survey_selection import (
            flux_limited_mask,
            flux_to_luminosity,
            luminosity_to_flux,
            threshold_mask,
        )

        exports = {
            "luminosity_to_flux": luminosity_to_flux,
            "flux_to_luminosity": flux_to_luminosity,
            "flux_limited_mask": flux_limited_mask,
            "threshold_mask": threshold_mask,
        }
        return exports[name]

    if name in {
        "eddington_luminosity",
        "accretion_luminosity",
        "capped_accretion_luminosity",
        "xray_flux",
        "select_xray_binary_candidates",
    }:
        from popkin.observables.xray_binaries import (
            accretion_luminosity,
            capped_accretion_luminosity,
            eddington_luminosity,
            select_xray_binary_candidates,
            xray_flux,
        )

        exports = {
            "eddington_luminosity": eddington_luminosity,
            "accretion_luminosity": accretion_luminosity,
            "capped_accretion_luminosity": capped_accretion_luminosity,
            "xray_flux": xray_flux,
            "select_xray_binary_candidates": select_xray_binary_candidates,
        }
        return exports[name]

    if name in {
        "select_event_rows",
        "classify_compact_merger",
        "select_compact_merger_candidates",
    }:
        from popkin.observables.electromagnetic_transients import (
            classify_compact_merger,
            select_compact_merger_candidates,
            select_event_rows,
        )

        exports = {
            "select_event_rows": select_event_rows,
            "classify_compact_merger": classify_compact_merger,
            "select_compact_merger_candidates": select_compact_merger_candidates,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
