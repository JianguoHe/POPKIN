"""Electromagnetic transient observables for compact-object populations.

This module currently contains lightweight candidate-selection helpers. More
physical transient models can later add ejecta masses, light curves, rates, and
survey detectability.
"""

import numpy as np


COMPACT_TRANSIENT_TYPES = ("HeWD", "COWD", "ONeWD", "NS", "BH")


def select_event_rows(data, events):
    """Select rows whose ``event`` field matches one or more event labels."""
    if isinstance(events, str):
        events = (events,)
    return np.isin(data["event"], events)


def classify_compact_merger(type1, type2):
    """Classify compact-object merger pairs.

    Returns labels ``BNS``, ``NSBH``, ``BBH``, ``WDWD``, ``WDNSBH``, or ``other``.
    """
    type1_arr, type2_arr = np.broadcast_arrays(np.asarray(type1), np.asarray(type2))
    labels = np.full(type1_arr.shape, "other", dtype="U8")

    ns1 = type1_arr == "NS"
    ns2 = type2_arr == "NS"
    bh1 = type1_arr == "BH"
    bh2 = type2_arr == "BH"
    wd_types = ("HeWD", "COWD", "ONeWD")
    wd1 = np.isin(type1_arr, wd_types)
    wd2 = np.isin(type2_arr, wd_types)

    labels[ns1 & ns2] = "BNS"
    labels[(ns1 & bh2) | (bh1 & ns2)] = "NSBH"
    labels[bh1 & bh2] = "BBH"
    labels[wd1 & wd2] = "WDWD"
    labels[(wd1 & (ns2 | bh2)) | ((ns1 | bh1) & wd2)] = "WDNSBH"

    return labels.item() if labels.shape == () else labels


def select_compact_merger_candidates(data, event="merge"):
    """Select compact-object merger candidates from binary population rows."""
    event_mask = select_event_rows(data, event)
    compact1 = np.isin(data["type1"], COMPACT_TRANSIENT_TYPES)
    compact2 = np.isin(data["type2"], COMPACT_TRANSIENT_TYPES)
    return event_mask & compact1 & compact2


__all__ = [
    "COMPACT_TRANSIENT_TYPES",
    "select_event_rows",
    "classify_compact_merger",
    "select_compact_merger_candidates",
]
