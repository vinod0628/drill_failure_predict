"""Geological roles of surface / string layers (after 3D MPM Studio's ``mpm_layers.ROLES``).

A role is a saved display setting of a ``SurfaceLayer`` / ``StringLayer`` (``layer.role``). The
drilling-risk tools use it to find the workspace's geological model:

* ``horizon``      a stratigraphic contact (the base of one formation / top of the next); the name
                   may carry the units above and below, e.g. ``horizon_H2_U2-U3``;
* ``unconformity`` an unconformity surface (also a contact; drawn grey);
* ``fault``        a fault surface (drawn red);
* ``fold_axis``    a fold axial surface (crushed zones in the hinge);
* ``formation``    a closed formation solid (the formation is the layer / object name);
* ``topography``   the ground surface;
* ``karst``        a mapped cavity / karst zone (solid or surface).

GeoViewer and 3D MPM Studio open the same workspace; they ignore roles they do not know.
"""

from __future__ import annotations

from ..i18n import N_

ROLES = {
    "": N_("(none)"),
    "horizon": N_("Horizon / contact"),
    "unconformity": N_("Unconformity"),
    "fault": N_("Fault"),
    "fold_axis": N_("Fold axial surface"),
    "formation": N_("Formation solid"),
    "topography": N_("Topography"),
    "karst": N_("Karst / cavity zone"),
    "orebody": N_("Orebody"),
    "mapping": N_("Pit / drive mapping"),
    "mined": N_("Mined-out"),
}

# Roles that make up the geological model used ahead of the bit.
MODEL_ROLES = ("horizon", "unconformity", "fault", "fold_axis", "formation", "topography", "karst")

# Default display colours by role (spec 8.1: faults red, unconformities grey).
ROLE_COLORS = {
    "fault": (0.90, 0.20, 0.20),
    "unconformity": (0.62, 0.62, 0.62),
    "fold_axis": (0.85, 0.55, 0.95),
    "topography": (0.55, 0.62, 0.45),
    "karst": (0.20, 0.55, 0.95),
}


def guess_role(name: str) -> str:
    """A role from a layer / file name ("" when nothing fits)."""
    import re

    n = name.lower()
    if re.match(r"^(f|flt)[_\- ]?\d+", n) or "fault" in n:
        return "fault"
    if "unconf" in n:
        return "unconformity"
    if "horizon" in n or "contact" in n or n.startswith(("hz", "h_")):
        return "horizon"
    if "topo" in n or "dem" in n:
        return "topography"
    if "karst" in n or "cav" in n:
        return "karst"
    if "axial" in n or "hinge" in n:
        return "fold_axis"
    return ""
