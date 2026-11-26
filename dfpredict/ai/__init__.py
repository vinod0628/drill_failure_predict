"""Decision support ("AI") for GeoViewer - runs entirely offline, on this computer.

Everything here is a local algorithm (statistics, search and optimisation on the project's own
data); nothing is sent anywhere and no internet connection or online model is used.

The functions only *suggest*: they return findings, rankings and designs with plain-language
explanations. They never change the project database or a data file - the windows let the user
look at a suggestion and accept it (or not).

    reserve_advisor   life of the reserve (forecast), a check of every mined cut, the best
                      From / To levels for a new cut
    closure_doctor    when a survey traverse does not close: which single reading mistake
                      would explain the misclosure
    route_planner     a new drive (decline, access) from a station to a target, within a gradient
                      limit and away from old workings, other drives and (optionally) the ore

No Qt here: the windows are in geoviewer/ui (ai_widgets, reserve_advisor_dialog, closure_dialog,
route_dialog).
"""

OK, WARN, BAD, INFO = "ok", "warn", "bad", "info"     # levels of a finding (green / amber / red / blue)
