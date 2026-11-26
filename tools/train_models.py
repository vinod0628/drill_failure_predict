"""Train the drilling risk models of a workspace from its project database (spec 7.4: retrain monthly or
every 10 new holes).

    python tools/train_models.py <workspace folder> [--no-validate]

Uses the SQLite project database of the workspace (geodatabase.sqlite). The models are saved in
<workspace>/drillrisk/models with a version and become the current models.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dfpredict.io.database import PROJECT_DB, Target  # noqa: E402
from dfpredict.risk import models  # noqa: E402
from dfpredict.risk.store import RiskStore  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace")
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args(argv)
    ws = Path(a.workspace)
    store = RiskStore(Target("sqlite", path=ws / PROJECT_DB))
    store.prepare()
    data = store.load_all()
    t = time.time()
    ms = models.train(data, validate=not a.no_validate, progress=lambda p, s: print(f"  {p:3d}% {s}", end="\r"))
    print()
    p = models.save(ms, ws)
    print("saved", p, f"({time.time() - t:.0f} s)")
    print("trained on:", json.dumps(ms.trained_on))
    print("validation:", json.dumps({k: v for k, v in ms.validation.items() if k != "auc_hole_by_area"},
                                    default=lambda o: round(o, 3) if isinstance(o, float) else str(o)))
    print("thresholds:", ms.thresholds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
