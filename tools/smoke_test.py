"""Headless smoke test of the installed environment (no window is shown).

Checks: every runtime package imports, VTK can render with OpenGL off-screen,
Qt widgets work (offscreen platform), and - when demo_data\ is present -
the demo files load into layers.

Exit code 0 = OK, 1 = failure. Run with the .venv interpreter.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DEMO = ROOT / "demo_data" / "sb3_workspace"
DEMO_FILES = ["model/horizon_H2_U2-U3.dtm", "faults/F4.dtm", "block_model/original_bm.gvbm"]


def step(name, fn):
    t = time.perf_counter()
    try:
        info = fn()
    except Exception:  # noqa: BLE001
        print(f"  FAIL  {name}")
        traceback.print_exc()
        return False
    print(f"  ok    {name}  ({time.perf_counter() - t:.1f}s){'  ' + info if info else ''}")
    return True


def imports():
    import access_parser  # noqa: F401
    import lightgbm  # noqa: F401
    import matplotlib  # noqa: F401
    import scipy  # noqa: F401
    import sklearn  # noqa: F401
    import ezdxf  # noqa: F401
    import numpy
    import pandas  # noqa: F401
    import pyarrow  # noqa: F401
    import pyvista
    import pyvistaqt  # noqa: F401
    import qtawesome  # noqa: F401
    import vtk
    from PySide6 import QtCore

    return (f"numpy {numpy.__version__}, vtk {vtk.vtkVersion.GetVTKVersion()}, "
            f"pyvista {pyvista.__version__}, Qt {QtCore.qVersion()}")


def opengl():
    import vtk

    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    ren = vtk.vtkRenderer()
    rw.AddRenderer(ren)
    src = vtk.vtkCubeSource()
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(src.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    ren.AddActor(a)
    rw.Render()
    caps = rw.ReportCapabilities()
    line = next((x for x in caps.splitlines() if "OpenGL version" in x), "OpenGL version unknown")
    rw.Finalize()
    return line.strip()


def qt_widgets():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from dfpredict.ui.icons import icon, text_icon
    from dfpredict.ui.theme import apply_dark_theme

    apply_dark_theme(app)
    assert not icon("cube-outline").isNull()
    assert not text_icon("XY").isNull()
    return "theme + icons"


def demo_data():
    if not DEMO.exists():
        return "skipped (no demo_data folder)"
    from dfpredict.scene.loaders import load_paths

    files = [DEMO / f for f in DEMO_FILES if (DEMO / f).exists()]
    layers, errors = load_paths(files)
    if errors:
        raise RuntimeError("; ".join(errors))
    return f"{len(layers)} layers from {len(files)} files"


def risk_database():
    """A new project database gets the risk tables and the predefined reference data."""
    import tempfile

    from dfpredict.io.database import Target
    from dfpredict.risk.store import RiskStore

    with tempfile.TemporaryDirectory() as d:
        st = RiskStore(Target("sqlite", path=Path(d) / "geodatabase.sqlite"))
        st.prepare()
        data = st.load_all()
        n = {k: len(data[k]) for k in ("rock", "rock_code", "measure", "rig")}
    assert all(n.values()), n
    return ", ".join(f"{k} {v}" for k, v in n.items())


def main() -> int:
    print("Smoke test")
    ok = all([
        step("imports", imports),
        step("OpenGL off-screen render", opengl),
        step("Qt widgets / theme / icons", qt_widgets),
        step("demo data", demo_data),
        step("risk database + predefined data", risk_database),
    ])
    print("SMOKE TEST: OK" if ok else "SMOKE TEST: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
