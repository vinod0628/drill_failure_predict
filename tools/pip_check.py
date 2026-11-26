"""Verify the .venv matches requirements-offline.txt exactly and has no broken dependencies.

Exit code 0 = OK, 1 = problems found. Run with the .venv interpreter.
"""

from __future__ import annotations

import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-offline.txt"


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    problems: list[str] = []
    wanted: dict[str, str] = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, ver = line.split("==", 1)
            wanted[norm(name)] = ver.strip()

    installed = {norm(d.metadata["Name"]): d.version for d in metadata.distributions()}
    for name, ver in sorted(wanted.items()):
        have = installed.get(name)
        if have is None:
            problems.append(f"missing    {name}=={ver}")
        elif have != ver:
            problems.append(f"version    {name}: installed {have}, expected {ver}")

    extra = sorted(set(installed) - set(wanted) - {"pip", "geoviewer"})
    r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if r.returncode != 0:
        problems.append("pip check:\n  " + r.stdout.strip().replace("\n", "\n  "))

    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    print(f"{len(wanted)} locked packages, {len(installed)} installed")
    if extra:
        print("note: installed but not in the lock file:", ", ".join(extra))
    if problems:
        print("\n".join(problems))
        print("PIP CHECK: FAILED")
        return 1
    print("PIP CHECK: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
