"""Submission assets: the figures of each paper as separate files in manuscript order (Fig1, Fig2, ...), and the
graphical abstract for Minerals.

    .venv\\Scripts\\python.exe tools\\make_submission_assets.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "docs" / "publications"
SRC = PUB / "src"
IMG = re.compile(r"\[\[IMG:([^|\]]+)\|\d+\]\]")


def export_figures(paper: str, folder: Path, fmt: str) -> list[str]:
    from PIL import Image

    html = (SRC / f"{paper}.html").read_text(encoding="utf-8")
    caps = re.findall(r'<p class="caption">\s*(Fig(?:ure|\.)?\s*\d+\.?.*?)</p>', html, re.S)
    out = folder
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    lines = []
    for i, rel in enumerate(IMG.findall(html), 1):
        im = Image.open(PUB / rel).convert("RGB")
        name = f"Fig{i}.{fmt}"
        if fmt == "tif":
            im.save(out / name, compression="tiff_lzw", dpi=(300, 300))
        else:
            im.save(out / name, dpi=(300, 300))
        cap = re.sub(r"<[^>]+>", "", caps[i - 1]).strip() if i - 1 < len(caps) else ""
        lines.append(f"{name}\t{im.size[0]} x {im.size[1]} px\t(source: {rel})\n    {cap}")
    (out / "Figure_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def graphical_abstract(path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    F = PUB / "figures"
    fig = plt.figure(figsize=(11, 5.6), dpi=200, facecolor="white")
    fig.text(0.5, 0.955, "Uncertainty-aware drillhole planning and evidence-ranked countermeasures",
             ha="center", va="center", fontsize=15, weight="bold", family="DejaVu Sans", color="#1f3b5c")
    panels = [
        (F / "view3d_all.png", "1  Uncertain geological model\n    + trial paths of the hole", (0.015, 0.20, 0.31, 0.64), None),
        (F / "risklog_DH-2026-001.png", "2  Danger by depth for the ongoing hole\n    (zones ±u, levels, reasons)",
         (0.345, 0.20, 0.31, 0.64), None),
        (F / "fig_planmaps.png", "3  Safer design for a planned hole\n    + measures ranked by past evidence",
         (0.675, 0.20, 0.31, 0.64), "left"),
    ]
    for img, title, box, crop in panels:
        ax = fig.add_axes(box)
        a = mpimg.imread(img)
        if crop == "left":
            a = a[:, : a.shape[1] // 2]
        ax.imshow(a)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#9aa9b8")
        fig.text(box[0] + 0.005, box[1] + box[3] + 0.015, title, fontsize=10.5, weight="bold", va="bottom",
                 family="DejaVu Sans", color="#1f3b5c")
    for x in (0.329, 0.659):
        fig.patches.append(FancyArrowPatch((x - 0.006, 0.52), (x + 0.013, 0.52), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=22, color="#c0392b", lw=2))
    fig.text(0.5, 0.115, "Ongoing hole: red zone 189–246 m (±11 m) at fault F4 / cavernous dolomite  →  casing at "
             "173 m (solved 16 of 20 similar cases)", ha="center", fontsize=10, family="DejaVu Sans")
    fig.text(0.5, 0.065, "Contact tie: ±12.7 m → ±5.2 m     ·     Planned hole on fault F2: P(reach) 0.865 → 0.966 "
             "after re-orientation     ·     Evidence vs truth: Spearman ρ = 0.83", ha="center", fontsize=10,
             family="DejaVu Sans")
    fig.text(0.5, 0.02, "Synthetic, process-based drilling benchmark on the SB3 demonstration model",
             ha="center", fontsize=8.5, style="italic", color="#555", family="DejaVu Sans")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    p1 = PUB / "paper1_MathGeosci"
    p2 = PUB / "paper2_Minerals"
    a = export_figures("paper1", p1 / "04_Figures", "tif")
    b = export_figures("paper2", p2 / "03_Figures", "tif")
    graphical_abstract(p2 / "04_Graphical_abstract.png")
    print(len(a), "figures for paper 1;", len(b), "figures for paper 2; graphical abstract written")


if __name__ == "__main__":
    main()
