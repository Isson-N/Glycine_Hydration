"""График вариантов участков: строка на вариант, слева rep2, справа rep0.

Запуск из папки проекта:
    python second_member/regions_variants.py [--no-check]
С ключом --no-check окно 2 (штриховка в варианте C) не рисуется → regions_variants_clean.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from segments import SMALL, VARIANT_TITLES, build_variants

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "gly_matrix_all_reps.csv"
SHOW_CHECK = "--no-check" not in sys.argv
OUT = ROOT / "second_member" / "figures" / ("regions_variants.png" if SHOW_CHECK else "regions_variants_clean.png")
DT_NS = 0.05

data = pd.read_csv(CSV, usecols=["rep", "frame", "n_gly_wat", "cn_gly_wat", "dist_ca_gly_com_bs"])
variants = build_variants(data)

fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex="col", sharey=True)
for col, rep in enumerate(["rep2", "rep0"]):
    r = data[data["rep"] == rep].set_index("frame")
    for row, variant in enumerate(["A", "B", "C"]):
        by_rep = dict(variants[variant])
        if SHOW_CHECK and variant == "C" and rep == "rep2":
            # Окно 2 — только для объяснения выбора: показано штриховкой как проверка
            by_rep = {rep: {**by_rep[rep], "Большие расстояния, окно 2 (проверка)": variants["C2"][rep]["Большие расстояния"]}}
        ax = axes[row, col]
        ax.plot(r.index * DT_NS, r["dist_ca_gly_com_bs"], lw=0.4, color="0.3")
        ax.axhline(5, ls="--", lw=0.7, color="k")
        ax.axhline(10, ls=":", lw=0.7, color="k")
        lines = []
        large_count = 0
        for name, (a, b) in by_rep[rep].items():
            seg = r.loc[a:b]
            if name == SMALL:
                color, alpha = "tab:blue", 0.25
            else:
                color, alpha = "tab:orange", 0.25 + 0.15 * large_count
                large_count += 1
            hatch = "//" if "проверка" in name else None
            ax.axvspan(a * DT_NS, b * DT_NS, facecolor="0.6" if hatch else color, alpha=0.3 if hatch else alpha,
                       hatch=hatch, edgecolor="0.4" if hatch else None)
            lines.append(f"{name}: {a * DT_NS:.2f}–{b * DT_NS:.2f} нс, {len(seg)} кадров, "
                         f"n = {seg['n_gly_wat'].mean():.2f}, cn = {seg['cn_gly_wat'].mean():.2f}")
        ax.set_title(f"{rep} — {VARIANT_TITLES[variant]}", loc="left", fontsize=10)
        ax.text(0.01, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=7.5,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"))
        if col == 0:
            ax.set_ylabel("Расстояние до центра сайта, Å")
    axes[-1, col].set_xlabel("Время, нс")

fig.suptitle("Варианты участков: синий — область малых расстояний (сайт), "
             "оранжевый — область около 8 Å; пунктир 5 Å и 10 Å — пороги правила фаз")
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(OUT)
