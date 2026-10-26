"""Итоговые участки (вариант C): rep2 — сайт и окно 1 плато; rep0 — до и после перехода.

Запуск из папки проекта:
    python second_member/regions_final.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from segments import LARGE, SMALL, build_variants

root = Path(__file__).resolve().parent.parent
file = root / "data" / "gly_matrix_all_reps.csv"
output = root / "second_member" / "figures" / "regions_final_C.png"
frame_step_ns = 0.05

data = pd.read_csv(file, usecols=["rep", "frame", "n_gly_wat", "cn_gly_wat", "dist_ca_gly_com_bs"])
segments = build_variants(data)["C"]

colors = {SMALL: "tab:blue", LARGE: "tab:orange"}


def draw(ax, rep, xlim=None):
    selected = data.loc[data["rep"] == rep].set_index("frame")
    if xlim:
        selected = selected.loc[xlim[0] / frame_step_ns:xlim[1] / frame_step_ns]
    ax.plot(selected.index * frame_step_ns, selected["dist_ca_gly_com_bs"], linewidth=0.5, color="0.25")
    ax.axhline(5, linestyle="--", linewidth=0.7, color="black")
    ax.axhline(10, linestyle=":", linewidth=0.7, color="black")

    lines = []
    for name, (start, end) in segments[rep].items():
        ax.axvspan(start * frame_step_ns, end * frame_step_ns, color=colors[name], alpha=0.3)
        section = data.loc[(data["rep"] == rep) & data["frame"].between(start, end)]
        lines.append(f"{name}: кадры {start}–{end}, {start * frame_step_ns:.2f}–{end * frame_step_ns:.2f} нс, "
                     f"N = {len(section)}, n = {section['n_gly_wat'].mean():.2f}, cn = {section['cn_gly_wat'].mean():.2f}")
    ax.set_ylabel("Расстояние, Å")
    if xlim:
        ax.set_xlim(*xlim)
    return lines


fig, axes = plt.subplots(3, 1, figsize=(14, 11))

lines = draw(axes[0], "rep2")
axes[0].set_title("rep2 (основная): сайт и первое окно плато около 8 Å", loc="left")
axes[0].text(0.01, 0.97, "\n".join(lines), transform=axes[0].transAxes, va="top", fontsize=8,
             bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.8"))

lines = draw(axes[1], "rep0")
axes[1].set_title("rep0 (контроль): вся траектория; участки по 149 кадров у перехода (около 410–423 нс)", loc="left")
axes[1].text(0.01, 0.97, "\n".join(lines), transform=axes[1].transAxes, va="top", fontsize=8,
             bbox=dict(facecolor="white", alpha=0.9, edgecolor="0.8"))

draw(axes[2], "rep0", xlim=(400, 425))
axes[2].set_title("rep0, увеличено 400–425 нс: 7,4 нс в сайте перед уходом и 7,4 нс остановки у 8 Å", loc="left")

for ax in axes:
    ax.set_xlabel("Время, нс")
fig.suptitle("Выбранные участки. Пунктир 5 Å — граница сайта, точки 10 Å — начало выхода")
fig.tight_layout()
fig.savefig(output, dpi=130)
print(output)
