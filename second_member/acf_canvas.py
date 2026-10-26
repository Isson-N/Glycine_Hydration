"""Все АКФ на одном полотне: участки rep2, которые сравниваются (сайт и окно 1 плато).

Запуск из папки проекта:
    python second_member/acf_canvas.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from autocorrelation import correlation_time, file, frame_step_ns, load_working_table, segment_values, threshold
from segments import LARGE, SMALL, build_variants

output = Path(__file__).resolve().parent / "figures" / "acf_all_rep2.png"

data = load_working_table(file, frame_step_ns)
variants = build_variants(data)
segments = {
    "Область малых расстояний": variants["C"]["rep2"][SMALL],
    "Область больших расстояний": variants["C"]["rep2"][LARGE],
}
colors = ["tab:blue", "tab:orange"]

fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
for row, variable in enumerate(["n_gly_wat", "cn_gly_wat"]):
    for (name, (start, end)), color in zip(segments.items(), colors):
        r, lag, tau = correlation_time(segment_values(data, "rep2", start, end, variable), frame_step_ns)
        lags_ns = np.arange(len(r)) * frame_step_ns
        for ax in axes[row]:
            ax.plot(lags_ns, r, color=color, linewidth=1, label=f"{name}: τc = {tau:.2f} нс")
            ax.plot(tau, r[lag], "o", color=color)
    for col, ax in enumerate(axes[row]):
        ax.axhline(threshold, color="black", linestyle=":", label="1/e ≈ 0,368")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xlabel("Лаг, нс")
        ax.set_ylabel("Нормированная автокорреляция")
        ax.set_title(f"{variable}: " + ("начальные лаги" if col == 0 else "весь диапазон до N/4"))
        if col == 0:
            ax.set_xlim(0, 5)
            ax.legend(fontsize=9)
fig.suptitle("Автокорреляционные функции числа молекул воды, rep2 (точки — первое пересечение 1/e)")
fig.tight_layout()
fig.savefig(output, dpi=200)
print(output)
