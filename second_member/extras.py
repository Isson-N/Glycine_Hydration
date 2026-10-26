"""Этап 7 (дополнительно): энтропия Шеннона счётчика воды и спектральная плотность мощности.

Запуск из папки проекта:
    python second_member/extras.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch.bootstrap import MovingBlockBootstrap
from scipy.signal import welch

from autocorrelation import correlation_time, load_working_table, segment_values
from bootstrap import block_length, bootstrap_p, min_blocks
from segments import LARGE, SMALL, build_variants


root = Path(__file__).resolve().parent.parent
file = root / "data" / "gly_matrix_all_reps.csv"
out_dir = root / "second_member" / "C" / "stage7"
frame_step_ns = 0.05
fs = 1 / frame_step_ns  # 20 отсчётов в нс, частота Найквиста fs/2 = 10 циклов/нс
replicates = 2000
window_lengths = [50, 100, 200, 500, 1000, 2000]  # плюс полная длина участка


def shannon_entropy(values):
    """H = −Σ p_k log2 p_k по долям значений счётчика, в битах."""
    p = np.unique(values, return_counts=True)[1] / len(values)
    return -(p * np.log2(p)).sum()


def miller_madow(values):
    """Поправка на конечную выборку: + (K − 1) / (2N ln 2), K — число встреченных значений. Рассчитана на независимые кадры."""
    return shannon_entropy(values) + (len(np.unique(values)) - 1) / (2 * len(values) * np.log(2))


def entropy_comparison(data, segments, rep):
    site = segment_values(data, rep, *segments[rep][SMALL], "n_gly_wat")
    plateau = segment_values(data, rep, *segments[rep][LARGE], "n_gly_wat")
    # Та же длина блока, что в bootstrap.py: по n и cn обоих участков
    length = block_length(site, plateau, *(segment_values(data, rep, *segments[rep][k], "cn_gly_wat") for k in (SMALL, LARGE)))

    rng = np.random.default_rng(1)
    boot_s = MovingBlockBootstrap(length, site, seed=rng)
    boot_p = MovingBlockBootstrap(length, plateau, seed=rng)
    draws = [shannon_entropy(p[0][0]) - shannon_entropy(s[0][0])
             for s, p in zip(boot_s.bootstrap(replicates), boot_p.bootstrap(replicates))]
    low, high = np.percentile(draws, [2.5, 97.5])

    return {
        "Реализация": rep,
        "Энтропия, малые расстояния, бит": shannon_entropy(site),
        "Энтропия, большие расстояния, бит": shannon_entropy(plateau),
        "С поправкой Миллера–Мэдоу, малые, бит": miller_madow(site),
        "С поправкой Миллера–Мэдоу, большие, бит": miller_madow(plateau),
        "Разность (большие − малые), бит": shannon_entropy(plateau) - shannon_entropy(site),
        "Длина блока, кадров": length,
        "95% ДИ: нижняя": low,
        "95% ДИ: верхняя": high,
        "p бутстрэпа": bootstrap_p(draws),
        "Блоков на участок": min(len(site), len(plateau)) // length,
        "Различие больше неопределённости": ("мало блоков" if min(len(site), len(plateau)) // length < min_blocks
                                            else "да" if low > 0 or high < 0 else "нет"),
    }


def entropy_by_length(data, segments, rep):
    """Энтропия по всем непересекающимся окнам длины L внутри участка: среднее и разброс."""
    rows = []
    for name, (start, end) in segments[rep].items():
        values = segment_values(data, rep, start, end, "n_gly_wat")
        for L in [L for L in window_lengths if L < len(values)] + [len(values)]:
            h = [shannon_entropy(values[i:i + L]) for i in range(0, len(values) - L + 1, L)]
            rows.append({"Реализация": rep, "Участок": name, "Длина окна, кадров": L,
                         "Длина окна, нс": L * frame_step_ns, "Число окон": len(h),
                         "Средняя энтропия, бит": np.mean(h), "SD энтропии, бит": np.std(h, ddof=1) if len(h) > 1 else np.nan})
    return pd.DataFrame(rows)


def plot_entropy_by_length(table, name_plt):
    fig, ax = plt.subplots(figsize=(9, 5))
    for (rep, name), rows in table.groupby(["Реализация", "Участок"], sort=False):
        ax.errorbar(rows["Длина окна, нс"], rows["Средняя энтропия, бит"], yerr=rows["SD энтропии, бит"],
                    marker="o", capsize=3, label=f"{rep}: {name}")
    ax.set_xscale("log")
    ax.set_xlabel("Длина окна, нс (логарифмическая шкала)")
    ax.set_ylabel("Энтропия Шеннона n_gly_wat, бит")
    ax.set_title("Чувствительность энтропии к длине участка: среднее ± SD по окнам")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(name_plt, dpi=200)
    plt.close(fig)


def power_spectra(data, segments_rep2):
    """PSD n_gly_wat методом Уэлча: окно 2048 кадров (102 нс), перекрытие 50% — 4 куска на участок;
    в каждом куске вычитается линейный тренд. Масштабы длиннее 102 нс не видны."""
    spectra, taus = {}, {}
    for name, (start, end) in segments_rep2.items():
        values = segment_values(data, "rep2", start, end, "n_gly_wat")
        spectra[name] = welch(values, fs=fs, nperseg=2048, detrend="linear")
        taus[name] = correlation_time(values, frame_step_ns)[2]
    return spectra, taus


def plot_spectra(spectra, taus, name_plt):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, (freq, psd) in spectra.items():
        line, = ax.loglog(freq[1:], psd[1:], linewidth=1, label=name)
        if taus.get(name):
            # Для одной экспоненты в ACF спектр лоренцев с изломом на 1/(2π·tau_c); здесь ACF сложнее — только ориентир
            ax.axvline(1 / (2 * np.pi * taus[name]), color=line.get_color(), linestyle="--", linewidth=0.8)
    ax.axvline(fs / 2, color="black", linestyle=":", label="частота Найквиста, 10 циклов/нс")
    ax.set_xlabel("Частота, циклов/нс")
    ax.set_ylabel("Спектральная плотность мощности, молекул² · нс")
    ax.set_title("Спектральная плотность мощности n_gly_wat, rep2 (метод Уэлча). Пунктир — 1/(2π·tau_c)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(name_plt, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_working_table(file, frame_step_ns)
    variants = build_variants(data)
    segments = variants["C"]

    entropy = pd.DataFrame([entropy_comparison(data, segments, rep) for rep in segments])
    by_length = pd.concat([entropy_by_length(data, segments, rep) for rep in segments], ignore_index=True)
    plot_entropy_by_length(by_length, out_dir / "Энтропия от длины окна.png")

    # PSD требует квазистационарного участка: по этапу 4 таким оказалось только окно 2 плато (C2);
    # сайт и окно 1 показаны для сравнения, хотя стационарность там отвергнута
    rep2_segments = {
        "Малые расстояния (сайт)": segments["rep2"][SMALL],
        "Большие расстояния, окно 1": segments["rep2"][LARGE],
        "Большие расстояния, окно 2 (стационарно)": variants["C2"]["rep2"][LARGE],
    }
    spectra, taus = power_spectra(data, rep2_segments)
    plot_spectra(spectra, taus, out_dir / "Спектральная плотность rep2.png")

    with pd.ExcelWriter(out_dir / "Дополнительно.xlsx") as writer:
        entropy.to_excel(writer, sheet_name="Энтропия", index=False)
        by_length.to_excel(writer, sheet_name="Энтропия от длины", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(entropy.round(3).to_string(index=False))
        print(by_length.round(3).to_string(index=False))
