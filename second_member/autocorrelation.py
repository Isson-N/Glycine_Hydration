import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from statsmodels.tsa.stattools import acf, adfuller, kpss

from segments import build_variants, VARIANT_TITLES


root = Path(__file__).resolve().parent.parent
file = root / "data" / "gly_matrix_all_reps.csv"
frame_step_ns = 0.05
threshold = np.exp(-1)
variables = ["n_gly_wat", "cn_gly_wat"]


def load_working_table(file_path, frame_step_ns):
    columns = ["rep", "frame", "n_gly_wat", "cn_gly_wat", "dist_ca_gly_com_bs"]
    data = pd.read_csv(file_path, usecols=columns)
    data = data.loc[data["rep"].isin(["rep0", "rep2"])].sort_values(["rep", "frame"])
    data["t_ns"] = data["frame"] * frame_step_ns
    return data.reset_index(drop=True)


def segment_values(data, rep, start, end, variable):
    section = data.loc[(data["rep"] == rep) & data["frame"].between(start, end)]

    # ACF имеет смысл только для непрерывного ряда с постоянным шагом
    if section.empty or (section["frame"].diff().dropna() != 1).any():
        raise ValueError(f"{rep} {start}–{end}: кадры идут не подряд")

    return section[variable].to_numpy(dtype=float)


def check_trend(values, frame_step_ns):
    t_ns = np.arange(len(values)) * frame_step_ns
    slope = np.polyfit(t_ns, values, 1)[0]
    trend_sd = slope * t_ns[-1] / values.std(ddof=1)
    half = len(values) // 2

    with warnings.catch_warnings():
        # p-value KPSS табличное: за пределами 0.01–0.1 возвращается граница
        warnings.simplefilter("ignore")
        _, kpss_p, kpss_lags, _ = kpss(values, regression="c", nlags="auto")
    _, adf_p, adf_lags, *_ = adfuller(values, autolag="AIC", result_object=False)

    # Заранее выбранное правило: стационарность вокруг среднего не отвергнута и тренд < 0.5 SD
    suitable = kpss_p > 0.05 and abs(trend_sd) < 0.5

    return suitable, {
        "Наклон тренда, ед./нс": slope,
        "Изменение по тренду за участок, в SD": trend_sd,
        "Среднее 1-й половины": values[:half].mean(),
        "Среднее 2-й половины": values[half:].mean(),
        "KPSS p (H0: стационарен около среднего)": format_p(kpss_p),
        "KPSS, число лагов": kpss_lags,
        "ADF p (H0: единичный корень)": format_p(adf_p),
        "ADF, число лагов": adf_lags,
        "Пригодность для стационарной ACF": "да" if suitable else "ограничена",
    }


def format_p(p):
    if p <= 0.01:
        return "<0.01"
    if p >= 0.1:
        return ">0.1"
    return round(p, 3)


def correlation_time(values, frame_step_ns):
    # Наибольший лаг N/4 — эвристика: дальше оценка ACF слишком шумная
    max_lag = len(values) // 4
    # statsmodels: ряд центрируется по среднему участка, ковариация делится на N
    r = acf(values, nlags=max_lag, fft=True)
    below = np.flatnonzero(r < threshold)
    lag = int(below[0]) if below.size else None
    return r, lag, (lag * frame_step_ns if lag is not None else None)


def assess(lag, taus, suitable):
    if lag is None:
        return "не оценивается: пересечения 1/e нет до N/4"
    if lag == 1:
        return "не оценивается: пересечение на первом лаге (окно короче медленной части ACF)"
    # Эвристика устойчивости: половины участка не должны расходиться больше чем вдвое
    if None in taus or max(taus) / min(taus) > 2:
        return "не оценивается: половины участка дают разные оценки"
    return "надёжно" if suitable else "условно: участок нестационарен"


def autocorrelation_table(data, segments, frame_step_ns):
    results, curves = [], {}

    for rep, by_name in segments.items():
        for name, (start, end) in by_name.items():
            for variable in variables:
                values = segment_values(data, rep, start, end, variable)
                r, lag, crossing = correlation_time(values, frame_step_ns)

                half = len(values) // 2
                taus = [correlation_time(part, frame_step_ns)[2] for part in (values[:half], values[half:])]
                detrended = correlation_time(detrend(values), frame_step_ns)[2]
                suitable, trend = check_trend(values, frame_step_ns)
                status = assess(lag, taus, suitable)
                tau = crossing if status in ("надёжно", "условно: участок нестационарен") else None
                curves[(rep, name, variable)] = (r, crossing, tau)

                results.append({
                    "Реализация": rep,
                    "Участок": name,
                    "Показатель": variable,
                    "Первый кадр": start,
                    "Последний кадр": end,
                    "Число кадров": len(values),
                    "Наибольший лаг (N/4), нс": (len(values) // 4) * frame_step_ns,
                    "tau_c, нс": tau,
                    "Оценка": status,
                    "Первое пересечение 1/e, нс": crossing,
                    "Пересечение: 1-я половина, нс": taus[0],
                    "Пересечение: 2-я половина, нс": taus[1],
                    "Пересечение после удаления тренда, нс": detrended,
                    **trend,
                })

    return pd.DataFrame(results), curves


def plot_acf(curves, rep, variable, frame_step_ns, title, name_plt):
    selected = {name: c for (r, name, v), c in curves.items() if r == rep and v == variable}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    crossings = [c for _, c, _ in selected.values() if c is not None]
    zoom_ns = max(5 * max(crossings), 2) if crossings else None

    for ax, xlim in zip(axes, [zoom_ns, None]):
        for name, (r, crossing, tau) in selected.items():
            lags_ns = np.arange(len(r)) * frame_step_ns
            label = f"{name}: tau_c = {tau:.2f} нс" if tau is not None else f"{name}: не оценивается"
            line, = ax.plot(lags_ns, r, linewidth=1, label=label)
            if tau is not None:
                ax.axvline(tau, color=line.get_color(), linestyle="--", linewidth=0.8)
            if crossing is not None:
                # Закрашенная точка — принятая оценка, пустая — пересечение, признанное ненадёжным
                ax.plot(crossing, r[round(crossing / frame_step_ns)], "o", color=line.get_color(),
                        markerfacecolor=line.get_color() if tau is not None else "white")

        ax.axhline(threshold, color="black", linestyle=":", label="1/e ≈ 0.368")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xlabel("Лаг, нс")
        ax.set_ylabel("Нормированная автокорреляция")
        if xlim:
            ax.set_xlim(0, xlim)

    axes[0].set_title("Начало: первое пересечение 1/e")
    axes[1].set_title("Весь диапазон лагов до N/4")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Автокорреляционная функция {variable}: {rep}, {title}")
    fig.tight_layout()
    fig.savefig(name_plt, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "A"
    out_dir = root / "second_member" / variant
    out_dir.mkdir(exist_ok=True)

    data = load_working_table(file, frame_step_ns)
    segments = build_variants(data)[variant]

    table, curves = autocorrelation_table(data, segments, frame_step_ns)
    table.to_excel(out_dir / f"Автокорреляция {variant}.xlsx", index=False)

    for rep in segments:
        for variable in variables:
            plot_acf(curves, rep, variable, frame_step_ns, VARIANT_TITLES[variant],
                     out_dir / f"ACF {variable} {rep}.png")

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table[["Реализация", "Участок", "Показатель", "Число кадров", "tau_c, нс", "Оценка"]])
        print(table[["Реализация", "Участок", "Показатель", "Первое пересечение 1/e, нс",
                     "Пересечение: 1-я половина, нс", "Пересечение: 2-я половина, нс",
                     "Пересечение после удаления тренда, нс"]])
        print(table[["Реализация", "Участок", "Показатель", "Изменение по тренду за участок, в SD",
                     "Среднее 1-й половины", "Среднее 2-й половины",
                     "KPSS p (H0: стационарен около среднего)", "ADF p (H0: единичный корень)",
                     "Пригодность для стационарной ACF"]].round(3))
