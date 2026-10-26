"""Этапы 5–6: неопределённость различий между участками блочным бутстрэпом (rep2 — этап 5, rep0 — этап 6).

Запуск из папки проекта:
    python second_member/bootstrap.py C
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch.bootstrap import MovingBlockBootstrap, optimal_block_length

from autocorrelation import correlation_time, load_working_table, segment_values
from segments import LARGE, SMALL, VARIANT_TITLES, build_variants


root = Path(__file__).resolve().parent.parent
file = root / "data" / "gly_matrix_all_reps.csv"
frame_step_ns = 0.05
replicates = 2000
block_factors = [0.5, 1, 2, 4]  # чувствительность к длине блока
seed = 1
# В rep0 участки по 149 кадров: tau_c не оценивается (этап 4), поэтому времена корреляции там не сравниваем.
# В rep2 tau_c сайта принимается по первому пересечению 1/e — допущение, согласованное в работе.
tau_reps = {"rep2"}
min_blocks = 10  # меньше блоков на участок — бутстрэп почти не видит изменчивость, вывод не делаем


def statistics(n, cn, with_tau):
    """Сравниваемые показатели одного участка."""
    result = {
        "Среднее n_gly_wat": n.mean(),
        "Дисперсия n_gly_wat": n.var(ddof=1),
        "Среднее cn_gly_wat": cn.mean(),
        "Дисперсия cn_gly_wat": cn.var(ddof=1),
    }
    if with_tau:
        result["tau_c n_gly_wat, нс"] = correlation_time(n, frame_step_ns)[2]
        result["tau_c cn_gly_wat, нс"] = correlation_time(cn, frame_step_ns)[2]
    return result


def tvd(a, b):
    """Расстояние полной вариации между распределениями целочисленного счётчика: 0 — совпадают, 1 — не пересекаются."""
    k = np.arange(min(a.min(), b.min()), max(a.max(), b.max()) + 1)
    pa = np.array([(a == v).mean() for v in k])
    pb = np.array([(b == v).mean() for v in k])
    return 0.5 * np.abs(pa - pb).sum()


def block_length(*series):
    # Оптимальная длина по Политису–Уайту, берём наибольшую из участков и показателей — осторожный выбор
    lengths = [optimal_block_length(s)["circular"].iloc[0] for s in series]
    return max(2, int(np.ceil(max(lengths))))


def bootstrap_differences(site, plateau, length, rng, with_tau):
    """Независимый блочный бутстрэп каждого участка; разность «большие минус малые» для каждого повтора.
    Блоки скользящие без склейки конца с началом: при тренде круговая склейка дала бы ложный скачок."""
    draws = []
    boot_s = MovingBlockBootstrap(length, *site, seed=rng)
    boot_p = MovingBlockBootstrap(length, *plateau, seed=rng)
    for (s, _), (p, _) in zip(boot_s.bootstrap(replicates), boot_p.bootstrap(replicates)):
        stats_s, stats_p = statistics(*s, with_tau), statistics(*p, with_tau)
        draws.append({key: _diff(stats_p[key], stats_s[key]) for key in stats_s})
    return pd.DataFrame(draws)


def bootstrap_p(draws):
    """Номинальный двусторонний p для H0 «разность = 0» по хвостовым долям повторов, согласованный с перцентильным ДИ
    (у границы возможны редкие расхождения из-за дискретности). Предел разрешения 2 / (повторов + 1).
    Калибровку не гарантирует: она зависит от того, верно ли бутстрэп воспроизводит изменчивость."""
    draws = np.asarray(draws)
    tail = min((draws <= 0).sum(), (draws >= 0).sum())
    return min(1.0, 2 * (tail + 1) / (len(draws) + 1))


def _diff(a, b):
    return np.nan if a is None or b is None else a - b


def block_permutation_tvd(site_n, plateau_n, length, rng):
    """Тест H0 «распределения одинаковы»: блоки обоих участков перемешиваются и делятся заново."""
    blocks = [x[i:i + length] for x in (site_n, plateau_n) for i in range(0, len(x) - length + 1, length)]
    n_site = len(site_n) // length
    # Наблюдаемое TVD — на тех же обрезанных до целого числа блоков рядах, что и нулевое распределение
    observed = tvd(np.concatenate(blocks[:n_site]), np.concatenate(blocks[n_site:]))
    null = []
    for _ in range(replicates):
        order = rng.permutation(len(blocks))
        a = np.concatenate([blocks[i] for i in order[:n_site]])
        b = np.concatenate([blocks[i] for i in order[n_site:]])
        null.append(tvd(a, b))
    return observed, (np.sum(np.array(null) >= observed) + 1) / (replicates + 1)


def compare_regions(data, segments, rep):
    rng = np.random.default_rng(seed)
    site = [segment_values(data, rep, *segments[rep][SMALL], v) for v in ("n_gly_wat", "cn_gly_wat")]
    plateau = [segment_values(data, rep, *segments[rep][LARGE], v) for v in ("n_gly_wat", "cn_gly_wat")]
    base_length = block_length(*site, *plateau)

    with_tau = rep in tau_reps
    observed_s, observed_p = statistics(*site, with_tau), statistics(*plateau, with_tau)
    results, draws_main = [], None

    for factor in block_factors:
        # Блок не длиннее самого короткого участка (иначе его не вырезать); такой случай всё равно «мало блоков»
        length = min(max(2, int(round(base_length * factor))), len(site[0]), len(plateau[0]))
        blocks = min(len(site[0]), len(plateau[0])) // length
        verdict = (lambda differs: "мало блоков" if blocks < min_blocks else ("да" if differs else "нет"))

        def interval_verdict(diff, mean, low, high):
            # Бутстрэп смещён, если оценка вне своего ДИ или среднее повторов уехало больше чем на полширины ДИ
            if blocks >= min_blocks and (not low <= diff <= high or abs(mean - diff) > 0.5 * (high - low) / 2):
                return "не разрешается: бутстрэп смещён"
            return verdict(low > 0 or high < 0)
        draws = bootstrap_differences(site, plateau, length, rng, with_tau)
        if factor == 1:
            draws_main = draws
        tvd_obs, tvd_p = block_permutation_tvd(site[0], plateau[0], length, rng)

        for key in observed_s:
            diff = _diff(observed_p[key], observed_s[key])
            valid = draws[key].dropna()
            low, high = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
            results.append({
                "Реализация": rep,
                "Длина блока, кадров": length,
                "Длина блока, нс": length * frame_step_ns,
                "Множитель длины блока": factor,
                "Показатель": key,
                "Малые расстояния": observed_s[key],
                "Большие расстояния": observed_p[key],
                "Разность (большие − малые)": diff,
                # Если среднее по повторам далеко от наблюдаемой разности, бутстрэп смещён и его ДИ ненадёжен
                "Среднее разностей по повторам": valid.mean(),
                "95% ДИ: нижняя": low,
                "95% ДИ: верхняя": high,
                "p бутстрэпа": bootstrap_p(valid) if len(valid) else np.nan,
                "Повторов без оценки": replicates - len(valid),
                "Блоков на участок": blocks,
                "Различие больше неопределённости": interval_verdict(diff, valid.mean(), low, high),
            })
        results.append({
            "Реализация": rep, "Длина блока, кадров": length, "Длина блока, нс": length * frame_step_ns,
            "Множитель длины блока": factor, "Показатель": "TVD распределений n_gly_wat",
            "Разность (большие − малые)": tvd_obs, "p блочной перестановки": tvd_p,
            "Блоков на участок": blocks,
            "Различие больше неопределённости": verdict(tvd_p < 0.05),
        })

    return pd.DataFrame(results), draws_main, (observed_s, observed_p), base_length


def summarize(table):
    """Итог по показателю с учётом всех длин блока, где блоков достаточно."""
    rows = []
    for (rep, key), group in table.groupby(["Реализация", "Показатель"], sort=False):
        verdicts = group.loc[group["Различие больше неопределённости"] != "мало блоков", "Различие больше неопределённости"]
        main_ok = (group.loc[group["Множитель длины блока"] == 1, "Различие больше неопределённости"] != "мало блоков").all()
        if verdicts.empty or not main_ok:
            # Если блоков достаточно только при укороченном блоке, вывод держится на наименее осторожной длине
            result = "не оценивается: при основной длине блока мало блоков"
        elif verdicts.str.startswith("не разрешается").any():
            result = "не разрешается: бутстрэп смещён"
        elif (verdicts == "да").all():
            result = "да при всех допустимых длинах блока"
        elif (verdicts == "нет").all():
            result = "нет при всех допустимых длинах блока"
        else:
            result = "зависит от длины блока"
        main = group.loc[group["Множитель длины блока"] == 1].iloc[0]
        rows.append({"Реализация": rep, "Показатель": key, "Малые расстояния": main["Малые расстояния"],
                     "Большие расстояния": main["Большие расстояния"],
                     "Разность (большие − малые)": main["Разность (большие − малые)"],
                     "95% ДИ при основной длине блока": (f"[{main['95% ДИ: нижняя']:.3f}; {main['95% ДИ: верхняя']:.3f}]"
                                                         if pd.notna(main.get("95% ДИ: нижняя")) else ""),
                     # p — бутстрэпа для разностей, блочной перестановки для TVD; при смещённом бутстрэпе не приводится
                     "p при основной длине блока": (np.nan if result.startswith("не разрешается")
                                                    else main.get("p бутстрэпа") if key[:3] != "TVD"
                                                    else main.get("p блочной перестановки")),
                     "Допустимых длин блока": len(verdicts), "Итог": result})
    return pd.DataFrame(rows)


def plot_intervals(table, rep, title, name_plt):
    keys = [k for k in table["Показатель"].unique() if not k.startswith("TVD")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    for ax, key in zip(axes.flat, keys):
        rows = table.loc[table["Показатель"] == key]
        y = np.arange(len(rows))
        # Перцентильный ДИ может не содержать точечную оценку (смещение бутстрэпа), поэтому рисуем раздельно
        ax.hlines(y, rows["95% ДИ: нижняя"], rows["95% ДИ: верхняя"], linewidth=2)
        ax.plot(rows["Разность (большие − малые)"], y, "o", color="black")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y, [f"блок {l:g} нс" for l in rows["Длина блока, нс"]])
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("Разность «большие − малые» с 95% ДИ")
    fig.suptitle(f"{rep}, {title}: различия между участками и их неопределённость при разной длине блока")
    fig.tight_layout()
    fig.savefig(name_plt, dpi=200)
    plt.close(fig)


def plot_tau_bootstrap(draws, rep, title, name_plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ["tau_c n_gly_wat, нс", "tau_c cn_gly_wat, нс"]):
        ax.hist(draws[key].dropna(), bins=40)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(f"Разность {key} (большие − малые)")
        ax.set_ylabel("Число повторов")
    fig.suptitle(f"{rep}, {title}: бутстрэп-распределение разности времён корреляции")
    fig.tight_layout()
    fig.savefig(name_plt, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "C"
    out_dir = root / "second_member" / variant
    out_dir.mkdir(exist_ok=True)

    data = load_working_table(file, frame_step_ns)
    segments = build_variants(data)[variant]

    tables = []
    for rep in segments:
        table, draws, _, base_length = compare_regions(data, segments, rep)
        print(f"{rep}: длина блока по Политису–Уайту {base_length} кадров ({base_length * frame_step_ns:.2f} нс)")
        tables.append(table)
        plot_intervals(table, rep, VARIANT_TITLES[variant], out_dir / f"Бутстрэп различия {rep}.png")
        if rep in tau_reps:
            plot_tau_bootstrap(draws, rep, VARIANT_TITLES[variant], out_dir / f"Бутстрэп tau_c {rep}.png")

    result = pd.concat(tables, ignore_index=True)
    summary = summarize(result)
    with pd.ExcelWriter(out_dir / f"Бутстрэп {variant}.xlsx") as writer:
        summary.to_excel(writer, sheet_name="Сводка", index=False)
        result.to_excel(writer, sheet_name="Все длины блока", index=False)

    main = result.loc[result["Множитель длины блока"] == 1]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(main[["Реализация", "Показатель", "Малые расстояния", "Большие расстояния",
                    "Разность (большие − малые)", "Среднее разностей по повторам", "95% ДИ: нижняя", "95% ДИ: верхняя",
                    "p бутстрэпа", "p блочной перестановки", "Различие больше неопределённости"]].round(3).to_string(index=False))
        print(summary.round(3).to_string(index=False))
        print(result.pivot_table(index=["Реализация", "Показатель"], columns="Длина блока, нс",
                                 values="Различие больше неопределённости", aggfunc="first"))
