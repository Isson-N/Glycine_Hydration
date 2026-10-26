import pandas as pd
from pathlib import Path
import xlwings as xw
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


file="gly_matrix_all_reps.csv"
output_path="data.xlsx"

def write_excel_sheet(table, sheet_name, output):
    path = Path(output_path).resolve()

    book = xw.Book(str(path)) if path.exists() else xw.Book()

    if sheet_name in [sheet.name for sheet in book.sheets]:
        sheet = book.sheets[sheet_name]
        sheet.clear_contents()
    else:
        sheet = book.sheets.add(sheet_name,after=book.sheets[-1])

    sheet.range("A1").options(index=False).value = table
    sheet.autofit()

    book.save(str(path))

def inspect_data(file_path, output, sep=","):
    data = pd.read_csv(file_path, sep=sep)

    summary = pd.DataFrame({
        "Показатель": ["Число строк", "Число столбцов"],
        "Значение": [data.shape[0], data.shape[1]],
    })

    columns = pd.DataFrame({
        "Название столбца": data.columns,
        "Тип данных": data.dtypes.astype(str).to_numpy(),
    })

    write_excel_sheet(summary, "Общая информация", output)
    write_excel_sheet(columns, "Переменные", output)

    return data

def count_frames_by_rep(data, output):
    summary = (
        data.groupby("rep", dropna=False, sort=False)
        .size()
        .reset_index(name="Число кадров (строк)")
    )

    options = (
        {"mode": "a", "if_sheet_exists": "replace"}
        if Path(output).exists()
        else {"mode": "w"}
    )

    write_excel_sheet(summary, 'Реализации', output)

    return summary

def check_frame_order_and_duplicates(data, output):
    current = data[["rep", "frame"]].reset_index(drop=True)
    sorted_data = (
        current.sort_values(["rep", "frame"], kind="stable")
        .reset_index(drop=True)
    )

    print("Порядок rep, frame:", "верный" if current.equals(sorted_data) else "нарушен")

    duplicates = data.loc[data.duplicated(["rep", "frame"], keep=False)]
    missing = data.loc[data[["rep", "frame"]].isna().any(axis=1)]

    if not duplicates.empty:
        write_excel_sheet(duplicates, "Дубликаты кадров", output)
    else:
        print("Дубликатов нет")

    if not missing.empty:
        write_excel_sheet(missing, "Пропуски rep и frame", output)
    else:
        print("Пропусков rep и frame нет")

    return duplicates, missing

def check_variables(data, output):
    variables = [
        "n_gly_wat",
        "cn_gly_wat",
        "dist_ca_gly_com_bs",
        "sasa_gly",
    ]

    results = []

    for name in variables:
        if name not in data.columns:
            results.append({
                "Переменная": name,
                "Статус": "Столбец отсутствует",
            })
            continue

        original = data[name]
        numeric = pd.to_numeric(original, errors="coerce")
        finite = numeric.notna() & ~numeric.isin([float("inf"), -float("inf")])

        results.append({
            "Переменная": name,
            "Статус": "Столбец найден",
            "Тип данных": str(original.dtype),
            "Пропуски": int(original.isna().sum()),
            "Нечисловые значения": int(
                (original.notna() & numeric.isna()).sum()
            ),
            "Бесконечные значения": int(numeric.isin([
                float("inf"), -float("inf")
            ]).sum()),
            "Минимум": numeric[finite].min(),
            "Максимум": numeric[finite].max(),
            "Отрицательные значения": int((numeric < 0).sum()),
            "Дробные значения счётчика": (
                int((numeric[finite] % 1 != 0).sum())
                if name == "n_gly_wat"
                else None
            ),
        })

    summary = pd.DataFrame(results)
    write_excel_sheet(summary, "Проверка переменных", output)

    return summary

def prepare_working_table(data, output, frame_step_ns):
    columns = [
        "rep",
        "frame",
        "n_gly_wat",
        "cn_gly_wat",
        "dist_ca_gly_com_bs",
        "sasa_gly",
    ]

    working = (
        data.loc[data["rep"].isin(['rep0', 'rep2']), columns]
        .copy()
        .sort_values(["rep", "frame"])
        .reset_index(drop=True)
    )

    working.insert(2, "t_ns", working["frame"] * frame_step_ns)

    write_excel_sheet(working, "Рабочая таблица", output)

    return working

def plot_distance(data, rep, name):
    selected = (
        data.loc[data["rep"] == rep]
        .sort_values("frame")
    )

    if selected.empty:
        raise ValueError(f"Нет данных для реализации {rep}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        selected["t_ns"],
        selected["dist_ca_gly_com_bs"],
        linewidth=0.7,
    )
    axes[0].set_xlabel("Время, нс")
    axes[0].set_ylabel("Расстояние, Å")
    axes[0].set_title(f"Расстояние во времени: rep={rep}")

    axes[1].hist(
        selected["dist_ca_gly_com_bs"].dropna(),
        bins="auto",
    )
    axes[1].set_xlabel("Расстояние, Å")
    axes[1].set_ylabel("Число кадров")
    axes[1].set_title("Распределение расстояний")

    fig.tight_layout()
    plt.savefig(name)
    plt.show()

    return fig

def describe_segments(data, output, segments, rep, name_plt):
    selected = data.loc[data["rep"] == rep].sort_values("frame")

    if selected.empty:
        raise ValueError(f"Нет данных для {rep}")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        selected["t_ns"],
        selected["dist_ca_gly_com_bs"],
        linewidth=0.7,
    )

    results = []

    for name, (start, end) in segments.items():
        if start > end:
            raise ValueError(f"{name}: начальный кадр больше конечного")

        section = selected.loc[selected["frame"].between(start, end)]

        if section.empty:
            raise ValueError(f"{name}: в указанных границах нет кадров")

        t_start = section["t_ns"].iloc[0]
        t_end = section["t_ns"].iloc[-1]

        ax.axvspan(t_start, t_end, alpha=0.25, label=name)

        results.append({
            "Участок": name,
            "Реализация": rep,
            "Первый кадр": section["frame"].iloc[0],
            "Последний кадр": section["frame"].iloc[-1],
            "Число кадров": len(section),
            "Начало, нс": t_start,
            "Конец, нс": t_end,
            "Временной охват, нс": t_end - t_start,
            "Среднее расстояние, Å": section["dist_ca_gly_com_bs"].mean(),
            "Минимальное расстояние, Å": section["dist_ca_gly_com_bs"].min(),
            "Максимальное расстояние, Å": section["dist_ca_gly_com_bs"].max(),
            "Средняя SASA": section["sasa_gly"].mean(),
        })

    summary = pd.DataFrame(results)
    write_excel_sheet(summary, f"Участки {rep}", output)

    ax.set_xlabel("Время, нс")
    ax.set_ylabel("Расстояние, Å")
    ax.set_title(f"Выбранные участки: {rep}")
    ax.legend()
    fig.tight_layout()
    plt.savefig(name_plt)
    plt.show()

    return summary, fig

def describe_segments_r0(data, output, segments, thres, rep, name_plt):
    selected = data.loc[data["rep"] == rep].sort_values("frame")

    if selected.empty:
        raise ValueError(f"Нет данных для {rep}")

    fig, axes = plt.subplots(
        1, len(segments),
        figsize=(10, 4),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    results = []

    for name, (start, end) in segments.items():
        if start > end:
            raise ValueError(f"{name}: начальный кадр больше конечного")

        section = selected.loc[selected["frame"].between(start, end)]

        if section.empty:
            raise ValueError(f"{name}: в указанных границах нет кадров")

        t_start = section["t_ns"].iloc[0]
        t_end = section["t_ns"].iloc[-1]

        results.append({
            "Участок": name,
            "Реализация": rep,
            "Первый кадр": section["frame"].iloc[0],
            "Последний кадр": section["frame"].iloc[-1],
            "Число кадров": len(section),
            "Начало, нс": t_start,
            "Конец, нс": t_end,
            "Временной охват, нс": t_end - t_start,
            "Среднее расстояние, Å": section["dist_ca_gly_com_bs"].mean(),
            "Минимальное расстояние, Å": section["dist_ca_gly_com_bs"].min(),
            "Максимальное расстояние, Å": section["dist_ca_gly_com_bs"].max(),
            "Средняя SASA": section["sasa_gly"].mean(),
        })

    summary = pd.DataFrame(results)
    write_excel_sheet(summary, f"Участки {rep}", output)

    for i, (ax, (name, (start, end))) in enumerate(
            zip(axes, segments.items())
    ):
        section = selected.loc[selected["frame"].between(start, end)]

        if section.empty:
            raise ValueError(f"{name}: нет кадров")

        # Данные для графика с запасом по краям.
        context = selected.loc[selected["frame"].between(start - thres, end + thres)]

        ax.plot(
            context["t_ns"],
            context["dist_ca_gly_com_bs"],
            linewidth=0.8,
        )

        # Выделяем только выбранный сегмент.
        ax.axvspan(
            section["t_ns"].min(),
            section["t_ns"].max(),
            color="skyblue",
            alpha=0.25,
            zorder=0,
        )



        ax.set_xlim(context["t_ns"].min(), context["t_ns"].max())
        ax.set_title(name)
        ax.tick_params(axis="x", labelrotation=0)

        # Убираем внутренние границы.
        if i > 0:
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)

        if i < len(axes) - 1:
            ax.spines["right"].set_visible(False)

        # Диагональные отметки разрыва оси.
        d = 0.015
        marks = dict(
            transform=ax.transAxes,
            color="black",
            clip_on=False,
        )

        if i > 0:
            ax.plot((-d, d), (-d, d), **marks)
            ax.plot((-d, d), (1 - d, 1 + d), **marks)

        if i < len(axes) - 1:
            ax.plot((1 - d, 1 + d), (-d, d), **marks)
            ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **marks)

    axes[0].set_ylabel("Расстояние, Å")
    fig.supxlabel("Время, нс")
    fig.suptitle(f"Выбранные участки: {rep}")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.08)
    plt.savefig(name_plt)
    plt.show()

    return summary, fig

def hydration_statistics(data, output, segments, rep):
    selected = data.loc[data["rep"] == rep]
    results = []

    for name, (start, end) in segments.items():
        section = selected.loc[selected["frame"].between(start, end)]

        if section.empty:
            raise ValueError(f"{name}: в указанных границах нет кадров")

        for variable in ["n_gly_wat", "cn_gly_wat"]:
            values = section[variable]

            results.append({
                "Участок": name,
                "Реализация": rep,
                "Показатель": variable,
                "Число кадров": len(section),
                "Число значений без пропусков": values.count(),
                "Среднее": values.mean(),
                "Выборочная дисперсия": values.var(ddof=1),
                "Стандартное отклонение": values.std(ddof=1),
                "Минимум": values.min(),
                "Максимум": values.max(),
            })

    summary = pd.DataFrame(results)
    write_excel_sheet(summary, f"Гидратация {rep}", output)

    return summary

def hydration_distribution(data, output, segments, rep, name_plt):
    selected = data.loc[data["rep"] == rep]
    distributions = {}

    for name, (start, end) in segments.items():
        values = selected.loc[
            selected["frame"].between(start, end), "n_gly_wat"
        ].dropna()

        if values.empty:
            raise ValueError(f"{name}: нет значений n_gly_wat")

        if (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"{name}: счётчик должен быть целым и неотрицательным")

        distributions[name] = values.astype(int).value_counts(normalize=True)

    minimum = min(series.index.min() for series in distributions.values())
    maximum = max(series.index.max() for series in distributions.values())

    table = pd.DataFrame(distributions).reindex(
        range(minimum, maximum + 1)
    ).fillna(0)

    table.index.name = "Число молекул воды"

    write_excel_sheet(
        table.reset_index(),
        f"Распределение воды {rep}",
        output,
    )

    ax = table.plot.bar(figsize=(12, 5), width=0.85, rot=0)
    ax.set_xlabel("Число молекул воды")
    ax.set_ylabel("Доля кадров")
    ax.set_title(f"Распределение n_gly_wat: {rep}")
    ax.legend(title="Участок")
    ax.figure.tight_layout()
    plt.savefig(name_plt,dpi=300)
    plt.show()

    return table

def compare_hydration_indicators(data, output, segments, rep, name_plt):
    selected = data.loc[data["rep"] == rep]
    results = []

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, (start, end) in segments.items():
        pairs = selected.loc[
            selected["frame"].between(start, end),
            ["n_gly_wat", "cn_gly_wat"],
        ].dropna()

        if pairs.empty:
            raise ValueError(f"{name}: нет пар значений для сравнения")

        correlation = (
            pairs["n_gly_wat"].corr(pairs["cn_gly_wat"])
            if len(pairs) > 1 and (pairs.nunique() > 1).all()
            else float("nan")
        )

        results.append({
            "Участок": name,
            "Реализация": rep,
            "Число полных пар": len(pairs),
            "Среднее n_gly_wat": pairs["n_gly_wat"].mean(),
            "Среднее cn_gly_wat": pairs["cn_gly_wat"].mean(),
            "Корреляция Пирсона": correlation,
        })

        ax.scatter(
            pairs["n_gly_wat"],
            pairs["cn_gly_wat"],
            s=10,
            alpha=0.2,
            label=name,
        )

    summary = pd.DataFrame(results)
    write_excel_sheet(summary, f"Сравнение n и cn {rep}", output)

    ax.set_xlabel("Число молекул воды — n_gly_wat")
    ax.set_ylabel("Непрерывный показатель — cn_gly_wat")
    ax.set_title(f"Связь показателей гидратации: {rep}")
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        name_plt,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return summary, fig

def plot_hydration_boxplots(data, segments, rep, name_plt):
    selected = data.loc[data["rep"] == rep]
    fig, ax = plt.subplots(figsize=(11, 5))

    variables = ["n_gly_wat", "cn_gly_wat"]
    colors = ["skyblue", "salmon"]
    centers = []

    for i, (name, (start, end)) in enumerate(segments.items()):
        section = selected.loc[selected["frame"].between(start, end)]
        center = i * 2
        centers.append(center)

        for variable, color, offset in zip(variables, colors, [-0.4, 0.4]):
            values = section[variable].dropna()

            if values.empty:
                raise ValueError(f"{name}: нет значений {variable}")

            box = ax.boxplot(
                values.to_numpy(),
                positions=[center + offset],
                widths=0.65,
                patch_artist=True,
                showmeans=True,
                manage_ticks=False,
                medianprops={"color": "red", 'zorder': 5},
                meanprops={
                    "marker": "x",
                    "markeredgecolor": "green",
                    "markersize": 7,
                    "markeredgewidth": 1.5,
                },
            )
            box["boxes"][0].set_facecolor(color)

    ax.set_xticks(centers)
    ax.set_xticklabels(list(segments), rotation=0)
    ax.set_ylabel("Значение показателя")
    ax.set_title(f"Показатели гидратации в выбранных областях: {rep}")
    ax.grid(axis="y", alpha=0.3)

    ax.legend(handles=[
        Patch(facecolor=colors[0], label="n_gly_wat"),
        Patch(facecolor=colors[1], label="cn_gly_wat"),
    ])

    fig.tight_layout()
    fig.savefig(
        name_plt,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    return fig

if __name__=='__main__':
    #Подготовка данных
    data=inspect_data(file,output_path)

    count_frames_by_rep(data,output_path)

    check_frame_order_and_duplicates(data,output_path)

    check_variables(data,output_path)

    #Создание рабочей таблицы
    work_data=prepare_working_table(data,output_path,0.05)

    # Работа с данными rep2
    plot_distance(work_data,'rep2', 'Распределение расстояний для rep2')

    seg={
        'Область малых расстояний': (0,5200),
        'Область больших расстояний': (5300,10500)
    }

    describe_segments(work_data,output_path,seg,'rep2','Распределение расстояния по сегментам')

    hydration_statistics(work_data,output_path,seg,'rep2')

    hydration_distribution(work_data,output_path,seg,'rep2', 'Распределение числа молекул воды')

    compare_hydration_indicators(work_data,output_path,seg,'rep2','Связь числа молекул воды с непрерывным показателем гидратации')

    plot_hydration_boxplots(work_data,seg,'rep2','Связь числа молекул воды с непрерывным показателем гидратации (ящики)')

    # Работа с данными rep0

    plot_distance(work_data, 'rep0', 'Распределение расстояний для rep0')

    seg_0 = {
        'Область малых расстояний': (100, 243),
        'Область больших расстояний': (8317, 8460)
    }

    describe_segments_r0(work_data, output_path, seg_0, 100, 'rep0', 'Распределение расстояния по сегментам (rep0)')

    hydration_statistics(work_data, output_path, seg_0, 'rep0')

    hydration_distribution(work_data, output_path, seg_0, 'rep0', 'Распределение числа молекул воды (rep0)')

    compare_hydration_indicators(work_data, output_path, seg_0, 'rep0',
                                 'Связь числа молекул воды с непрерывным показателем гидратации (rep0)')

    plot_hydration_boxplots(work_data, seg_0, 'rep0',
                            'Связь числа молекул воды с непрерывным показателем гидратации (ящики) (rep0)')


