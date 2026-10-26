"""Этапы 1–3: код участника 1 (first_member/main.py) без изменений, но на участках варианта C.

Подменяется только write_excel_sheet: в оригинале он пишет в открытый Excel через xlwings,
которого нет на сервере. Здесь листы собираются и сохраняются одной книгой через pandas.

Запуск из папки проекта:
    python second_member/run_member1_C.py
"""
import os
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

root = Path(__file__).resolve().parent.parent
sys.modules["xlwings"] = types.ModuleType("xlwings")
sys.path.insert(0, str(root / "first_member"))
import main as member1

from segments import LARGE, SMALL, build_variants

out_dir = root / "second_member" / "C" / "stage1-3"
book = {}


def write_excel_sheet(table, sheet_name, output):
    book[sheet_name] = table


member1.write_excel_sheet = write_excel_sheet

# Названия участков как у участника 1
names = {SMALL: "Область малых расстояний", LARGE: "Область больших расстояний"}

if __name__ == "__main__":
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)  # PNG участника 1 сохраняются в текущую папку
    output = "data_C.xlsx"

    data = member1.inspect_data(root / "data" / "gly_matrix_all_reps.csv", output)
    member1.count_frames_by_rep(data, output)
    member1.check_frame_order_and_duplicates(data, output)
    member1.check_variables(data, output)
    work_data = member1.prepare_working_table(data, output, 0.05)

    variant = build_variants(work_data)["C"]
    seg = {names[k]: v for k, v in variant["rep2"].items()}
    seg_0 = {names[k]: v for k, v in variant["rep0"].items()}
    print("rep2:", seg, "\nrep0:", seg_0)

    member1.plot_distance(work_data, "rep2", "Распределение расстояний для rep2")
    member1.describe_segments(work_data, output, seg, "rep2", "Распределение расстояния по сегментам")
    member1.hydration_statistics(work_data, output, seg, "rep2")
    member1.hydration_distribution(work_data, output, seg, "rep2", "Распределение числа молекул воды")
    member1.compare_hydration_indicators(work_data, output, seg, "rep2",
                                         "Связь числа молекул воды с непрерывным показателем гидратации")
    member1.plot_hydration_boxplots(work_data, seg, "rep2",
                                    "Связь числа молекул воды с непрерывным показателем гидратации (ящики)")

    member1.plot_distance(work_data, "rep0", "Распределение расстояний для rep0")
    member1.describe_segments_r0(work_data, output, seg_0, 100, "rep0", "Распределение расстояния по сегментам (rep0)")
    member1.hydration_statistics(work_data, output, seg_0, "rep0")
    member1.hydration_distribution(work_data, output, seg_0, "rep0", "Распределение числа молекул воды (rep0)")
    member1.compare_hydration_indicators(work_data, output, seg_0, "rep0",
                                         "Связь числа молекул воды с непрерывным показателем гидратации (rep0)")
    member1.plot_hydration_boxplots(work_data, seg_0, "rep0",
                                    "Связь числа молекул воды с непрерывным показателем гидратации (ящики) (rep0)")

    with pd.ExcelWriter(output) as writer:
        for sheet_name, table in book.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    print(out_dir / output)
