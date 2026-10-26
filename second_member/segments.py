"""Варианты участков A, B, C. Границы — только по расстоянию dist_ca_gly_com_bs.

A — участки участника 1 без изменений.
B — полные фазы: весь сайт против всей области около 8 Å.
C — итоговый: rep2 — сайт и окно 1 плато (первые кадры плато длиной, равной сайту);
    rep0 — до и после перехода (равные окна вплотную до ухода из сайта и после).
C2 — проверка устойчивости: rep2 — сайт и следующее окно плато той же длины.

Правило фаз для B и C (одно для обеих реплик):
  * сглаживаем dist скользящей медианой по 41 кадру (2 нс);
  * уход из сайта — первый кадр, где медиана > 5 Å (минимум между модами гистограммы);
  * начало выхода — последний кадр, где медиана <= 10 Å; дальше глицин ниже 10 Å не возвращается;
  * вокруг переходов отбрасываем по MARGIN кадров (2 нс).
Все границы — номера кадров, обе включены.
"""
import pandas as pd

MEDIAN_WINDOW = 41
SITE_MAX_A = 5.0
PLATEAU_MAX_A = 10.0
MARGIN = 41

SMALL = "Малые расстояния"
LARGE = "Большие расстояния"


def phase_bounds(dist):
    """dist индексирован номером кадра. Возвращает (кадр ухода из сайта, последний кадр до выхода)."""
    median = dist.rolling(MEDIAN_WINDOW, center=True, min_periods=1).median()
    leave_site = median[median > SITE_MAX_A].index.min()
    exit_start = median[median <= PLATEAU_MAX_A].index.max()
    if pd.isna(leave_site) or exit_start <= leave_site:
        raise ValueError("Правило фаз не нашло ухода из сайта до выхода")
    return int(leave_site), int(exit_start)


def build_variants(data):
    """{вариант: {rep: {название участка: (первый кадр, последний кадр)}}}"""
    dist = data.set_index("frame").groupby("rep")["dist_ca_gly_com_bs"]

    leave, exit_start = phase_bounds(dist.get_group("rep2"))
    site = (0, leave - MARGIN)
    plateau = (leave + MARGIN, exit_start - MARGIN)
    length = site[1] - site[0] + 1
    # Плато режем подряд на окна длины сайта: окно 1 — основное (сразу после перехода), окно 2 — проверка
    window_1 = (plateau[0], plateau[0] + length - 1)
    window_2 = (window_1[1] + 1, window_1[1] + length)

    leave, exit_start = phase_bounds(dist.get_group("rep0"))
    # Остановка у 8 Å в rep0 короткая (~7 нс), поля от неё не отрезаем
    stop = (leave, exit_start)
    stop_length = stop[1] - stop[0] + 1
    site_end = leave - MARGIN

    return {
        "A": {
            "rep2": {SMALL: (0, 5200), LARGE: (5300, 10500)},
            "rep0": {SMALL: (100, 243), LARGE: (8317, 8460)},
        },
        "B": {
            "rep2": {SMALL: site, LARGE: plateau},
            "rep0": {SMALL: (0, site_end), LARGE: stop},
        },
        "C": {
            "rep2": {SMALL: site, LARGE: window_1},
            "rep0": {SMALL: (site_end - stop_length + 1, site_end), LARGE: stop},
        },
        "C2": {
            "rep2": {SMALL: site, LARGE: window_2},
        },
    }


VARIANT_TITLES = {
    "A": "A. Участник 1",
    "B": "B. Полные фазы",
    "C": "выбранные участки",
    "C2": "C2. Проверка: rep2 — сайт и окно 2 плато",
}
