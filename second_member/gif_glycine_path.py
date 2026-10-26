"""Схематичная GIF: белок GlyT1 (8WFL) вращается, глицин движется по оси внеклеточного вестибюля
на расстоянии dist_ca_gly_com_bs(t) от центра сайта из выбранной реплики (по умолчанию rep2).

Это реконструкция, а не траектория: координат глицина в CSV нет, только расстояние.
Направление пути — ось канала, найденная программой HOLE от центра масс сайта связывания.
Ориентация в мембране — по гомологу SERT из базы OPM (5I6X, уложен в мембрану): 8WFL накладывается на него,
нормаль к мембране ставится вертикально (внеклеточная сторона вверху), вращение — вокруг неё через центр сайта.

Запуск из папки проекта:
    python second_member/gif_glycine_path.py [rep2|rep0] [число кадров для пробы]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pymol import cmd

root = Path(__file__).resolve().parent.parent
structure = root / "data" / "structures" / "8WFL.pdb"
membrane_ref = root / "data" / "structures" / "opm_5i6x.pdb"  # в файлах OPM ось z — нормаль к мембране
file = root / "data" / "gly_matrix_all_reps.csv"
rep = sys.argv[1] if len(sys.argv) > 1 else "rep2"
output = root / "second_member" / "report" / (f"glycine_path_{rep}.gif" if len(sys.argv) <= 2 else "glycine_path_test.gif")
env = Path(sys.prefix)  # окружение conda, где установлен hole2
frame_step_ns = 0.05
n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 120  # для пробного прогона: число кадров вторым аргументом
size = 700

# Остатки сайта в нумерации CSV; в 8WFL они на 43 больше (GLY18 → GLY61, TRP279 → TRP322, THR375 → THR418)
site_csv = [18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 98, 99, 270, 272, 273, 274, 275, 276, 279, 306,
            371, 372, 375, 376, 431]
site_sel = "protein and resid " + " ".join(str(r + 43) for r in site_csv)


def open_direction(protein, com):
    """Направление из сайта, вдоль которого лучи не задевают белок: открытый внеклеточный вестибюль."""
    from scipy.spatial import cKDTree
    tree = cKDTree(protein)
    i = np.arange(2000) + 0.5
    phi, theta = np.arccos(1 - 2 * i / 2000), np.pi * (1 + 5 ** 0.5) * i
    dirs = np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    steps = np.arange(4, 40, 0.5)
    free = [d for d in dirs if (tree.query(com + np.outer(steps, d))[0] > 2.5).all()]
    d = np.mean(free, axis=0)
    return d / np.linalg.norm(d)


def hole_path(pdb, com, direction, workdir):
    radius = env / "share" / "hole2" / "rad" / "simple.rad"
    (workdir / "hole.inp").write_text(
        f"coord {pdb}\nradius {radius}\nsphpdb {workdir / 'hole.sph'}\n"
        f"cpoint {com[0]:.3f} {com[1]:.3f} {com[2]:.3f}\ncvect {direction[0]:.3f} {direction[1]:.3f} {direction[2]:.3f}\n"
        "sample 0.5\nendrad 12.\n")
    subprocess.run([env / "bin" / "hole"], stdin=open(workdir / "hole.inp"), capture_output=True, check=True)
    centres = mda.Universe(str(workdir / "hole.sph"), format="PDB").atoms.positions
    # Оставляем точки по внеклеточную сторону от сайта и упорядочиваем по удалению от него
    ahead = centres[(centres - com) @ direction > 0]
    return ahead[np.argsort((ahead - com) @ direction)]


def place_on_path(path, com, distances):
    """Точка пути, где расстояние до центра сайта равно заданному; за концом пути — продолжение по прямой."""
    path = np.vstack([com, path])
    reach = np.maximum.accumulate(np.linalg.norm(path - com, axis=1))
    tail = path[-1] - path[-5]
    tail /= np.linalg.norm(tail)
    points = []
    for d in distances:
        if d <= reach[-1]:
            points.append(np.array([np.interp(d, reach, path[:, k]) for k in range(3)]))
        else:
            points.append(path[-1] + tail * (d - reach[-1]))
    return np.array(points)


def align_to_membrane(workdir):
    """Накладывает 8WFL на гомолог из OPM; возвращает путь к выровненному белку и полутолщину мембраны, Å."""
    cmd.load(str(membrane_ref), "ref")
    cmd.load(str(structure), "glyt1")
    cmd.align("glyt1 and polymer and name CA", "ref and polymer and name CA")
    half = max(abs(a.coord[2]) for a in cmd.get_model("ref and resn DUM").atom)
    aligned = workdir / "aligned.pdb"
    cmd.save(str(aligned), "glyt1 and polymer")
    cmd.reinitialize()
    return aligned, half


# Поворот, переводящий нормаль к мембране (z в OPM) в вертикаль кадра (y)
z_to_y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def membrane_planes(y_levels, half_width=38):
    """Полупрозрачные плоскости границ мембраны (CGO PyMOL)."""
    from pymol.cgo import ALPHA, BEGIN, COLOR, END, NORMAL, TRIANGLE_STRIP, VERTEX
    obj = []
    for y in y_levels:
        obj += [ALPHA, 0.25, COLOR, 0.55, 0.55, 0.55, BEGIN, TRIANGLE_STRIP, NORMAL, 0.0, 1.0, 0.0]
        for x, z in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            obj += [VERTEX, x * half_width, y, z * half_width]
        obj += [END]
    return obj


def trajectory_samples():
    data = pd.read_csv(file, usecols=["rep", "frame", "n_gly_wat", "dist_ca_gly_com_bs"])
    selected = data.loc[data["rep"] == rep].sort_values("frame")
    dist = selected["dist_ca_gly_com_bs"].rolling(41, center=True, min_periods=1).median().to_numpy()
    water = selected["n_gly_wat"].rolling(41, center=True, min_periods=1).mean().to_numpy()
    t_ns = selected["frame"].to_numpy() * frame_step_ns
    idx = np.linspace(0, len(selected) - 1, n_frames).astype(int)
    return t_ns, dist, water, idx


def inset(t_ns, dist, i):
    fig, ax = plt.subplots(figsize=(3.4, 1.6), dpi=100)
    ax.plot(t_ns, dist, linewidth=0.6, color="0.3")
    ax.plot(t_ns[i], dist[i], "o", color="red")
    ax.set_xlabel("время, нс", fontsize=7)
    ax.set_ylabel("Å", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    image = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return image


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        aligned, half = align_to_membrane(tmp)
        u = mda.Universe(str(aligned))
        protein = u.select_atoms("protein")
        com = u.select_atoms(site_sel + " and not name H*").center_of_mass()
        direction = open_direction(protein.select_atoms("not name H*").positions, com)
        protein.write(str(tmp / "prot.pdb"))
        path = hole_path(tmp / "prot.pdb", com, direction, tmp)

        # Центр сайта — в начало координат, нормаль к мембране — вертикально вверх
        protein.positions = (protein.positions - com) @ z_to_y.T
        protein.write(str(tmp / "oriented.pdb"))
        path = (path - com) @ z_to_y.T
        origin = np.zeros(3)
        membrane_y = [half - com[2], -half - com[2]]

        t_ns, dist, water, idx = trajectory_samples()
        positions = place_on_path(path, origin, dist[idx])

        cmd.load(str(tmp / "oriented.pdb"), "glyt1")
        cmd.hide("everything")
        cmd.show("cartoon", "glyt1")
        cmd.spectrum("resi", "rainbow", "glyt1 and name CA")
        cmd.set("cartoon_transparency", 0.55)
        cmd.show("sticks", "glyt1 and resi " + "+".join(str(r + 43) for r in site_csv) + " and not name N+C+O")
        cmd.set("stick_radius", 0.15)
        for k, p in enumerate(path[::4]):
            cmd.pseudoatom("axis", pos=[float(x) for x in p], name=f"P{k}")
        cmd.show("spheres", "axis")
        cmd.set("sphere_scale", 0.25, "axis")
        cmd.color("grey70", "axis")
        cmd.pseudoatom("gly", pos=[0.0, 0.0, 0.0])
        cmd.show("spheres", "gly")
        cmd.set("sphere_scale", 1.6, "gly")
        cmd.color("red", "gly")
        cmd.load_cgo(membrane_planes(membrane_y), "membrane")
        cmd.bg_color("white")
        cmd.set("ray_opaque_background", 1)
        # Камера смотрит вдоль −z, ось y экрана — нормаль к мембране; вращение вокруг вертикали через центр сайта
        cmd.set_view((1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -250, 0, 6, 0, 200, 300, 0))

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 17)
            small = ImageFont.truetype("DejaVuSans.ttf", 13)
        except OSError:
            font = small = ImageFont.load_default()

        frames = []
        for k, (i, p) in enumerate(zip(idx, positions)):
            cmd.alter_state(1, "gly", f"(x, y, z) = ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
            cmd.turn("y", 360 / n_frames)
            png = tmp / f"f{k:03d}.png"
            cmd.png(str(png), width=size, height=size, dpi=100, ray=1, quiet=1)
            image = Image.open(png).convert("RGB")
            draw = ImageDraw.Draw(image)
            draw.text((12, 10), f"{rep}, t = {t_ns[i]:.0f} нс (схема)", fill="black", font=font)
            draw.text((12, 34), f"расстояние до центра сайта: {dist[i]:.1f} Å", fill="black", font=font)
            draw.text((12, 58), f"молекул воды рядом: {water[i]:.1f}", fill="black", font=font)
            draw.text((12, 92), "↑ внеклеточная сторона", fill="dimgray", font=small)
            draw.text((12, size - 26), "↓ цитоплазма   (серые плоскости — границы мембраны)", fill="dimgray", font=small)
            plot = inset(t_ns, dist, i)
            image.paste(plot, (size - plot.width - 8, 8))
            frames.append(np.asarray(image))
            print(f"\r{k + 1}/{n_frames}", end="")

    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, duration=0.09, loop=0)
    print("\n", output)
