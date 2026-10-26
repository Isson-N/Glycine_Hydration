"""Склейка двух схематичных GIF: слева rep2 (основная), справа rep0 (контрольная).

Кадры идут синхронно: каждая реплика проходит свою траекторию целиком за те же 120 кадров
(rep2 — 916 нс, rep0 — 425 нс), поэтому время в подписях разное; вращение белка одинаковое.

Запуск из папки проекта (после gif_glycine_path.py rep2 и rep0):
    python second_member/gif_side_by_side.py
"""
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

report = Path(__file__).resolve().parent / "report"
left, right = report / "glycine_path_rep2.gif", report / "glycine_path_rep0.gif"
output = report / "glycine_path_rep2_vs_rep0.gif"
header = 34

font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
frames = []
for a, b in zip(imageio.mimread(left, memtest=False), imageio.mimread(right, memtest=False)):
    a, b = Image.fromarray(a).convert("RGB"), Image.fromarray(b).convert("RGB")
    canvas = Image.new("RGB", (a.width + b.width + 10, a.height + header), "white")
    canvas.paste(a, (0, header))
    canvas.paste(b, (a.width + 10, header))
    draw = ImageDraw.Draw(canvas)
    draw.line([(a.width + 5, header), (a.width + 5, canvas.height)], fill="lightgray", width=2)
    draw.text((12, 6), "rep2 — основная траектория (916 нс)", fill="black", font=font)
    draw.text((a.width + 22, 6), "rep0 — контрольная траектория (425 нс)", fill="black", font=font)
    frames.append(np.asarray(canvas))

imageio.mimsave(output, frames, duration=0.09, loop=0)
print(output, len(frames), "кадров")
