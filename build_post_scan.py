#!/usr/bin/env python3
"""Variante « scan » du post conférence Scanner intra-oral.

Format 1080x1350, 30 i/s, 7 s. Clin d'œil au sujet de la conférence :
le visuel est numérisé comme une empreinte intra-orale. Sur un écran
sombre quadrillé où l'on devine à peine l'affiche, une ligne laser cyan
balaie l'image de haut en bas ; juste derrière elle, le visuel apparaît
d'abord en contours lumineux (modèle numérique), puis prend ses vraies
couleurs. Ensuite : reflet sur les encarts, LED du scanner qui pulse,
léger travelling arrière, et l'image se pose sur le visuel d'origine.
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter

from build_post_anime import (H, LAYERS, SRC, W, FPS,
                              build_layers, led_glow, prog, shine)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "video/conference-sidi-mani-scan.mp4")
DUR = 7.0
CYAN = np.array([70, 225, 255], np.float32)
NAVY = np.array([10, 12, 30], np.float32)
SCAN_T, SCAN_D = 0.45, 2.3            # balayage : début, durée
SCAN_FROM, SCAN_TO = -40, H + 300     # la bande de révélation sort de l'image
WIRE, REVEAL = 70, 280                # contours seuls, puis passage à la couleur (px)
SHINE = [("pill", 3.15), ("conf", 3.4)]
ZOOM0, SETTLE = 1.05, 6.4


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ease_in_out(p):
    return 0.5 - 0.5 * math.cos(math.pi * p)


def prepare(final):
    img = np.asarray(final).astype(np.float32)
    # Écran « avant scan » : affiche à peine devinée + quadrillage technique.
    dark = NAVY + img * 0.07
    yy, xx = np.mgrid[0:H, 0:W]
    grid = ((xx % 54 == 0) | (yy % 54 == 0)).astype(np.float32) * 0.10
    dark += grid[..., None] * CYAN
    # Contours lumineux = rendu « modèle numérique ».
    edges = final.convert("L").filter(ImageFilter.FIND_EDGES)
    edges = edges.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.8))
    e = np.clip(np.asarray(edges).astype(np.float32) / 90.0, 0, 1)
    wire = dark + e[..., None] * CYAN * 1.1
    return img, dark, wire


def scan_frame(t, img, dark, wire):
    p = prog(t, SCAN_T, SCAN_D)
    line = SCAN_FROM + (SCAN_TO - SCAN_FROM) * ease_in_out(p)
    y = np.arange(H, dtype=np.float32)
    d = (line - y)[:, None, None]                      # > 0 : déjà scanné
    w_wire = (d >= 0).astype(np.float32)
    w_col = smooth((d - WIRE) / (REVEAL - WIRE))
    out = dark * (1 - w_wire) + wire * (w_wire - w_col) + img * w_col
    if 0 < p < 1:
        # Ligne laser : cœur fin + halo, plus intense au centre de l'image.
        dy = (y - line)[:, None]
        x = np.arange(W, dtype=np.float32)[None, :]
        across = 0.55 + 0.45 * np.cos((x / W - 0.5) * math.pi) ** 2
        glow = (np.exp(-dy ** 2 / (2 * 3.0 ** 2)) * 1.4
                + np.exp(-dy ** 2 / (2 * 38.0 ** 2)) * 0.35
                + np.exp(-np.clip(dy, None, 0) ** 2 / (2 * 160.0 ** 2)) * (dy < 0) * 0.10)
        out = out + (glow * across)[..., None] * CYAN
        core = np.exp(-dy ** 2 / (2 * 1.2 ** 2)) * across
        out = out + core[..., None] * 160
    return np.clip(out, 0, 255).astype(np.uint8)


def frame(t, img, dark, wire, layers, final):
    canvas = Image.fromarray(scan_frame(t, img, dark, wire)).convert("RGBA")
    boxes = {L["name"]: L["box"] for L in LAYERS}
    for name, start in SHINE:
        shine(canvas, layers[name], boxes[name], t, start)
    if t > SCAN_T + SCAN_D:
        led_glow(canvas, t)
    out = canvas.convert("RGB")
    z = 1 + (ZOOM0 - 1) * (1 - (1 - (1 - min(1.0, t / SETTLE)) ** 3))
    if z > 1.0005:
        cw, ch = W / z, H / z
        out = out.resize((W, H), Image.BICUBIC,
                         box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))
    f = prog(t, SETTLE, 0.4)
    if f > 0:
        out = Image.blend(out, final, f)
    return out


def main():
    final = Image.open(SRC).convert("RGB")
    img, dark, wire = prepare(final)
    _, layers, _ = build_layers(final)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", OUT],
        stdin=subprocess.PIPE)
    for i in range(int(DUR * FPS)):
        ff.stdin.write(frame(i / FPS, img, dark, wire, layers, final).tobytes())
    ff.stdin.close()
    if ff.wait():
        sys.exit("ffmpeg a échoué")
    print(f"→ {OUT} ({DUR:.1f}s)")


if __name__ == "__main__":
    main()
