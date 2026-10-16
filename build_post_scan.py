#!/usr/bin/env python3
"""Variante « scan » du post conférence Scanner intra-oral.

Format 1080x1350, 30 i/s, 7 s. Un même effet pour plusieurs visuels
(voir POSTS). Clin d'œil au sujet de la conférence :
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
from PIL import Image, ImageDraw, ImageFilter

from build_post_anime import H, W, FPS, prog

ROOT = os.path.dirname(os.path.abspath(__file__))
# Visuels : encarts traversés par le reflet (boîte, début) et anneau LED.
POSTS = {
    "sidi-mani": dict(
        src="posts/conference-sidi-mani.webp",
        out="video/conference-sidi-mani-scan.mp4",
        shine=[((480, 55, 1040, 150), 3.15), ((495, 274, 1040, 480), 3.4)],
        led=(385, 779)),
    "abdel": dict(
        src="posts/conference-abdel.webp",
        out="video/conference-abdel-scan.mp4",
        shine=[((480, 55, 1040, 150), 3.15), ((486, 231, 1036, 430), 3.4)],
        led=(403, 799)),
}
BASE = np.array([118, 34, 130], np.float32)   # violet du fond
DUR = 7.0
CYAN = np.array([70, 225, 255], np.float32)
NAVY = np.array([10, 12, 30], np.float32)
SCAN_T, SCAN_D = 0.45, 2.3            # balayage : début, durée
SCAN_FROM, SCAN_TO = -40, H + 300     # la bande de révélation sort de l'image
WIRE, REVEAL = 70, 280                # contours seuls, puis passage à la couleur (px)
SHINE_D = 0.8
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


def glass_masks(img, post):
    """Masque de chaque encart vitré : rectangle arrondi x opacité du verre."""
    diff = np.clip(np.abs(img - BASE).max(axis=2) / 60.0, 0, 1)
    masks = []
    for box, start in post["shine"]:
        x0, y0, x1, y1 = box
        r = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(r).rounded_rectangle((0, 0, x1 - x0 - 1, y1 - y0 - 1),
                                            radius=(y1 - y0) // 4, fill=255)
        m = np.asarray(r) / 255.0 * diff[y0:y1, x0:x1]
        masks.append((box, start, m))
    return masks


def shine(out, box, start, m, t):
    """Reflet diagonal qui traverse l'encart."""
    p = prog(t, start, SHINE_D)
    if p <= 0 or p >= 1:
        return
    x0, y0, x1, y1 = box
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w]
    pos = -h + p * (w + 2 * h)
    band = np.exp(-(((xx + yy * 0.5) - pos) / 38.0) ** 2) * 0.55 * m
    sub = out[y0:y1, x0:x1]
    out[y0:y1, x0:x1] = sub + (255 - sub) * band[..., None]


def led_glow(out, t, led):
    """Halo cyan qui pulse sur l'anneau lumineux du scanner."""
    start = SCAN_T + SCAN_D
    if t < start:
        return
    k = 0.5 + 0.5 * math.sin((t - start) * 2 * math.pi / 1.4 - math.pi / 2)
    k *= min(1.0, (t - start) / 0.5)
    r = 70
    yy, xx = np.mgrid[0:2 * r, 0:2 * r]
    a = np.exp(-((xx - r) ** 2 + ((yy - r) / 0.45) ** 2) / (2 * 22 ** 2)) * 0.75 * k
    x0, y0 = led[0] - r, led[1] - r
    sub = out[y0:y0 + 2 * r, x0:x0 + 2 * r]
    out[y0:y0 + 2 * r, x0:x0 + 2 * r] = sub + (CYAN - sub) * a[..., None]


def frame(t, img, dark, wire, masks, post, final):
    arr = scan_frame(t, img, dark, wire).astype(np.float32)
    for box, start, m in masks:
        shine(arr, box, start, m, t)
    led_glow(arr, t, post["led"])
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    z = 1 + (ZOOM0 - 1) * (1 - (1 - (1 - min(1.0, t / SETTLE)) ** 3))
    if z > 1.0005:
        cw, ch = W / z, H / z
        out = out.resize((W, H), Image.BICUBIC,
                         box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))
    f = prog(t, SETTLE, 0.4)
    if f > 0:
        out = Image.blend(out, final, f)
    return out


def render(post):
    final = Image.open(os.path.join(ROOT, post["src"])).convert("RGB")
    assert final.size == (W, H), final.size
    img, dark, wire = prepare(final)
    masks = glass_masks(img, post)
    out = os.path.join(ROOT, post["out"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    for i in range(int(DUR * FPS)):
        ff.stdin.write(frame(i / FPS, img, dark, wire, masks, post, final).tobytes())
    ff.stdin.close()
    if ff.wait():
        sys.exit("ffmpeg a échoué")
    print(f"→ {out} ({DUR:.1f}s)")


if __name__ == "__main__":
    # Usage : build_post_scan.py [visuel ...]   (par défaut : tous)
    for key in sys.argv[1:] or POSTS:
        render(POSTS[key])
