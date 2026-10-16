#!/usr/bin/env python3
"""Anime le visuel « Conférence Scanner intra-oral » (journée d'inauguration).

Format 1080x1350 (post Instagram / Facebook / LinkedIn), 30 i/s, ~7 s.
Le visuel est découpé en calques à partir de l'image finale : le fond
dégradé violet est reconstruit ligne par ligne, chaque élément est isolé
par différence avec ce fond. Les éléments entrent ensuite l'un après
l'autre (glissés, pops, balayage), un reflet lumineux traverse les
encarts, la LED du scanner pulse, puis l'image se pose exactement sur le
visuel d'origine.
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "posts/conference-sidi-mani.webp")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "video/conference-sidi-mani.mp4")
W, H, FPS, DUR = 1080, 1350, 30, 7.0
SPLIT = 1009          # début du bandeau blanc du bas

# Éléments du haut : boîte (x0, y0, x1, y1), début (s), durée (s), effet.
LAYERS = [
    dict(name="logo",   box=(15, 25, 425, 185),    t=0.15, d=0.7, fx="pop"),
    dict(name="pill",   box=(480, 55, 1040, 150),  t=0.35, d=0.7, fx="right"),
    dict(name="nom",    box=(30, 188, 770, 262),   t=0.55, d=0.8, fx="wipe"),
    dict(name="interv", box=(30, 262, 285, 314),   t=0.85, d=0.6, fx="up"),
    dict(name="itero",  box=(45, 328, 247, 426),   t=1.05, d=0.6, fx="pop"),
    dict(name="chariot", box=(78, 478, 312, SPLIT), t=0.95, d=0.9, fx="rise"),
    dict(name="camera", box=(338, 488, 432, SPLIT), t=1.15, d=0.9, fx="rise"),
    dict(name="photo",  box=(688, 583, W, SPLIT),  t=1.25, d=0.9, fx="right"),
    dict(name="conf",   box=(495, 274, 1040, 480), t=1.55, d=0.8, fx="right"),
    dict(name="horloge", box=(570, 480, 644, 558), t=1.95, d=0.7, fx="spin"),
    dict(name="heure",  box=(644, 480, 925, 560),  t=2.05, d=0.6, fx="left"),
]
PANEL_T, PANEL_D = 2.1, 0.8          # bandeau du bas qui remonte
SHINE = [("pill", 3.0), ("conf", 3.25)]  # reflet lumineux (calque, début)
SHINE_D = 0.8
LED = (385, 779)                     # anneau lumineux du scanner
ZOOM0 = 1.035                        # léger dézoom global sur toute la durée
SETTLE = 6.4                         # fondu final vers l'image d'origine


def ease_out(p):
    return 1 - (1 - p) ** 3


def ease_back(p, s=1.7):
    p -= 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2


def prog(t, start, dur):
    return min(1.0, max(0.0, (t - start) / dur))


def build_layers(img):
    a = np.asarray(img).astype(np.float32)
    rows = np.median(a[:SPLIT, 0:18], axis=1)
    rows = np.vstack([rows, np.repeat(rows[-1:], H - SPLIT, axis=0)])
    bg = np.repeat(rows[:, None, :], W, axis=1)
    diff = np.abs(a - bg).max(axis=2)
    diff[SPLIT:] = 0
    mask = Image.fromarray(((diff > 4) * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.2))

    rgba = img.convert("RGBA")
    layers = {}
    for L in LAYERS:
        x0, y0, x1, y1 = L["box"]
        piece = rgba.crop(L["box"])
        piece.putalpha(mask.crop(L["box"]))
        layers[L["name"]] = piece
    panel = rgba.crop((0, SPLIT, W, H))
    bg_img = Image.fromarray(bg.clip(0, 255).astype(np.uint8)).convert("RGBA")
    return bg_img, layers, panel


def place(canvas, piece, box, t, L):
    p = prog(t, L["t"], L["d"])
    if p <= 0:
        return
    x0, y0, x1, y1 = box
    e = ease_out(p)
    alpha = min(1.0, p * 2.2)
    dx = dy = 0
    fx = L["fx"]
    if fx == "right":
        dx = int(140 * (1 - e))
    elif fx == "left":
        dx = int(-80 * (1 - e))
    elif fx == "up":
        dy = int(30 * (1 - e))
    elif fx == "rise":
        dy = int(220 * (1 - e))
    elif fx in ("pop", "spin"):
        s = max(0.05, 0.4 + 0.6 * ease_back(p))
        w, h = piece.size
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        piece = piece.resize((nw, nh), Image.BICUBIC)
        if fx == "spin":
            piece = piece.rotate(-120 * (1 - e), Image.BICUBIC)
        dx, dy = (w - nw) // 2, (h - nh) // 2
    elif fx == "wipe":
        w, h = piece.size
        edge = e * (w + 120)
        ramp = np.clip((edge - np.arange(w)) / 120.0, 0, 1)
        a = np.asarray(piece.getchannel("A")).astype(np.float32) * ramp[None, :]
        piece = piece.copy()
        piece.putalpha(Image.fromarray(a.astype(np.uint8)))
        dx = int(-40 * (1 - e))
        alpha = 1.0
    if alpha < 1:
        piece = piece.copy()
        piece.putalpha(piece.getchannel("A").point(lambda v: int(v * alpha)))
    canvas.alpha_composite(piece, (x0 + dx, y0 + dy))


def shine(canvas, piece, box, t, start):
    p = prog(t, start, SHINE_D)
    if p <= 0 or p >= 1:
        return
    w, h = piece.size
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    pos = -h + p * (w + 2 * h)
    band = np.exp(-(((xx + yy * 0.5) - pos) / 38.0) ** 2) * 0.55
    a = band * np.asarray(piece.getchannel("A")) / 255.0
    glow = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    glow.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
    canvas.alpha_composite(glow, box[:2])


def led_glow(canvas, t):
    start = 2.0
    if t < start:
        return
    k = 0.5 + 0.5 * math.sin((t - start) * 2 * math.pi / 1.4 - math.pi / 2)
    k *= min(1.0, (t - start) / 0.5) * max(0.0, min(1.0, (SETTLE + 0.4 - t) / 0.5))
    if k <= 0.01:
        return
    r = 70
    xx, yy = np.meshgrid(np.arange(2 * r), np.arange(2 * r))
    d2 = ((xx - r) / 1.0) ** 2 + ((yy - r) / 0.45) ** 2
    a = np.exp(-d2 / (2 * 22 ** 2)) * 0.75 * k
    glow = Image.new("RGBA", (2 * r, 2 * r), (70, 230, 255, 0))
    glow.putalpha(Image.fromarray((a * 255).astype(np.uint8)))
    canvas.alpha_composite(glow, (LED[0] - r, LED[1] - r))


def frame(t, bg, layers, panel, final):
    canvas = bg.copy()
    for L in LAYERS:
        place(canvas, layers[L["name"]], L["box"], t, L)
    boxes = {L["name"]: L["box"] for L in LAYERS}
    for name, start in SHINE:
        shine(canvas, layers[name], boxes[name], t, start)
    led_glow(canvas, t)
    # Les éléments qui montent ne débordent pas sur la zone du bandeau.
    canvas.paste(bg.crop((0, SPLIT, W, H)), (0, SPLIT))

    p = prog(t, PANEL_T, PANEL_D)
    if p > 0:
        y = SPLIT + int((H - SPLIT) * (1 - ease_out(p)))
        canvas.alpha_composite(panel, (0, y))

    img = canvas.convert("RGB")
    z = 1 + (ZOOM0 - 1) * (1 - ease_out(min(1.0, t / SETTLE)))
    if z > 1.0005:
        cw, ch = W / z, H / z
        img = img.resize((W, H), Image.BICUBIC,
                         box=((W - cw) / 2, (H - ch) / 2, (W + cw) / 2, (H + ch) / 2))
    f = prog(t, SETTLE, 0.4)
    if f > 0:
        img = Image.blend(img, final, f)
    return img


def main():
    final = Image.open(SRC).convert("RGB")
    assert final.size == (W, H), final.size
    bg, layers, panel = build_layers(final)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", OUT],
        stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    for i in range(n):
        ff.stdin.write(frame(i / FPS, bg, layers, panel, final).tobytes())
    ff.stdin.close()
    if ff.wait():
        sys.exit("ffmpeg a échoué")
    print(f"→ {OUT} ({DUR:.1f}s)")


if __name__ == "__main__":
    main()
