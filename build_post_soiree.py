#!/usr/bin/env python3
"""Anime le visuel « atmosphere – Sugah Lewis » (soirée LOMA, 28/11/26).

Format 1080x1350 (post 4:5), 30 i/s, 7 s. Effet « soirée » calé sur un
tempo de 120 BPM (un temps toutes les 0,5 s) :
  - le fond granuleux s'allume et « respire » sur chaque temps, avec un
    halo orange qui pulse à gauche ;
  - le logo tourne en apparaissant, « atmosphere » se dévoile ;
  - la photo s'ouvre comme un obturateur, avec un flash, puis recule
    lentement ;
  - SUGAH puis LEWIS claquent sur les temps (léger tremblement à l'impact),
    la date tombe chiffre par chiffre, puis LOMA ;
  - une traînée de lumière traverse l'image, et tout se pose exactement sur
    le visuel d'origine.
Les calques sont extraits de l'image finale : textes blancs isolés par
seuil, fond reconstruit sous les textes et la photo, grain recréé.
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "posts/atmosphere-sugah-lewis.png")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "video/atmosphere-sugah-lewis.mp4")
W, H, FPS, DUR = 1080, 1350, 30, 7.0

BEAT0, BEAT = 0.5, 0.5                  # premier temps, durée d'un temps (120 BPM)
PHOTO = (240, 295, 840, 854)
GLOW = (0, 700)                         # halo orange, bord gauche
ORANGE = np.array([255, 110, 40], np.float32)
# Éléments blancs : boîte, début (s), durée (s), effet.
TEXTS = [
    dict(name="icone",  box=(490, 62, 590, 142),   t=0.5, d=0.7,  fx="spin"),
    dict(name="mot",    box=(298, 140, 782, 258),  t=0.75, d=0.6, fx="wipe"),
    dict(name="sugah",  box=(672, 752, 954, 832),  t=1.5, d=0.28, fx="slam"),
    dict(name="lewis",  box=(672, 834, 954, 902),  t=2.0, d=0.28, fx="slam"),
    dict(name="d28",    box=(314, 960, 404, 1026), t=2.5, d=0.3,  fx="drop"),
    dict(name="sep1",   box=(446, 960, 468, 1026), t=2.6, d=0.4,  fx="fade"),
    dict(name="d11",    box=(508, 960, 563, 1026), t=3.0, d=0.3,  fx="drop"),
    dict(name="sep2",   box=(606, 960, 628, 1026), t=3.1, d=0.4,  fx="fade"),
    dict(name="d26",    box=(668, 960, 760, 1026), t=3.5, d=0.3,  fx="drop"),
    dict(name="loma",   box=(456, 1240, 624, 1300), t=4.0, d=0.6, fx="rise"),
]
PHOTO_T, PHOTO_D = 1.0, 0.5             # ouverture de la photo
PHOTO_Z0 = 1.12                         # zoom de départ dans la photo
STREAK_T, STREAK_D = 4.4, 1.1           # traînée de lumière
PULSE_END = 5.9                         # les pulsations s'éteignent avant la fin
SETTLE = 6.4


def ease_out(p):
    return 1 - (1 - p) ** 3


def ease_back(p, s=1.7):
    p -= 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2


def prog(t, start, dur):
    return min(1.0, max(0.0, (t - start) / dur))


def pulse(t):
    """1 sur le temps, puis décroît vite (kick)."""
    if t < BEAT0:
        return 0.0
    since = (t - BEAT0) % BEAT
    fade = max(0.0, min(1.0, (PULSE_END - t) / 0.6))
    return math.exp(-since / 0.12) * fade


def _box(a, r, axis):
    """Moyenne glissante de rayon r (bords répétés)."""
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / (2 * r + 1)


def blur(a, s):
    """Flou gaussien approché (3 passes de flou moyen) sur un tableau 2D."""
    r = max(1, int(round(s * 0.87)))
    a = a.astype(np.float64)
    for _ in range(3):
        a = _box(_box(a, r, 0), r, 1)
    return a.astype(np.float32)


def fill_holes(img, valid, scales):
    """Bouche les trous par convolution normalisée, de la plus fine à la plus large."""
    out = img.copy()
    todo = ~valid
    for s in scales:
        w = blur(valid.astype(np.float32), s)
        ok = todo & (w > 0.25)
        for c in range(3):
            v = blur(img[..., c] * valid, s) / np.maximum(w, 1e-6)
            out[..., c][ok] = v[ok]
        todo &= ~ok
    return out


def coons(img, box):
    """Remplit un rectangle en interpolant depuis ses quatre bords (patch de Coons)."""
    x0, y0, x1, y1 = box
    out = img.copy()
    L, R = img[y0:y1, x0 - 3], img[y0:y1, x1 + 2]          # (h, 3)
    T, B = img[y0 - 3, x0:x1], img[y1 + 2, x0:x1]          # (w, 3)
    c00, c10, c01, c11 = img[y0 - 3, x0 - 3], img[y0 - 3, x1 + 2], img[y1 + 2, x0 - 3], img[y1 + 2, x1 + 2]
    # Bords lissés : sinon le grain du bord crée des rayures dans tout le rectangle.
    L, R, T, B = (_box(_box(e, 20, 0), 20, 0) for e in (L, R, T, B))
    u = np.linspace(0, 1, x1 - x0)[None, :, None]
    v = np.linspace(0, 1, y1 - y0)[:, None, None]
    L, R, T, B = L[:, None], R[:, None], T[None], B[None]
    out[y0:y1, x0:x1] = ((1 - u) * L + u * R + (1 - v) * T + v * B
                         - ((1 - u) * (1 - v) * c00 + u * (1 - v) * c10
                            + (1 - u) * v * c01 + u * v * c11))
    return out


def build(final):
    a = np.asarray(final).astype(np.float32)
    mn = a.min(axis=2)
    core = Image.fromarray(((mn > 200) * 255).astype(np.uint8))
    near = np.asarray(core.filter(ImageFilter.MaxFilter(7))) > 0
    text_a = np.clip((mn - 110) / 130.0, 0, 1) * near

    texts = {}
    for T in TEXTS:
        x0, y0, x1, y1 = T["box"]
        layer = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
        layer[..., :3] = 255
        layer[..., 3] = (text_a[y0:y1, x0:x1] * 255).astype(np.uint8)
        texts[T["name"]] = Image.fromarray(layer, "RGBA")

    # Photo sans les lettres qui la chevauchent.
    x0, y0, x1, y1 = PHOTO
    ph = a[y0:y1, x0:x1]
    hole = np.asarray(Image.fromarray((text_a[y0:y1, x0:x1] > 0.02).astype(np.uint8) * 255)
                      .filter(ImageFilter.MaxFilter(5))) > 0
    photo = Image.fromarray(np.clip(fill_holes(ph, ~hole, (3, 8, 20)), 0, 255).astype(np.uint8))

    # Fond sans textes ni photo, grain recréé.
    valid = ~(np.asarray(Image.fromarray((text_a > 0.02).astype(np.uint8) * 255)
                         .filter(ImageFilter.MaxFilter(9))) > 0)
    m = 14
    valid[y0 - m:y1 + m, x0 - m:x1 + m] = False
    vw = np.maximum(blur(valid.astype(np.float32), 6), 1e-3)
    smooth = np.stack([blur(a[..., c] * valid, 6) / vw for c in range(3)], axis=2)
    grain_std = float((a - smooth)[valid].std())
    base = fill_holes(smooth, valid, (12, 40, 120, 300))
    base = coons(base, (x0 - m, y0 - m, x1 + m, y1 + m))
    base = np.stack([blur(base[..., c], 4) for c in range(3)], axis=2)
    rng = np.random.default_rng(3)
    grain = rng.normal(0, grain_std, (H, W, 1)).astype(np.float32)
    bg = np.where(valid[..., None], a, base + grain)
    return bg, texts, photo


def place_text(canvas, piece, T, t):
    p = prog(t, T["t"], T["d"])
    if p <= 0:
        return 0.0
    x0, y0, x1, y1 = T["box"]
    w, h = piece.size
    e = ease_out(p)
    alpha, dx, dy, fx = min(1.0, p * 2), 0, 0, T["fx"]
    if fx == "spin":
        s = max(0.05, ease_back(p))
        piece = piece.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
        piece = piece.rotate(-180 * (1 - e), Image.BICUBIC, expand=True)
    elif fx == "slam":
        s = 1 + 0.8 * (1 - e)
        piece = piece.resize((int(w * s), int(h * s)), Image.BICUBIC)
        alpha = min(1.0, p * 3)
    elif fx == "drop":
        dy = int(-26 * (1 - ease_back(p)))
    elif fx == "rise":
        dy = int(18 * (1 - e))
    elif fx == "wipe":
        edge = e * (w + 100)
        ramp = np.clip((edge - np.arange(w)) / 100.0, 0, 1)
        al = np.asarray(piece.getchannel("A")).astype(np.float32) * ramp[None, :]
        piece = piece.copy()
        piece.putalpha(Image.fromarray(al.astype(np.uint8)))
        dx, alpha = int(-30 * (1 - e)), 1.0
    if alpha < 1:
        piece = piece.copy()
        piece.putalpha(piece.getchannel("A").point(lambda v: int(v * alpha)))
    pw, ph = piece.size
    cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
    canvas.alpha_composite(piece, (int(round(cx - pw / 2)), int(round(cy - ph / 2))))
    # Intensité de l'impact (pour le tremblement) juste à l'arrivée d'un « slam ».
    return max(0.0, 1 - (t - T["t"] - T["d"]) / 0.2) if fx == "slam" and p >= 1 else 0.0


def place_photo(canvas, photo, t):
    p = prog(t, PHOTO_T, PHOTO_D)
    if p <= 0:
        return
    x0, y0, x1, y1 = PHOTO
    w, h = x1 - x0, y1 - y0
    z = PHOTO_Z0 + (1 - PHOTO_Z0) * ease_out(prog(t, PHOTO_T, SETTLE - PHOTO_T))
    cw, ch = w / z, h / z
    img = photo.resize((w, h), Image.BICUBIC,
                       box=((w - cw) / 2, (h - ch) / 2, (w + cw) / 2, (h + ch) / 2))
    arr = np.asarray(img).astype(np.float32)
    flash = max(0.0, 1 - prog(t, PHOTO_T + PHOTO_D * 0.4, 0.35)) * 0.6
    arr = arr + (255 - arr) * flash
    # Obturateur : s'ouvre depuis la ligne médiane.
    half = h / 2 * ease_out(p)
    yy = np.arange(h)[:, None]
    m = np.clip(half - np.abs(yy - h / 2) + 1, 0, 1)
    rgba = np.dstack([np.clip(arr, 0, 255), np.broadcast_to(m * 255, (h, w))]).astype(np.uint8)
    canvas.alpha_composite(Image.fromarray(rgba, "RGBA"), (x0, y0))


def background(bg, t, glow_map, streak_x):
    on = ease_out(prog(t, 0, 0.6))
    k = pulse(t)
    out = bg * (on * (1 + 0.07 * k))
    out = out + glow_map[..., None] * ORANGE * (0.10 + 0.30 * k) * on
    p = prog(t, STREAK_T, STREAK_D)
    if 0 < p < 1:
        pos = -400 + p * (W + 800)
        band = np.exp(-((streak_x - pos) / 140.0) ** 2) * math.sin(math.pi * p) * 0.35
        out = out + band[..., None] * np.array([255, 150, 120], np.float32)
    return out


def frame(t, bg, texts, photo, glow_map, streak_x, final):
    canvas = Image.fromarray(np.clip(background(bg, t, glow_map, streak_x), 0, 255)
                             .astype(np.uint8)).convert("RGBA")
    place_photo(canvas, photo, t)
    shake = 0.0
    for T in TEXTS:
        shake = max(shake, place_text(canvas, texts[T["name"]], T, t))
    img = canvas.convert("RGB")

    z = max(1 + 0.012 * pulse(t) * (t >= PHOTO_T), 1 + 0.016 * shake)  # marge pour trembler
    dx = 5 * shake * math.sin(t * 90)
    dy = 4 * shake * math.cos(t * 70)
    if z > 1.0005 or shake > 0:
        cw, ch = W / z, H / z
        bx, by = (W - cw) / 2 + dx, (H - ch) / 2 + dy
        img = img.resize((W, H), Image.BICUBIC, box=(bx, by, bx + cw, by + ch))
    f = prog(t, SETTLE, 0.4)
    if f > 0:
        img = Image.blend(img, final, f)
    return img


def main():
    final = Image.open(SRC).convert("RGB").resize((W, H), Image.LANCZOS)
    bg, texts, photo = build(final)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    glow_map = np.exp(-(((xx - GLOW[0]) / 380) ** 2 + ((yy - GLOW[1]) / 300) ** 2))
    streak_x = xx + (yy - H / 2) * 0.45
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", OUT],
        stdin=subprocess.PIPE)
    for i in range(int(DUR * FPS)):
        ff.stdin.write(frame(i / FPS, bg, texts, photo, glow_map, streak_x, final).tobytes())
    ff.stdin.close()
    if ff.wait():
        sys.exit("ffmpeg a échoué")
    print(f"→ {OUT} ({DUR:.1f}s)")


if __name__ == "__main__":
    main()
