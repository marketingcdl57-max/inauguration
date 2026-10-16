#!/usr/bin/env python3
"""Anime l'affiche « Travaux pratiques — Collage » (atelier G-CEM ONE™ / GC).

Format 4:5 1080x1350 (post Instagram / Facebook / LinkedIn), 30 i/s, 10 s.

L'affiche est découpée en calques : chaque élément (logo, pastille,
titre, badge GC, horaires, produit, bandeau infos) est extrait par
différence avec une « plaque » de fond reconstruite (remplissage
harmonique des zones occupées). Chaque calque vaut donc
(affiche - plaque) et s'ajoute au fond : une fois tous les éléments
posés, l'image finale est exactement l'affiche d'origine.

Séquence : fondu d'ouverture avec léger recul de caméra, entrées
décalées des éléments, puis vie de l'image (reflet mint sur « Collage »,
pulsation du badge horaire, produit qui flotte doucement).
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H, FPS, DUR = 1080, 1350, 30, 10.0
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "assets/post-collage.webp")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "video/post-collage-anime.mp4")
K = W / 1600                       # coordonnées ci-dessous en pixels de l'affiche 1600x2000
MINT = np.array([94, 234, 190], np.float32)

BOXES = {
    "logo":    (85, 75, 610, 212),
    "pill":    (880, 95, 1510, 192),
    "tp":      (90, 315, 645, 362),
    "collage": (88, 385, 1030, 667),
    "part":    (1145, 396, 1508, 434),
    "gc":      (1102, 438, 1514, 576),
    "sub":     (90, 680, 1355, 808),
    "anim":    (90, 832, 580, 952),
    "time":    (996, 812, 1524, 966),
    "halo":    (1260, 966, 1524, 995),  # bas du halo du badge horaire, à droite du produit
    "prod":    (340, 966, 1260, 1515),
}
PANEL_Y = 1515                     # début du bandeau blanc d'informations
TIME_PILL = (1013, 830, 1505, 955) # badge « 14h30 – 15h » (sans le halo)
DOT = (933, 143)                   # point de la pastille « Journée d'inauguration »


def box(b):
    return tuple(int(round(v * K)) for v in b)


# --- courbes -----------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, start, dur):
    return clamp((t - start) / dur)


def out_cubic(p):
    return 1 - (1 - p) ** 3


def out_expo(p):
    return 1.0 if p >= 1 else 1 - 2 ** (-10 * p)


def out_back(p, s=1.6):
    p -= 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2


def in_out(p):
    return 3 * p * p - 2 * p ** 3


# --- préparation ---------------------------------------------------------------
def harmonic_fill(img, mask, iters=400):
    """Remplit la zone masquée par interpolation lisse depuis ses bords."""
    small_w, small_h = W // 8, H // 8
    m_small = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (small_w, small_h), Image.BILINEAR)) > 0
    s = np.array(Image.fromarray(img.astype(np.uint8)).resize(
        (small_w, small_h), Image.BOX)).astype(np.float32)

    def relax(a, m, n):
        for _ in range(n):
            p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
            avg = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4
            a[m] = avg[m]
        return a

    s[m_small] = s[~m_small].mean(0)   # départ neutre : converge bien plus vite
    s = relax(s, m_small, 6000)
    up = np.stack([np.array(Image.fromarray(s[..., c]).resize((W, H), Image.BICUBIC))
                   for c in range(3)], -1)
    out = img.copy()
    out[mask] = up[mask]
    return relax(out, mask, iters)


def prepare():
    poster = np.array(Image.open(SRC).convert("RGB").resize((W, H), Image.LANCZOS)).astype(np.float32)
    py = int(round(PANEL_Y * K))
    mask = np.zeros((H, W), bool)
    for b in BOXES.values():
        x0, y0, x1, y1 = box(b)
        mask[y0:y1, x0:x1] = True
    mask[py:] = True
    plate = harmonic_fill(poster, mask)
    layers = {}
    for name, b in BOXES.items():
        x0, y0, x1, y1 = box(b)
        layers[name] = (x0, y0, poster[y0:y1, x0:x1] - plate[y0:y1, x0:x1])
    panel = poster[py:].copy()
    return poster, plate, layers, panel, py


# --- composition ---------------------------------------------------------------
def add_layer(frame, layer, dx=0.0, dy=0.0, scale=1.0, alpha=1.0, clip=False, wipe=None):
    """Ajoute un calque (différence) transformé autour de son centre.

    clip : limite le calque à sa zone d'origine (révélation par masque).
    wipe : fraction 0..1 révélée de gauche à droite.
    """
    if alpha <= 0:
        return
    x0, y0, d = layer
    h, w = d.shape[:2]
    if dx == dy == 0 and scale == 1 and wipe is None:
        frame[y0:y0 + h, x0:x0 + w] += alpha * d
        return
    m = 40 if not clip else 0
    cx, cy = w / 2, h / 2
    # zone de destination (coordonnées locales du calque)
    lx0, ly0 = -m - int(abs(dx)) - 2, -m - int(abs(dy)) - 2
    lx1, ly1 = w + m + int(abs(dx)) + 2, h + m + int(abs(dy)) + 2
    if clip:
        lx0, ly0, lx1, ly1 = 0, 0, w, h
    gx0, gy0 = max(0, x0 + lx0), max(0, y0 + ly0)
    gx1, gy1 = min(W, x0 + lx1), min(H, y0 + ly1)
    if gx1 <= gx0 or gy1 <= gy0:
        return
    ox, oy = gx0 - x0, gy0 - y0
    a = 1 / scale
    # source = (dest - centre - décalage) / échelle + centre
    data = (a, 0, cx - (cx + dx) * a + ox * a, 0, a, cy - (cy + dy) * a + oy * a)
    size = (gx1 - gx0, gy1 - gy0)
    out = np.stack([np.array(Image.fromarray(d[..., c]).transform(
        size, Image.AFFINE, data, Image.BILINEAR, fillcolor=0)) for c in range(3)], -1)
    if wipe is not None:
        xs = np.arange(size[0]) + ox
        edge = wipe * (w + 60) - 30
        out *= np.clip((edge - xs) / 30 + 0.5, 0, 1)[None, :, None]
    frame[gy0:gy1, gx0:gx1] += alpha * out


def shine(frame, b, t0, dur, t, glyph, color, strength):
    """Reflet diagonal qui balaie la zone b, limité aux pixels de glyph."""
    p = (t - t0) / dur
    if not 0 <= p <= 1:
        return
    x0, y0, x1, y1 = box(b)
    h, w = y1 - y0, x1 - x0
    yy, xx = np.mgrid[0:h, 0:w]
    diag = xx + 0.45 * yy
    span = w + 0.45 * h
    c = -0.25 * span + in_out(p) * 1.5 * span
    band = np.exp(-((diag - c) / (0.07 * span)) ** 2) * glyph * strength
    reg = frame[y0:y1, x0:x1]
    reg += band[..., None] * (color - reg)


def ring(img, rect, grow, alpha, radius, width):
    """Contour arrondi mint qui s'étend autour de rect (RGBA dessiné sur img)."""
    if alpha <= 0:
        return
    x0, y0, x1, y1 = (v * K for v in rect)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).rounded_rectangle(
        (x0 - grow, y0 - grow, x1 + grow, y1 + grow), radius=radius * K + grow,
        outline=(94, 234, 190, int(255 * alpha)), width=width)
    img.alpha_composite(ov.filter(ImageFilter.GaussianBlur(0.8)))


def render(poster, plate, layers, panel, py):
    # masques pour les reflets
    x0, y0, x1, y1 = box(BOXES["collage"])
    glyph_collage = np.clip((poster[y0:y1, x0:x1].mean(-1) - 120) / 90, 0, 1)
    x0, y0, x1, y1 = box(BOXES["time"])
    reg = poster[y0:y1, x0:x1]
    glyph_time = np.clip((reg[..., 1] - reg[..., 0] - 60) / 40, 0, 1)
    white = np.array([255, 255, 255], np.float32)

    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)

    n = int(DUR * FPS)
    for i in range(n):
        t = i / FPS
        f = plate.copy()

        def enter(name, start, dur=0.7, **kw):
            p = prog(t, start, dur)
            add_layer(f, layers[name], **{k: v(p) for k, v in kw.items()},
                      alpha=clamp(p * 1.6))

        enter("logo", 0.35, 0.8, dx=lambda p: -50 * (1 - out_cubic(p)))
        enter("pill", 0.5, 0.8, dx=lambda p: 50 * (1 - out_cubic(p)))
        p = prog(t, 0.9, 0.8)
        add_layer(f, layers["tp"], wipe=out_cubic(p), alpha=1.0 if p > 0 else 0)
        p = prog(t, 1.1, 0.9)
        add_layer(f, layers["collage"], dy=110 * K * (1 - out_expo(p)), clip=True,
                  alpha=1.0 if p > 0 else 0)
        enter("part", 1.55, 0.6, dy=lambda p: -14 * (1 - out_cubic(p)))
        enter("gc", 1.7, 0.7, scale=lambda p: 0.82 + 0.18 * out_back(p))
        enter("sub", 2.0, 0.8, dy=lambda p: 26 * (1 - out_cubic(p)))
        enter("anim", 2.35, 0.8, dx=lambda p: -40 * (1 - out_cubic(p)))
        enter("time", 2.55, 0.7, scale=lambda p: 0.7 + 0.3 * out_back(p))
        enter("halo", 2.55, 0.7)
        # produit : monte et se pose, puis flotte doucement
        p = prog(t, 2.85, 1.1)
        float_amp = 3.5 * in_out(prog(t, 3.9, 0.8))
        bob = float_amp * math.sin(2 * math.pi * (t - 3.9) / 3.2)
        add_layer(f, layers["prod"], dy=70 * (1 - out_cubic(p)) + bob,
                  scale=0.94 + 0.06 * out_cubic(p), alpha=clamp(p * 1.4))

        # bandeau d'informations qui monte
        p = out_cubic(prog(t, 3.5, 0.9))
        top = py + int(round((H - py) * (1 - p)))
        if top < H:
            f[top:] = panel[:H - top]

        # reflets
        shine(f, BOXES["collage"], 4.7, 1.1, t, glyph_collage, MINT, 0.9)
        shine(f, BOXES["time"], 5.4, 0.9, t, glyph_time, white, 0.55)
        shine(f, BOXES["collage"], 8.0, 1.1, t, glyph_collage, MINT, 0.9)

        img = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert("RGBA")
        # pulsations : badge horaire et point de la pastille
        for t0 in (5.5, 7.6):
            q = prog(t, t0, 1.2)
            if 0 < q < 1:
                ring(img, TIME_PILL, 26 * out_cubic(q), 0.8 * (1 - q), 62, 3)
        q = ((t - 1.4) % 1.6) / 1.2 if t > 1.4 else 1
        if q < 1:
            dx_, dy_ = DOT[0] * K, DOT[1] * K
            r = 5 + 14 * out_cubic(q)
            ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(ov).ellipse((dx_ - r, dy_ - r, dx_ + r, dy_ + r),
                                       outline=(94, 234, 190, int(200 * (1 - q))), width=2)
            img.alpha_composite(ov)

        # caméra : léger recul + fondu d'ouverture
        img = img.convert("RGB")
        z = 1 + 0.06 * (1 - in_out(prog(t, 0, 4.0)))
        if z > 1.0001:
            cw, ch = W / z, H / z
            img = img.resize((W, H), Image.BICUBIC,
                             box=((W - cw) / 2, (H - ch) * 0.35, (W + cw) / 2, (H - ch) * 0.35 + ch))
        fade = out_cubic(prog(t, 0, 0.7))
        if fade < 1:
            img = Image.blend(Image.new("RGB", (W, H)), img, fade)
        ff.stdin.write(img.tobytes())

    ff.stdin.close()
    if ff.wait():
        sys.exit("ffmpeg a échoué")


if __name__ == "__main__":
    render(*prepare())
    print(OUT)
