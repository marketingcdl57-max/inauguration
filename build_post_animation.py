#!/usr/bin/env python3
"""Anime les affiches des travaux pratiques de la journée d'inauguration.

    python3 build_post_animation.py [collage|lea] [sortie.mp4]

  - collage : « Travaux pratiques — Collage » (atelier G-CEM ONE™ / GC)
  - lea     : « Léa FERRAND — TP Collage », version avec photo

Format 4:5 1080x1350 (post Instagram / Facebook / LinkedIn), 60 i/s,
6 s pour « collage », 4,5 s pour « lea ».

L'affiche est découpée en calques : chaque élément (logo, pastille,
titre, badge GC, horaires, produit, photo...) est extrait par
différence avec une « plaque » de fond reconstruite (remplissage
harmonique des zones occupées). Chaque calque vaut donc
(affiche - plaque) et s'ajoute au fond : une fois tous les éléments
posés, l'image finale est exactement l'affiche d'origine. Le bandeau
blanc d'informations, lui, monte depuis le bas.

Séquence rythmée : tous les éléments sont en place en moins de 2 s
(entrées courtes en ease-out exponentiel, sans rebond), puis vie de
l'image (reflets, pulsations, produit qui flotte doucement).
Version « lea » : révélations par bandeau (nom, photo, logo GC) et flou
de mouvement sur toutes les entrées.
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H, FPS, DUR = 1080, 1350, 60, 6.0
ROOT = os.path.dirname(os.path.abspath(__file__))
K = 1.0                            # échelle affiche source -> vidéo (fixée par l'affiche choisie)
MINT = np.array([94, 234, 190], np.float32)
WHITE = np.array([255, 255, 255], np.float32)


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


def prepare(cfg):
    poster = np.array(Image.open(cfg["src"]).convert("RGB").resize((W, H), Image.LANCZOS)).astype(np.float32)
    py = int(round(cfg["panel_y"] * K))
    mask = np.zeros((H, W), bool)
    for b in cfg["boxes"].values():
        x0, y0, x1, y1 = box(b)
        mask[y0:y1, x0:x1] = True
    mask[py:] = True
    plate = harmonic_fill(poster, mask)
    plate[py:] = plate[py - 1]         # sous le bandeau : prolonge la dernière ligne du fond
    layers = {}
    for name, b in cfg["boxes"].items():
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


def ring(img, rect, grow, alpha, radius, width, color=(94, 234, 190)):
    """Contour arrondi qui s'étend autour de rect (RGBA dessiné sur img)."""
    if alpha <= 0:
        return
    x0, y0, x1, y1 = (v * K for v in rect)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).rounded_rectangle(
        (x0 - grow, y0 - grow, x1 + grow, y1 + grow), radius=radius * K + grow,
        outline=(*color, int(255 * alpha)), width=width)
    img.alpha_composite(ov.filter(ImageFilter.GaussianBlur(0.8)))


def enter(f, layers, t, name, start, dur=0.45, **kw):
    """Entrée d'un calque : chaque kw (dx, dy, scale) est une fonction de la progression."""
    p = prog(t, start, dur)
    add_layer(f, layers[name], **{k: v(p) for k, v in kw.items()}, alpha=clamp(p * 2.5))


def block_reveal(f, L, t, name, start, dur=0.6, color=MINT, **kw):
    """Un bandeau plein balaie le calque de gauche à droite puis se retire en le dévoilant.

    kw : transformation du calque une fois dévoilé (dx, dy, scale).
    """
    p = prog(t, start, dur)
    if p <= 0:
        return
    x0, y0, d = L[name]
    h, w = d.shape[:2]
    if p < 0.5:
        bx0, bx1 = x0, x0 + w * out_expo(p * 2)
    else:
        add_layer(f, L[name], **kw)
        bx0, bx1 = x0 + w * out_expo((p - 0.5) * 2), x0 + w
    a, b = int(bx0), int(math.ceil(bx1))
    if b > a:
        f[y0:y0 + h, a:b] = color


def e(p):
    """Distance restant à parcourir, très nerveux au départ."""
    return 1 - out_expo(p)


def float_bob(t, start):
    return 3.0 * in_out(prog(t, start, 0.6)) * math.sin(2 * math.pi * (t - start) / 2.6)


def glyph_mask(poster, b, fn):
    x0, y0, x1, y1 = box(b)
    return np.clip(fn(poster[y0:y1, x0:x1]), 0, 1)


def dot_pulse(img, t, center, start=0.7):
    q = ((t - start) % 1.2) / 0.9 if t > start else 1
    if q < 1:
        cx, cy = center[0] * K, center[1] * K
        r = 5 + 14 * out_cubic(q)
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(ov).ellipse((cx - r, cy - r, cx + r, cy + r),
                                   outline=(94, 234, 190, int(200 * (1 - q))), width=2)
        img.alpha_composite(ov)


# --- affiche « Travaux pratiques — Collage » ----------------------------------
COLLAGE_BOXES = {                  # pixels de l'affiche 1600x2000
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


def collage_setup(poster):
    return dict(
        collage=glyph_mask(poster, COLLAGE_BOXES["collage"], lambda r: (r.mean(-1) - 120) / 90),
        time=glyph_mask(poster, COLLAGE_BOXES["time"], lambda r: (r[..., 1] - r[..., 0] - 60) / 40))


def collage_frame(t, f, L, ctx):
    enter(f, L, t, "logo", 0.10, dx=lambda p: -70 * e(p))
    enter(f, L, t, "pill", 0.16, dx=lambda p: 70 * e(p))
    p = prog(t, 0.28, 0.40)
    add_layer(f, L["tp"], wipe=out_expo(p), alpha=1.0 if p > 0 else 0)
    p = prog(t, 0.34, 0.55)
    add_layer(f, L["collage"], dy=150 * K * e(p), clip=True, alpha=1.0 if p > 0 else 0)
    enter(f, L, t, "part", 0.52, 0.35, dy=lambda p: -16 * e(p))
    enter(f, L, t, "gc", 0.58, scale=lambda p: 0.88 + 0.12 * out_expo(p))
    enter(f, L, t, "sub", 0.70, dy=lambda p: 28 * e(p))
    enter(f, L, t, "anim", 0.84, dx=lambda p: -60 * e(p))
    enter(f, L, t, "time", 0.92, scale=lambda p: 0.82 + 0.18 * out_back(p, 1.2))
    enter(f, L, t, "halo", 0.92)
    p = prog(t, 1.00, 0.60)
    add_layer(f, L["prod"], dy=90 * e(p) + float_bob(t, 1.6),
              scale=0.93 + 0.07 * out_expo(p), alpha=clamp(p * 2.2))


def collage_shine(t, f, ctx):
    shine(f, COLLAGE_BOXES["collage"], 2.0, 0.7, t, ctx["collage"], MINT, 0.9)
    shine(f, COLLAGE_BOXES["time"], 2.45, 0.55, t, ctx["time"], WHITE, 0.55)
    shine(f, COLLAGE_BOXES["collage"], 4.9, 0.7, t, ctx["collage"], MINT, 0.9)


def collage_overlay(t, img, ctx):
    for t0 in (2.5, 4.4):
        q = prog(t, t0, 0.8)
        if 0 < q < 1:
            ring(img, (1013, 830, 1505, 955), 24 * out_cubic(q), 0.85 * (1 - q), 62, 3)
    dot_pulse(img, t, (933, 143))


# --- affiche « Léa FERRAND — TP Collage » (avec photo) -----------------------
LEA_BOXES = {                      # pixels de l'affiche 1080x1350
    "logo":  (5, 38, 402, 188),
    "pill":  (466, 66, 1020, 154),
    "name":  (15, 212, 482, 288),
    "gc":    (36, 356, 367, 446),
    "card":  (456, 321, 1008, 464),
    "time":  (552, 478, 902, 555),
    "photo": (0, 522, 340, 1021),
    "prod":  (455, 620, 980, 955),
}


def lea_setup(poster):
    white_text = lambda r: (r.min(-1) - 170) / 60
    return dict(name=glyph_mask(poster, LEA_BOXES["name"], white_text),
                time=glyph_mask(poster, LEA_BOXES["time"], white_text))


def lea_frame(t, f, L, ctx):
    enter(f, L, t, "logo", 0.08, dx=lambda p: -70 * e(p))
    enter(f, L, t, "pill", 0.14, dx=lambda p: 90 * e(p))
    block_reveal(f, L, t, "name", 0.22, 0.55)
    # photo : dévoilée par le bandeau, puis léger dézoom
    block_reveal(f, L, t, "photo", 0.32, 0.65,
                 scale=1.08 - 0.08 * out_cubic(prog(t, 0.645, 1.2)))
    enter(f, L, t, "card", 0.50, scale=lambda p: 0.9 + 0.1 * out_expo(p),
          dy=lambda p: 24 * e(p))
    block_reveal(f, L, t, "gc", 0.56, 0.5, WHITE)
    enter(f, L, t, "time", 0.70, dy=lambda p: 26 * e(p))
    p = prog(t, 0.80, 0.60)
    add_layer(f, L["prod"], dy=110 * e(p) + float_bob(t, 1.4),
              scale=0.92 + 0.08 * out_expo(p), alpha=clamp(p * 2.2))


def lea_shine(t, f, ctx):
    shine(f, LEA_BOXES["name"], 1.55, 0.6, t, ctx["name"], MINT, 0.9)
    shine(f, LEA_BOXES["time"], 2.05, 0.5, t, ctx["time"], MINT, 0.8)


def lea_overlay(t, img, ctx):
    q = prog(t, 1.8, 0.8)
    if 0 < q < 1:
        ring(img, (460, 325, 1004, 460), 22 * out_cubic(q), 0.8 * (1 - q), 22, 3, (255, 255, 255))


POSTERS = {
    "collage": dict(src="assets/post-collage.webp", out="video/post-collage-anime.mp4",
                    src_w=1600, boxes=COLLAGE_BOXES, panel_y=1515, panel_in=1.25,
                    setup=collage_setup, frame=collage_frame, shine=collage_shine,
                    overlay=collage_overlay),
    "lea": dict(src="assets/post-lea-ferrand.webp", out="video/post-lea-ferrand-anime.mp4",
                src_w=1080, boxes=LEA_BOXES, panel_y=1022,
                setup=lea_setup, frame=lea_frame, shine=lea_shine, overlay=lea_overlay,
                panel_in=1.0, dur=4.5, motion_blur=1.8),
}


def compose(cfg, t, plate, layers, panel, py, ctx):
    f = plate.copy()
    cfg["frame"](t, f, layers, ctx)

    # bandeau d'informations qui monte
    p = out_expo(prog(t, cfg["panel_in"], 0.55))
    top = py + int(round((H - py) * (1 - p)))
    if top < H:
        f[top:] = panel[:H - top]

    cfg["shine"](t, f, ctx)
    img = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert("RGBA")
    cfg["overlay"](t, img, ctx)

    # caméra : recul rapide + fondu d'ouverture
    img = img.convert("RGB")
    z = 1 + 0.07 * (1 - out_expo(prog(t, 0, 1.6)))
    if z > 1.0001:
        cw, ch = W / z, H / z
        img = img.resize((W, H), Image.BICUBIC,
                         box=((W - cw) / 2, (H - ch) * 0.35, (W + cw) / 2, (H - ch) * 0.35 + ch))
    fade = out_cubic(prog(t, 0, 0.25))
    if fade < 1:
        img = Image.blend(Image.new("RGB", (W, H)), img, fade)
    return img


def render(cfg, out, poster, plate, layers, panel, py):
    ctx = cfg["setup"](poster)
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", out], stdin=subprocess.PIPE)

    blur = cfg.get("motion_blur", 0)   # flou de mouvement jusqu'à cet instant (s)
    for i in range(int(cfg.get("dur", DUR) * FPS)):
        t = i / FPS
        if t < blur:
            # obturateur à 180° : moyenne de 4 instants sur la demi-image
            acc = sum(np.asarray(compose(cfg, t + k / (8 * FPS), plate, layers, panel, py, ctx),
                                 np.float32) for k in range(4))
            img = Image.fromarray((acc / 4 + 0.5).astype(np.uint8))
        else:
            img = compose(cfg, t, plate, layers, panel, py, ctx)
        ff.stdin.write(img.tobytes())

    ff.stdin.close()
    if ff.wait():
        sys.exit("ffmpeg a échoué")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "collage"
    cfg = POSTERS[name]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, cfg["out"])
    cfg["src"] = os.path.join(ROOT, cfg["src"])
    K = W / cfg["src_w"]
    render(cfg, out, *prepare(cfg))
    print(out)
