#!/usr/bin/env python3
"""Anime les affiches des travaux pratiques de la journée d'inauguration.

    python3 build_post_animation.py [collage|lea|lea-eclat|segala|segala-pop|segala-pro|segala-eclat|segala-titres|segala-naturel|leo] [sortie.mp4]

  - collage : « Travaux pratiques — Collage » (atelier G-CEM ONE™ / GC)
  - lea     : « Léa FERRAND — TP Collage », version avec photo
  - lea-eclat : même affiche, version « choc » (flash, impacts, explosion)
  - segala  : « Charlotte SEGALA — Conférence NSK VCP2 », style épuré
  - segala-pop : même affiche, version plus vivante (vague, rebonds, lumières)
  - segala-pro : même affiche, version sobre et nette (ligne d'accent, ouvertures)
  - segala-eclat : même affiche, version percutante sans flou (impacts, explosion)
  - segala-titres : même affiche, intro typographique puis ouverture en cercle
  - segala-naturel : même affiche, profondeur (parallaxe) et mouvements souples
  - leo     : « Léo METZINGER — Conférence Oxygénothérapie », flux d'oxygène et pouls

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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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
    # sous le bandeau : couleur moyenne du bas du fond (sans liseré ni traînée verticale)
    plate[py:] = plate[py - 6].mean(0)
    layers = {}
    for name, b in cfg["boxes"].items():
        x0, y0, x1, y1 = box(b)
        layers[name] = (x0, y0, poster[y0:y1, x0:x1] - plate[y0:y1, x0:x1])
    panel = poster[py:].copy()
    return poster, plate, layers, panel, py


# --- composition ---------------------------------------------------------------
def add_layer(frame, layer, dx=0.0, dy=0.0, scale=1.0, alpha=1.0, clip=False, wipe=None,
              open_=None):
    """Ajoute un calque (différence) transformé autour de son centre.

    clip : limite le calque à sa zone d'origine (révélation par masque).
    wipe : fraction 0..1 révélée de gauche à droite.
    open_ : fraction 0..1 révélée depuis le centre vers les côtés.
    """
    if alpha <= 0:
        return
    x0, y0, d = layer
    h, w = d.shape[:2]
    if dx == dy == 0 and scale == 1 and wipe is None and open_ is None:
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
    if open_ is not None:
        xs = np.arange(size[0]) + ox
        half = open_ * (w / 2 + 20)
        out *= np.clip((half - np.abs(xs - w / 2)) / 20 + 0.5, 0, 1)[None, :, None]
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


def collage_setup(poster, plate=None):
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


def lea_setup(poster, plate=None):
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


# --- affiche « Léa FERRAND », version « éclat » --------------------------------
# Flash blanc, dézoom éclair, éléments qui s'écrasent à leur place (tremblement,
# onde de choc, aberration chromatique), explosion de particules et rayons
# lumineux derrière le produit.
PINK = (232, 40, 92)
# impacts : (instant, centre x, centre y, intensité)
ECLAT_IMPACTS = [(0.40, 248, 250, 0.6), (0.70, 732, 392, 1.0), (0.84, 201, 401, 0.5),
                 (1.15, 717, 787, 1.3), (1.62, 540, 1140, 0.35)]
PROD_C = (717, 787)


def _shock(t, decay):
    return sum(k * math.exp(-(t - t0) / decay) for t0, _, _, k in ECLAT_IMPACTS if t >= t0)


def _particles(seed, n, origin, t0, colors, speed=(300, 950)):
    rng = np.random.default_rng(seed)
    return [dict(t0=t0, x=origin[0], y=origin[1], a=rng.uniform(0, 2 * math.pi),
                 v=rng.uniform(*speed), r=rng.uniform(2.5, 8), life=rng.uniform(0.7, 1.3),
                 c=colors[rng.integers(len(colors))]) for _ in range(n)]


def bg_mask(poster, plate):
    """1 sur le fond seul, 0 sur les éléments : les effets de lumière passent derrière."""
    fg = np.clip((np.abs(poster - plate).sum(-1) - 20) / 40, 0, 1)
    fg = np.array(Image.fromarray((fg * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
                  .filter(ImageFilter.GaussianBlur(4))) / 255.0
    return (1 - fg).astype(np.float32)


def eclat_setup(poster, plate):
    ctx = lea_setup(poster)
    ctx["bg"] = bg_mask(poster, plate)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - PROD_C[0], yy - PROD_C[1]
    ctx["theta"] = np.arctan2(dy, dx)
    ctx["radial"] = np.exp(-(np.hypot(dx, dy) / 520) ** 2)
    ctx["sparks"] = (_particles(1, 80, PROD_C, 1.15, [(94, 234, 190), (255, 255, 255), PINK]) +
                     _particles(2, 34, (732, 392), 0.70, [(94, 234, 190), (255, 255, 255)],
                                (250, 700)))
    return ctx


def eclat_frame(t, f, L, ctx):
    def slam(name, start, dur, s0, dy=0.0):
        p = prog(t, start, dur)
        add_layer(f, L[name], scale=1 + (s0 - 1) * (1 - p) ** 2, dy=dy * (1 - p) ** 2,
                  alpha=clamp(p * 3))

    enter(f, L, t, "logo", 0.05, 0.4, dx=lambda p: -90 * e(p))
    enter(f, L, t, "pill", 0.10, 0.4, dx=lambda p: 110 * e(p))
    slam("name", 0.15, 0.25, 2.4)
    p = prog(t, 0.30, 0.38)
    add_layer(f, L["photo"], dx=-380 * e(p), alpha=clamp(p * 3))
    slam("card", 0.45, 0.25, 2.0)
    slam("gc", 0.62, 0.22, 1.8)
    enter(f, L, t, "time", 0.78, 0.35, dy=lambda p: 50 * e(p))
    p = prog(t, 0.85, 0.30)
    add_layer(f, L["prod"], scale=1 + 0.7 * (1 - p) ** 2, dy=-80 * (1 - p) ** 2 + float_bob(t, 1.6),
              alpha=clamp(p * 3))


def eclat_background(t, f, ctx):
    """Rayons lumineux tournants derrière le produit, après l'impact."""
    if t < 1.15:
        return
    k = 0.55 * math.exp(-(t - 1.15) / 0.35) + 0.22 * out_cubic(prog(t, 1.15, 0.4))
    rays = (0.5 + 0.5 * np.cos(14 * ctx["theta"] + 0.6 * t)) ** 6
    a = (rays * ctx["radial"] * k * ctx["bg"])[..., None]
    f += a * (np.array([200, 255, 235], np.float32) - f)


def eclat_overlay(t, img, ctx):
    d = ImageDraw.Draw(img, "RGBA")
    # ondes de choc
    for t0, cx, cy, k in ECLAT_IMPACTS[:4]:
        q = prog(t, t0, 0.45)
        if 0 < q < 1:
            r = 30 + 330 * k * out_cubic(q)
            d.ellipse((cx - r, cy - r, cx + r, cy + r),
                      outline=(255, 255, 255, int(200 * (1 - q))), width=max(1, int(8 * (1 - q))))
    # éclair au moment où le produit touche
    q = prog(t, 1.15, 0.3)
    if 0 < q < 1:
        ov = Image.new("RGBA", img.size, (255, 255, 255, int(110 * (1 - q) ** 2)))
        img.alpha_composite(ov)
    # particules
    for s in ctx["sparks"]:
        dt = t - s["t0"]
        if not 0 < dt < s["life"]:
            continue
        travel = s["v"] * (1 - math.exp(-3 * dt)) / 3
        x = s["x"] + math.cos(s["a"]) * travel
        y = s["y"] + math.sin(s["a"]) * travel + 260 * dt * dt
        fade = 1 - dt / s["life"]
        r = s["r"] * (0.4 + 0.6 * fade)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(*s["c"], int(255 * fade)))
    # pulsation de la carte puis reflets (comme la version précédente)
    q = prog(t, 2.2, 0.8)
    if 0 < q < 1:
        ring(img, (460, 325, 1004, 460), 22 * out_cubic(q), 0.8 * (1 - q), 22, 3, (255, 255, 255))


def eclat_shine(t, f, ctx):
    shine(f, LEA_BOXES["name"], 1.9, 0.6, t, ctx["name"], MINT, 0.9)
    shine(f, LEA_BOXES["time"], 2.4, 0.5, t, ctx["time"], MINT, 0.8)


def eclat_camera(t):
    z = 1 + 0.30 * (1 - out_expo(prog(t, 0, 1.0)))       # dézoom éclair
    a = 16 * _shock(t, 0.10)
    z = max(z, 1 + 0.035 * min(1, a / 6)) if a > 0.05 else z
    return z, a * math.sin(t * 97), a * math.cos(t * 71)


def eclat_post(t, img, ctx):
    k = int(round(9 * _shock(t, 0.07)))                   # aberration chromatique
    if k < 1:
        return img
    a = np.asarray(img).copy()
    a[..., 0] = np.roll(a[..., 0], k, axis=1)
    a[..., 2] = np.roll(a[..., 2], -k, axis=1)
    return Image.fromarray(a)


# --- affiche « Charlotte SEGALA — Conférence NSK VCP2 » ------------------------
# Style épuré : révélations par masque et rideau, halo doux derrière l'appareil.
SEGALA_BOXES = {                   # pixels de l'affiche 1080x1350
    "logo":   (20, 30, 418, 182),
    "pill":   (484, 58, 1036, 146),
    "name":   (30, 205, 632, 280),
    "role":   (30, 286, 284, 340),
    "nsk":    (30, 354, 305, 487),
    "card":   (470, 328, 1024, 524),
    "time":   (568, 530, 918, 606),
    "device": (100, 615, 505, 1003),   # jusqu'au bandeau : inclut l'ombre portée
}
ICE = np.array([170, 205, 255], np.float32)
NAVY = (12, 28, 78)
DEVICE_C = (300, 790)


def segala_setup(poster, plate):
    ctx = dict(name=glyph_mask(poster, SEGALA_BOXES["name"], lambda r: (r.min(-1) - 170) / 60),
               card=glyph_mask(poster, SEGALA_BOXES["card"], lambda r: (140 - r.mean(-1)) / 60),
               bg=bg_mask(poster, plate))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ctx["halo"] = np.exp(-((xx - DEVICE_C[0]) ** 2 + ((yy - DEVICE_C[1]) * 1.3) ** 2) / 300 ** 2)
    return ctx


def segala_frame(t, f, L, ctx):
    enter(f, L, t, "logo", 0.10, 0.55, dx=lambda p: -50 * e(p))
    enter(f, L, t, "pill", 0.18, 0.55, dx=lambda p: 60 * e(p))
    p = prog(t, 0.30, 0.60)
    add_layer(f, L["name"], dy=85 * e(p), clip=True, alpha=1.0 if p > 0 else 0)
    p = prog(t, 0.52, 0.50)
    add_layer(f, L["role"], wipe=out_expo(p), alpha=1.0 if p > 0 else 0)
    block_reveal(f, L, t, "nsk", 0.55, 0.60, WHITE)
    p = prog(t, 0.62, 0.60)
    add_layer(f, L["card"], wipe=out_expo(p), dy=14 * e(p), alpha=1.0 if p > 0 else 0)
    enter(f, L, t, "time", 0.88, 0.50, dy=lambda p: 24 * e(p))
    p = prog(t, 0.90, 0.75)
    add_layer(f, L["device"], dy=70 * e(p) + float_bob(t, 1.7),
              scale=0.95 + 0.05 * out_expo(p), alpha=clamp(p * 2))


def segala_background(t, f, ctx):
    """Halo de lumière douce qui s'allume derrière l'appareil."""
    k = 0.16 * out_cubic(prog(t, 1.1, 0.9)) * (1 + 0.15 * math.sin(2 * math.pi * (t - 2) / 2.6))
    if k > 0:
        a = (ctx["halo"] * ctx["bg"] * k)[..., None]
        f += a * (np.array([190, 215, 255], np.float32) - f)


def segala_shine(t, f, ctx):
    shine(f, SEGALA_BOXES["name"], 1.85, 0.7, t, ctx["name"], ICE, 0.85)
    shine(f, SEGALA_BOXES["card"], 2.45, 0.8, t, ctx["card"], np.array([60, 110, 220], np.float32), 0.7)
    shine(f, SEGALA_BOXES["name"], 4.0, 0.7, t, ctx["name"], ICE, 0.85)


def segala_overlay(t, img, ctx):
    q = prog(t, 3.1, 0.9)
    if 0 < q < 1:
        ring(img, (476, 333, 1019, 518), 20 * out_cubic(q), 0.7 * (1 - q), 22, 2, (255, 255, 255))


# --- même affiche, version « pop » : plus vivante et attirante -----------------
# Ouverture par balayage lumineux en diagonale, nom qui apparaît en vague,
# entrées avec petit rebond, bulles de lumière, étincelles sur l'appareil.
SPARKLES = [(1.75, 292, 640), (1.95, 462, 650), (2.15, 185, 700), (2.30, 420, 810),
            (2.50, 330, 760), (3.60, 320, 655), (3.80, 150, 860), (4.00, 470, 700),
            (4.20, 250, 720)]


def slices(layer, width):
    x0, y0, d = layer
    return [(x0 + i, y0, d[:, i:i + width]) for i in range(0, d.shape[1], width)]


def pop_setup(poster, plate):
    ctx = segala_setup(poster, plate)
    rng = np.random.default_rng(7)
    ctx["bokeh"] = [dict(x=rng.uniform(0, W), y=rng.uniform(150, 1000), r=rng.uniform(18, 55),
                         v=rng.uniform(15, 45), a=rng.uniform(0.10, 0.22), ph=rng.uniform(0, 6.3))
                    for _ in range(16)]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ctx["diag"] = xx + 0.6 * yy
    return ctx


def pop_frame(t, f, L, ctx):
    enter(f, L, t, "logo", 0.15, 0.5, dx=lambda p: -60 * e(p))
    enter(f, L, t, "pill", 0.20, 0.5, dx=lambda p: 70 * e(p))
    # nom puis fonction : vague de gauche à droite, tranche par tranche
    for name, start, step, rise in (("name", 0.30, 0.020, 80), ("role", 0.62, 0.025, 50)):
        for i, sl in enumerate(slices(L[name], 30)):
            p = prog(t, start + i * step, 0.40)
            if p > 0:
                add_layer(f, sl, dy=rise * e(p), clip=True)
    p = prog(t, 0.60, 0.50)
    add_layer(f, L["nsk"], scale=0.5 + 0.5 * out_back(p, 1.7), alpha=clamp(p * 3))
    p = prog(t, 0.72, 0.55)
    add_layer(f, L["card"], scale=0.7 + 0.3 * out_back(p, 1.4), dy=30 * e(p), alpha=clamp(p * 3))
    enter(f, L, t, "time", 0.98, 0.45, dx=lambda p: 70 * e(p))
    p = prog(t, 1.00, 0.70)
    add_layer(f, L["device"], dy=120 * (1 - out_back(p, 1.3)) + float_bob(t, 1.8),
              scale=0.9 + 0.1 * out_expo(p), alpha=clamp(p * 2.5))


def pop_background(t, f, ctx):
    segala_background(t, f, ctx)
    # bulles de lumière qui montent doucement (derrière les éléments)
    k_in = out_cubic(prog(t, 0.3, 1.0))
    for b in ctx["bokeh"]:
        y = (b["y"] - b["v"] * t) % 1000 + 50
        x = b["x"] + 12 * math.sin(t * 1.3 + b["ph"])
        r = b["r"]
        x0, x1 = int(max(0, x - 2 * r)), int(min(W, x + 2 * r))
        y0, y1 = int(max(0, y - 2 * r)), int(min(H, y + 2 * r))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / r ** 2) * b["a"] * k_in
        g = (g * ctx["bg"][y0:y1, x0:x1])[..., None]
        f[y0:y1, x0:x1] += g * (np.array([200, 225, 255], np.float32) - f[y0:y1, x0:x1])
    # traînée de lumière diagonale qui traverse l'affiche
    q = prog(t, 1.85, 0.9)
    if 0 < q < 1:
        c = -300 + in_out(q) * (W + 0.6 * H + 600)
        band = np.exp(-((ctx["diag"] - c) / 70) ** 2) * 0.22 * ctx["bg"]
        f += band[..., None] * (WHITE - f)


def pop_shine(t, f, ctx):
    shine(f, SEGALA_BOXES["name"], 2.6, 0.7, t, ctx["name"], ICE, 0.85)
    shine(f, SEGALA_BOXES["card"], 3.1, 0.8, t, ctx["card"], np.array([60, 110, 220], np.float32), 0.7)


def sparkle(d, x, y, size, alpha):
    w = size * 0.18
    pts = [(x, y - size), (x + w, y - w), (x + size, y), (x + w, y + w),
           (x, y + size), (x - w, y + w), (x - size, y), (x - w, y - w)]
    d.polygon(pts, fill=(255, 255, 255, alpha))
    g = size * 0.35
    d.ellipse((x - g, y - g, x + g, y + g), fill=(200, 225, 255, alpha // 3))


def pop_overlay(t, img, ctx):
    d = ImageDraw.Draw(img, "RGBA")
    for t0, x, y in SPARKLES:
        q = prog(t, t0, 0.55)
        if 0 < q < 1:
            sparkle(d, x, y, 42 * math.sin(math.pi * q), int(255 * math.sin(math.pi * q)))
    q = prog(t, 3.3, 0.9)
    if 0 < q < 1:
        ring(img, (476, 333, 1019, 518), 20 * out_cubic(q), 0.7 * (1 - q), 22, 2, (255, 255, 255))


def pop_post(t, img, ctx):
    """Ouverture : balayage diagonal avec une lisière lumineuse."""
    q = prog(t, 0.0, 0.6)
    if q >= 1:
        return img
    c = -150 + out_cubic(q) * (W + 0.6 * H + 300)
    diag = ctx["diag"]
    m = np.clip((c - diag) / 40, 0, 1)[..., None]
    a = np.asarray(img, np.float32)
    out = np.array(NAVY, np.float32) * (1 - m) + a * m
    edge = np.exp(-((diag - c) / 22) ** 2)[..., None]
    out += edge * (np.array([215, 235, 255], np.float32) - out) * 0.9
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# --- même affiche, version « pro » : sobre, nette, sans effets décoratifs ------
ACCENT = np.array([120, 175, 255], np.float32)


def pro_frame(t, f, L, ctx):
    enter(f, L, t, "logo", 0.10, 0.60, dx=lambda p: -40 * e(p))
    p = prog(t, 0.15, 0.55)
    add_layer(f, L["pill"], open_=out_expo(p), alpha=1.0 if p > 0 else 0)
    # ligne d'accent qui se trace sous le nom, le nom sort de derrière, puis elle repart
    x0, x1, y = 30, 632, 281
    p, q = out_expo(prog(t, 0.28, 0.45)), out_expo(prog(t, 0.95, 0.45))
    if 0 < p and q < 1:
        a, b = int(x0 + (x1 - x0) * q), int(x0 + (x1 - x0) * p)
        if b > a:
            f[y:y + 3, a:b] = ACCENT
    p = prog(t, 0.40, 0.60)
    add_layer(f, L["name"], dy=75 * e(p), clip=True, alpha=1.0 if p > 0 else 0)
    enter(f, L, t, "role", 0.60, 0.50, dy=lambda p: 16 * e(p))
    p = prog(t, 0.62, 0.55)
    add_layer(f, L["nsk"], open_=out_expo(p), alpha=1.0 if p > 0 else 0)
    p = prog(t, 0.72, 0.60)
    add_layer(f, L["card"], open_=out_expo(p), dy=12 * e(p), alpha=1.0 if p > 0 else 0)
    enter(f, L, t, "time", 0.95, 0.50, dx=lambda p: 40 * e(p))
    p = prog(t, 1.00, 0.80)
    add_layer(f, L["device"], dy=60 * e(p), scale=0.96 + 0.04 * out_expo(p), alpha=clamp(p * 2))


def pro_shine(t, f, ctx):
    shine(f, SEGALA_BOXES["name"], 2.1, 0.8, t, ctx["name"], ICE, 0.8)


def pro_camera(t):
    # apparition avec léger recul, puis lente avancée qui se pose
    z = 1 + 0.04 * (1 - out_expo(prog(t, 0, 1.4))) + 0.02 * in_out(prog(t, 1.4, 2.6))
    return z, 0.0, 0.0


# --- même affiche, version « éclat » : percutante mais nette (sans flou) -------
SG_IMPACTS = [(0.40, 331, 242, 0.6), (0.70, 747, 426, 1.0), (0.84, 167, 420, 0.5),
              (1.15, 300, 790, 1.3), (1.62, 540, 1130, 0.35)]
SG_COLORS = [(255, 255, 255), (170, 205, 255), (120, 175, 255), (175, 125, 255)]


def sg_shock(t, decay):
    return sum(k * math.exp(-(t - t0) / decay) for t0, _, _, k in SG_IMPACTS if t >= t0)


def sg_eclat_setup(poster, plate):
    ctx = segala_setup(poster, plate)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - DEVICE_C[0], yy - DEVICE_C[1]
    ctx["theta"] = np.arctan2(dy, dx)
    ctx["radial"] = np.exp(-(np.hypot(dx, dy) / 520) ** 2)
    ctx["sparks"] = (_particles(3, 85, DEVICE_C, 1.15, SG_COLORS) +
                     _particles(4, 36, (747, 426), 0.70, SG_COLORS[:3], (250, 700)))
    return ctx


def sg_eclat_frame(t, f, L, ctx):
    def slam(name, start, dur, s0, dy=0.0, bob=0.0):
        p = prog(t, start, dur)
        add_layer(f, L[name], scale=1 + (s0 - 1) * (1 - p) ** 2, dy=dy * (1 - p) ** 2 + bob,
                  alpha=clamp(p * 3))

    enter(f, L, t, "logo", 0.05, 0.40, dx=lambda p: -90 * e(p))
    enter(f, L, t, "pill", 0.10, 0.40, dx=lambda p: 110 * e(p))
    slam("name", 0.15, 0.25, 2.3)
    enter(f, L, t, "role", 0.38, 0.35, dx=lambda p: -120 * e(p))
    slam("card", 0.45, 0.25, 2.0)
    slam("nsk", 0.62, 0.22, 1.9)
    enter(f, L, t, "time", 0.78, 0.35, dy=lambda p: 50 * e(p))
    slam("device", 0.85, 0.30, 1.7, dy=-80, bob=float_bob(t, 1.6))


def sg_eclat_background(t, f, ctx):
    segala_background(t, f, ctx)
    if t < 1.15:
        return
    k = 0.5 * math.exp(-(t - 1.15) / 0.35) + 0.2 * out_cubic(prog(t, 1.15, 0.4))
    rays = (0.5 + 0.5 * np.cos(14 * ctx["theta"] + 0.6 * t)) ** 6
    a = (rays * ctx["radial"] * k * ctx["bg"])[..., None]
    f += a * (np.array([200, 222, 255], np.float32) - f)


def sg_eclat_overlay(t, img, ctx):
    d = ImageDraw.Draw(img, "RGBA")
    for t0, cx, cy, k in SG_IMPACTS[:4]:              # ondes de choc
        q = prog(t, t0, 0.45)
        if 0 < q < 1:
            r = 30 + 330 * k * out_cubic(q)
            d.ellipse((cx - r, cy - r, cx + r, cy + r),
                      outline=(255, 255, 255, int(200 * (1 - q))), width=max(1, int(8 * (1 - q))))
    q = prog(t, 1.15, 0.3)                            # éclair à l'arrivée de l'appareil
    if 0 < q < 1:
        img.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, int(100 * (1 - q) ** 2))))
    for s in ctx["sparks"]:                           # particules
        dt = t - s["t0"]
        if not 0 < dt < s["life"]:
            continue
        travel = s["v"] * (1 - math.exp(-3 * dt)) / 3
        x = s["x"] + math.cos(s["a"]) * travel
        y = s["y"] + math.sin(s["a"]) * travel + 260 * dt * dt
        fade = 1 - dt / s["life"]
        r = s["r"] * (0.4 + 0.6 * fade)
        d.ellipse((x - r, y - r, x + r, y + r), fill=(*s["c"], int(255 * fade)))
    q = prog(t, 2.3, 0.8)
    if 0 < q < 1:
        ring(img, (476, 333, 1019, 518), 22 * out_cubic(q), 0.8 * (1 - q), 22, 3, (255, 255, 255))


def sg_eclat_shine(t, f, ctx):
    shine(f, SEGALA_BOXES["name"], 1.9, 0.6, t, ctx["name"], ICE, 0.9)
    shine(f, SEGALA_BOXES["card"], 2.5, 0.7, t, ctx["card"], np.array([60, 110, 220], np.float32), 0.7)


def sg_eclat_camera(t):
    z = 1 + 0.30 * (1 - out_expo(prog(t, 0, 1.0)))   # dézoom éclair
    a = 9 * sg_shock(t, 0.09)                           # tremblement court et léger
    z = max(z, 1 + 0.02 * min(1, a / 4)) if a > 0.05 else z
    return z, a * math.sin(t * 97), a * math.cos(t * 71)


# --- même affiche, version « titres » : intro typographique puis ouverture -----
# Trois écrans de titres plein cadre qui se poussent l'un l'autre, puis un
# cercle s'ouvre au centre et dévoile l'affiche, qui finit de se mettre en place.
FONT_DIR = "/usr/share/fonts/opentype/inter"
TITLES = [  # (début, fond, couleur titre, titre, sous-titre, couleur sous-titre)
    (0.00, NAVY, (255, 255, 255), "CONFÉRENCE", "Charlotte SEGALA", (150, 190, 255)),
    (0.42, (255, 255, 255), (16, 36, 104), "DEEP CLEANING", "avec le VCP2 · NSK", (50, 100, 215)),
    (0.84, (40, 95, 215), (255, 255, 255), "15H – 15H30", "29 OCTOBRE", (205, 222, 255)),
]
IRIS_AT = 1.25


def text_layer(text, font_file, max_w, max_size, color, track=0.04):
    """Texte espacé sur calque RGBA transparent, à la taille qui tient dans max_w."""
    size = max_size
    while True:
        font = ImageFont.truetype(os.path.join(FONT_DIR, font_file), size)
        widths = [font.getlength(c) for c in text]
        total = sum(widths) + track * size * (len(text) - 1)
        if total <= max_w or size <= 20:
            break
        size -= 4
    asc, desc = font.getmetrics()
    img = Image.new("RGBA", (int(total) + 4, asc + desc + 4), (0, 0, 0, 0))
    d, x = ImageDraw.Draw(img), 2
    for c, w in zip(text, widths):
        d.text((x, 2), c, font=font, fill=color)
        x += w + track * size
    return img


def titles_setup(poster, plate):
    ctx = segala_setup(poster, plate)
    ctx["titles"] = [(t0, bg, text_layer(word, "InterDisplay-Black.otf", 900, 190, fg, 0.03),
                      text_layer(cap, "InterDisplay-SemiBold.otf", 800, 64, cc, 0.06), fg)
                     for t0, bg, fg, word, cap, cc in TITLES]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ctx["dist"] = np.hypot(xx - W / 2, yy - H / 2)
    return ctx


def title_scene(t, scene):
    t0, bg, word, cap, fg = scene
    img = Image.new("RGB", (W, H), bg)
    k = 1.10 - 0.10 * out_expo(prog(t, t0, 0.5))          # le titre se pose en se resserrant
    w = word.resize((int(word.width * k), int(word.height * k)), Image.BICUBIC)
    img.paste(w, ((W - w.width) // 2, H // 2 - w.height // 2 - 40), w)
    p = prog(t, t0 + 0.08, 0.35)                            # sous-titre qui monte
    if p > 0:
        c = cap.copy()
        c.putalpha(c.getchannel("A").point(lambda v: int(v * clamp(p * 2))))
        img.paste(c, ((W - c.width) // 2, H // 2 + 60 + int(30 * e(p))), c)
    p = out_expo(prog(t, t0 + 0.05, 0.35))                  # filet d'accent qui se trace
    if p > 0:
        bw = int(140 * p)
        ImageDraw.Draw(img).rectangle(((W - bw) // 2, H // 2 - 170, (W + bw) // 2, H // 2 - 164),
                                      fill=fg)
    return img


def titles_intro(t, ctx):
    scenes = ctx["titles"]
    cur = max(i for i, sc in enumerate(scenes) if t >= sc[0])
    img = title_scene(t, scenes[cur])
    if cur > 0:                                             # le nouvel écran pousse l'ancien
        p = out_expo(prog(t, scenes[cur][0], 0.30))
        if p < 1:
            prev = title_scene(t, scenes[cur - 1])
            canvas = Image.new("RGB", (W, H))
            canvas.paste(prev, (int(-0.35 * W * p), 0))
            canvas.paste(img, (int(W * (1 - p)), 0))
            img = canvas
    return img


def titles_frame(t, f, L, ctx):
    for name in ("logo", "pill", "role", "nsk", "card", "time"):
        add_layer(f, L[name])
    p = prog(t, IRIS_AT + 0.10, 0.55)
    add_layer(f, L["name"], dy=75 * e(p), clip=True, alpha=1.0 if p > 0 else 0)
    p = prog(t, IRIS_AT + 0.20, 0.70)
    add_layer(f, L["device"], dy=80 * e(p) + float_bob(t, IRIS_AT + 0.9),
              scale=0.94 + 0.06 * out_expo(p), alpha=clamp(p * 2))


def titles_shine(t, f, ctx):
    shine(f, SEGALA_BOXES["name"], 2.4, 0.7, t, ctx["name"], ICE, 0.85)
    shine(f, SEGALA_BOXES["card"], 2.9, 0.8, t, ctx["card"], np.array([60, 110, 220], np.float32), 0.7)


def titles_overlay(t, img, ctx):
    q = prog(t, 3.3, 0.8)
    if 0 < q < 1:
        ring(img, (476, 333, 1019, 518), 20 * out_cubic(q), 0.75 * (1 - q), 22, 2, (255, 255, 255))


def titles_camera(t):
    return 1 + 0.12 * (1 - out_expo(prog(t, IRIS_AT, 1.0))), 0.0, 0.0


def titles_post(t, img, ctx):
    if t < IRIS_AT:
        return titles_intro(t, ctx)
    q = prog(t, IRIS_AT, 0.55)
    if q >= 1:
        return img
    r = out_expo(q) * (math.hypot(W / 2, H / 2) + 20)
    m = np.clip((r - ctx["dist"]) / 3, 0, 1)[..., None]
    intro = np.asarray(titles_intro(t, ctx), np.float32)
    out = intro * (1 - m) + np.asarray(img, np.float32) * m
    edge = np.exp(-((ctx["dist"] - r) / 4) ** 2)[..., None] * (1 - q)
    out += edge * (WHITE - out)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# --- même affiche, version « naturelle » : profondeur et mouvements souples ----
# La caméra glisse lentement et revient au centre ; chaque élément bouge selon
# sa profondeur (parallaxe), apparaît en douceur avec un ressort à peine
# perceptible, et une lumière douce se déplace dans le fond.
DEPTH = {"logo": 0.6, "pill": 0.6, "name": 1.0, "role": 1.0, "nsk": 1.2,
         "card": 1.4, "time": 1.3, "device": 2.0}
NAT_ENTRIES = [("logo", 0.20), ("pill", 0.30), ("name", 0.45), ("role", 0.60), ("nsk", 0.70),
               ("card", 0.80), ("time", 0.95), ("device", 1.00)]


def spring(p):
    return 1 - math.exp(-5 * p) * math.cos(9 * p)


def nat_cam(t):
    k = 1 - in_out(prog(t, 0, 3.6))
    return -14 * k, -10 * k          # décalage léger : les éléments partent vers la droite


def nat_setup(poster, plate):
    ctx = segala_setup(poster, plate)
    ctx["plate_pad"] = np.pad(plate, ((24, 24), (24, 24), (0, 0)), mode="edge")
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ctx["xx"], ctx["yy"] = xx, yy
    return ctx


def nat_plate(t, plate, ctx):
    cx, cy = nat_cam(t)
    ox, oy = 24 - 0.3 * cx, 24 - 0.3 * cy          # le fond bouge peu (lointain)
    if abs(ox - 24) < 1e-3 and abs(oy - 24) < 1e-3:
        return plate.copy()
    pad = ctx["plate_pad"]
    return np.stack([np.array(Image.fromarray(pad[..., c]).transform(
        (W, H), Image.AFFINE, (1, 0, ox, 0, 1, oy), Image.BILINEAR)) for c in range(3)], -1)


def nat_frame(t, f, L, ctx):
    cx, cy = nat_cam(t)
    for name, start in NAT_ENTRIES:
        p = prog(t, start, 0.9)
        if p <= 0:
            continue
        d = DEPTH[name]
        rise = (50 if name == "device" else 30) * (1 - spring(p))
        bob = float_bob(t, 2.0) if name == "device" else 0.0
        add_layer(f, L[name], dx=-d * cx, dy=-d * cy + rise + bob, alpha=clamp(p * 1.8))


def nat_background(t, f, ctx):
    segala_background(t, f, ctx)
    # lumière douce qui dérive lentement en haut à droite
    k = 0.10 * out_cubic(prog(t, 0.3, 1.2))
    lx, ly = 760 + 140 * math.sin(t * 0.7), 260 + 60 * math.cos(t * 0.5)
    g = np.exp(-(((ctx["xx"] - lx) / 380) ** 2 + ((ctx["yy"] - ly) / 300) ** 2)) * k * ctx["bg"]
    f += g[..., None] * (np.array([185, 210, 255], np.float32) - f)


def nat_shine(t, f, ctx):
    shine(f, SEGALA_BOXES["name"], 2.6, 0.9, t, ctx["name"], ICE, 0.7)


def nat_camera(t):
    return 1 + 0.02 * (1 - in_out(prog(t, 0, 3.6))), 0.0, 0.0


# --- affiche « Léo METZINGER — Conférence Oxygénothérapie » -------------------
# Thème de la conférence : l'oxygénothérapie. Des flux d'oxygène lumineux
# sortent de la valve de la bouteille et montent vers la carte, un tracé de
# pouls (moniteur) traverse l'affiche, les éléments arrivent comme portés par
# l'air, et l'affiche « respire » sur un cycle complet.
LEO_BOXES = {                      # pixels de l'affiche 1080x1350
    "logo":    (20, 30, 418, 182),
    "pill":    (484, 58, 1036, 146),
    "name":    (30, 205, 570, 278),
    "role":    (30, 290, 282, 342),
    "labodal": (32, 364, 405, 465),
    "card":    (472, 328, 1024, 466),
    "time":    (572, 474, 916, 546),
    "kit":     (85, 575, 425, 1003),
    "bottle":  (425, 660, 512, 1003),
    "photo":   (728, 572, 1080, 1003),
}
VALVE = (462, 668)
BREATH_AT, BREATH_LEN = 1.8, 3.2
LEO_ENTRIES = [  # (calque, début, montée en px, arrivée latérale en px)
    ("logo", 0.10, 0, -50), ("pill", 0.18, 0, 60), ("name", 0.30, 40, 0), ("role", 0.45, 30, 0),
    ("labodal", 0.55, 34, 0), ("card", 0.65, 40, 0), ("time", 0.80, 30, 0),
    ("kit", 0.90, 70, 0), ("photo", 0.95, 0, 80), ("bottle", 1.00, 90, 0),
]


def bezier(p0, p1, p2, p3, n=160):
    u = np.linspace(0, 1, n)[:, None]
    pts = ((1 - u) ** 3 * np.array(p0) + 3 * (1 - u) ** 2 * u * np.array(p1) +
           3 * (1 - u) * u ** 2 * np.array(p2) + u ** 3 * np.array(p3))
    return pts


def ecg_y(x, y0=560.0, period=270.0):
    """Tracé de pouls : onde P, complexe QRS, onde T, sur une ligne de base."""
    u = (x % period) / period
    g = lambda c, w, a: a * math.exp(-((u - c) / w) ** 2)
    return y0 - g(0.30, 0.025, 8) + g(0.42, 0.010, 14) - g(0.45, 0.012, 52) + g(0.48, 0.010, 22) \
        - g(0.66, 0.040, 14)


def leo_setup(poster, plate):
    # courants d'oxygène : de la valve vers la carte « Oxygénothérapie »
    flows = []
    for i, (spread, lift) in enumerate([(-40, 0), (0, 20), (40, 40), (80, 10), (120, 50), (160, 30)]):
        flows.append(dict(pts=bezier((VALVE[0] + 2 * i - 5, VALVE[1] - 6),
                                     (VALVE[0] - 10 + spread * 0.2, 560 - lift),
                                     (560 + spread, 600 - lift * 2),
                                     (600 + spread * 1.6, 470 - lift)),
                          phase=i * 0.27, period=1.5 + 0.12 * i, width=2.2 + (i % 3) * 0.8))
    xs = np.arange(0, W + 1, 2, dtype=np.float32)
    ecg = np.array([(x, ecg_y(x)) for x in xs], np.float32)
    return dict(name=glyph_mask(poster, LEO_BOXES["name"], lambda r: (r.min(-1) - 200) / 40),
                bg=bg_mask(poster, plate), flows=flows, ecg=ecg)


def leo_frame(t, f, L, ctx):
    for name, start, rise, side in LEO_ENTRIES:
        p = prog(t, start, 0.9)
        if p <= 0:
            continue
        sway = 6 * math.sin(7 * p) * (1 - p) if rise else 0.0       # porté par l'air
        bob = float_bob(t, 2.0) if name == "bottle" else 0.0
        add_layer(f, L[name], dx=side * (1 - out_expo(p)) + sway,
                  dy=rise * (1 - spring(p)) + bob, alpha=clamp(p * 1.8))


def leo_oxygen(t, f, ctx):
    """Flux d'oxygène et tracé de pouls, dessinés en 2x, passés derrière les éléments."""
    S = 2
    ov = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    # flux : des traînées qui glissent le long des courbes, queue qui s'efface
    k_in = out_cubic(prog(t, 1.25, 0.6))
    for fl in ctx["flows"]:
        if k_in <= 0:
            break
        pts, n = fl["pts"], len(fl["pts"])
        for rep_ in range(2):                                   # deux traînées par courant
            head = ((t - 1.25) / fl["period"] + fl["phase"] + rep_ * 0.5) % 1.0 * 1.4
            i1 = int(min(n - 1, head * n))
            i0 = int(max(0, (head - 0.35) * n))
            if i1 - i0 < 2:
                continue
            for j in range(i0, i1):
                q = (j - i0) / max(1, i1 - i0)                   # 0 = queue, 1 = tête
                fade_end = 1 - clamp((j / n - 0.85) / 0.15)      # s'efface en arrivant
                a = int(230 * q * fade_end * k_in)
                if a <= 2:
                    continue
                (x0, y0), (x1, y1) = pts[j] * S, pts[j + 1] * S
                d.line((x0, y0, x1, y1), fill=(255, 255, 255, a),
                       width=max(2, int(fl["width"] * S * (0.5 + 0.5 * q))))
    # tracé de pouls : la tête lumineuse balaie l'affiche, la trace s'efface derrière
    for start in (1.15, 3.05):
        p = prog(t, start, 1.7)
        if p <= 0 or p >= 1 and t > start + 2.4:
            continue
        xh = p * (W + 40)
        ecg = ctx["ecg"]
        fade_all = 1 - clamp((t - start - 1.7) / 0.6)
        seg = ecg[(ecg[:, 0] <= xh) & (ecg[:, 0] >= xh - 620)]
        for j in range(len(seg) - 1):
            q = 1 - (xh - seg[j, 0]) / 620
            a = int(235 * q * fade_all)
            if a > 2:
                d.line((seg[j, 0] * S, seg[j, 1] * S, seg[j + 1, 0] * S, seg[j + 1, 1] * S),
                       fill=(255, 255, 255, a), width=3 * S)
        if p < 1:
            yh = ecg_y(xh) * S
            for r, a in ((16, 60), (9, 140), (5, 255)):
                d.ellipse((xh * S - r * S / 2, yh - r * S / 2, xh * S + r * S / 2, yh + r * S / 2),
                          fill=(255, 255, 255, a))
    ov = np.asarray(ov.resize((W, H), Image.LANCZOS), np.float32)
    a = (ov[..., 3] / 255 * ctx["bg"])[..., None]
    f *= 1 - a
    f += a * ov[..., :3]


def leo_shine(t, f, ctx):
    shine(f, LEO_BOXES["name"], 2.4, 0.8, t, ctx["name"], np.array([200, 245, 255], np.float32), 0.8)


def leo_overlay(t, img, ctx):
    # onde « inspiration » autour de la carte, au rythme de la respiration
    q = prog(t, BREATH_AT + 0.1, 1.4)
    if 0 < q < 1:
        ring(img, (476, 333, 1019, 461), 34 * out_cubic(q), 0.75 * (1 - q), 22, 3, (255, 255, 255))


def leo_camera(t):
    breath = 0.5 - 0.5 * math.cos(2 * math.pi * prog(t, BREATH_AT, BREATH_LEN))
    return 1 + 0.03 * (1 - out_expo(prog(t, 0, 1.4))) + 0.014 * breath, 0.0, 0.0


POSTERS = {
    "collage": dict(src="assets/post-collage.webp", out="video/post-collage-anime.mp4",
                    src_w=1600, boxes=COLLAGE_BOXES, panel_y=1515, panel_in=1.25,
                    setup=collage_setup, frame=collage_frame, shine=collage_shine,
                    overlay=collage_overlay),
    "lea": dict(src="assets/post-lea-ferrand.webp", out="video/post-lea-ferrand-anime.mp4",
                src_w=1080, boxes=LEA_BOXES, panel_y=1022,
                setup=lea_setup, frame=lea_frame, shine=lea_shine, overlay=lea_overlay,
                panel_in=1.0, dur=4.5, motion_blur=1.8),
    "lea-eclat": dict(src="assets/post-lea-ferrand.webp", out="video/post-lea-ferrand-eclat.mp4",
                      src_w=1080, boxes=LEA_BOXES, panel_y=1022, panel_in=1.2, panel_ease="back",
                      setup=eclat_setup, frame=eclat_frame, shine=eclat_shine,
                      overlay=eclat_overlay, background_fx=eclat_background,
                      camera=eclat_camera, post=eclat_post,
                      intro_color=(255, 255, 255), intro_dur=0.22, dur=5.0, motion_blur=1.9),
    "segala": dict(src="assets/post-charlotte-segala.png", out="video/post-charlotte-segala-anime.mp4",
                   src_w=1080, boxes=SEGALA_BOXES, panel_y=1003, panel_in=1.2,
                   setup=segala_setup, frame=segala_frame, shine=segala_shine,
                   overlay=segala_overlay, background_fx=segala_background,
                   intro_color=NAVY, intro_dur=0.35, dur=5.0, motion_blur=1.9),
    "segala-pop": dict(src="assets/post-charlotte-segala.png", out="video/post-charlotte-segala-pop.mp4",
                       src_w=1080, boxes=SEGALA_BOXES, panel_y=1003, panel_in=1.3, panel_ease="back",
                       setup=pop_setup, frame=pop_frame, shine=pop_shine, overlay=pop_overlay,
                       background_fx=pop_background, post=pop_post,
                       intro_dur=0.001, dur=5.0),
    "segala-pro": dict(src="assets/post-charlotte-segala.png", out="video/post-charlotte-segala-pro.mp4",
                       src_w=1080, boxes=SEGALA_BOXES, panel_y=1003, panel_in=1.3,
                       setup=segala_setup, frame=pro_frame, shine=pro_shine,
                       overlay=lambda t, img, ctx: None, camera=pro_camera,
                       intro_color=NAVY, intro_dur=0.35, dur=4.5),
    "segala-eclat": dict(src="assets/post-charlotte-segala.png",
                         out="video/post-charlotte-segala-eclat.mp4",
                         src_w=1080, boxes=SEGALA_BOXES, panel_y=1003, panel_in=1.2,
                         panel_ease="back", setup=sg_eclat_setup, frame=sg_eclat_frame,
                         shine=sg_eclat_shine, overlay=sg_eclat_overlay,
                         background_fx=sg_eclat_background, camera=sg_eclat_camera,
                         intro_color=(255, 255, 255), intro_dur=0.22, dur=5.0),
    "segala-titres": dict(src="assets/post-charlotte-segala.png",
                          out="video/post-charlotte-segala-titres.mp4",
                          src_w=1080, boxes=SEGALA_BOXES, panel_y=1003, panel_in=IRIS_AT + 0.45,
                          setup=titles_setup, frame=titles_frame, shine=titles_shine,
                          overlay=titles_overlay, camera=titles_camera, post=titles_post,
                          intro_dur=0.001, dur=4.8),
    "segala-naturel": dict(src="assets/post-charlotte-segala.png",
                           out="video/post-charlotte-segala-naturel.mp4",
                           src_w=1080, boxes=SEGALA_BOXES, panel_y=1003, panel_in=1.3,
                           panel_dur=0.9, setup=nat_setup, plate_fx=nat_plate, frame=nat_frame,
                           shine=nat_shine, overlay=lambda t, img, ctx: None,
                           background_fx=nat_background, camera=nat_camera,
                           intro_color=NAVY, intro_dur=0.5, dur=5.0),
    "leo": dict(src="assets/post-leo-metzinger.webp", out="video/post-leo-metzinger-anime.mp4",
                src_w=1080, boxes=LEO_BOXES, panel_y=1003, panel_in=1.4, panel_dur=0.8,
                setup=leo_setup, frame=leo_frame, background_fx=leo_oxygen,
                shine=leo_shine, overlay=leo_overlay, camera=leo_camera,
                intro_color=(255, 255, 255), intro_dur=0.4, dur=5.0),
}


def compose(cfg, t, plate, layers, panel, py, ctx):
    f = cfg["plate_fx"](t, plate, ctx) if "plate_fx" in cfg else plate.copy()
    cfg["frame"](t, f, layers, ctx)
    if "background_fx" in cfg:
        cfg["background_fx"](t, f, ctx)

    # bandeau d'informations qui monte
    p = prog(t, cfg["panel_in"], cfg.get("panel_dur", 0.55))
    p = out_back(p, 1.3) if cfg.get("panel_ease") == "back" else out_expo(p)
    top = max(py - 40, py + int(round((H - py) * (1 - p))))
    if top < py:                       # dépassement (rebond) : on étire la 1re ligne
        f[top:py] = panel[0]
        f[py:] = panel
    elif top < H:
        f[top:] = panel[:H - top]

    cfg["shine"](t, f, ctx)
    img = Image.fromarray(np.clip(f, 0, 255).astype(np.uint8)).convert("RGBA")
    cfg["overlay"](t, img, ctx)

    # caméra : recul rapide (+ tremblement éventuel) + fondu d'ouverture
    img = img.convert("RGB")
    if "camera" in cfg:
        z, ox, oy = cfg["camera"](t)
    else:
        z, ox, oy = 1 + 0.07 * (1 - out_expo(prog(t, 0, 1.6))), 0.0, 0.0
    if z > 1.0001:
        cw, ch = W / z, H / z
        bx = clamp((W - cw) / 2 + ox, 0, W - cw)
        by = clamp((H - ch) * 0.35 + oy, 0, H - ch)
        img = img.resize((W, H), Image.BICUBIC, box=(bx, by, bx + cw, by + ch))
    if "post" in cfg:
        img = cfg["post"](t, img, ctx)
    fade = out_cubic(prog(t, 0, cfg.get("intro_dur", 0.25)))
    if fade < 1:
        start = Image.new("RGB", (W, H), cfg.get("intro_color", (0, 0, 0)))
        img = Image.blend(start, img, fade)
    return img


def render(cfg, out, poster, plate, layers, panel, py):
    ctx = cfg["setup"](poster, plate)
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
