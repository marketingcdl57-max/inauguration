#!/usr/bin/env python3
"""Anime l'affiche « Owandy-IOS Wireless » (Comptoir Dentaire Lorrain).

Format 1080x1350 (post Instagram / Facebook / LinkedIn 4:5), 30 i/s, 10 s.
Tout est tiré de l'affiche d'origine (assets/owandy/affiche-owandy-ios.webp) :
le fond est reconstitué, chaque bloc de texte et le scanner sont détourés,
puis réassemblés avec des animations :
  - halo lumineux pulsant derrière le scanner + particules de lumière ;
  - le scanner monte en douceur, puis flotte, avec des reflets qui balayent
    sa coque ;
  - les textes entrent un à un (glissés, révélations masquées), les filets
    bleus se tracent, « liberté » scintille ;
  - le bouton CTA rebondit, pulse et reçoit un éclat de lumière ;
  - la vague se dessine de gauche à droite, le logo CDL apparaît.

Dépendances : numpy, opencv-python-headless, ffmpeg.
Usage : python3 build_owandy.py [sortie.mp4]
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "assets/owandy/affiche-owandy-ios.webp")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "video/owandy-ios-wireless.mp4")
FPS, DUR = 30, 10.0

# ---------------------------------------------------------------- découpage
img = cv2.imread(SRC)  # BGR
H, W = img.shape[:2]
f32 = img.astype(np.float32)
lum = lambda a: a.astype(np.float32).mean(2)

# 1) Fond propre : on masque tout ce qui se détache du dégradé, puis inpainting
#    basse résolution (le fond est très doux, ça suffit et c'est sans artefact).
light = cv2.GaussianBlur(cv2.dilate(f32, np.ones((25, 25))), (0, 0), 15)
mask = ((np.abs(f32 - light).max(2) > 18) * 255).astype(np.uint8)
mask = cv2.dilate(mask, np.ones((9, 9)))
mask[40:1200, 600:1000] = 255  # scanner + ombre
small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
ms = cv2.resize(mask, (W // 4, H // 4), interpolation=cv2.INTER_NEAREST)
inp = cv2.inpaint(small, ms, 12, cv2.INPAINT_TELEA)
inp = cv2.GaussianBlur(cv2.resize(inp, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 12)
clean = np.where(mask[..., None] > 0, inp, img)
cleanf = clean.astype(np.float32)

# 2) Scanner : coque gris neutre ou bleu marine, alors que le fond est bleuté.
fi = img.astype(np.int16)
body = ((((fi[..., 0] - fi[..., 2]) < 14) | (fi.max(2) < 200)) * 255).astype(np.uint8)
roi = np.zeros((H, W), np.uint8)
roi[50:1150, 630:950] = 1
body *= roi
body = cv2.morphologyEx(body, cv2.MORPH_OPEN, np.ones((5, 5)))
n, lab, st, _ = cv2.connectedComponentsWithStats(body)
body = np.where(lab == 1 + np.argmax(st[1:, 4]), 255, 0).astype(np.uint8)
cs, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
body = np.zeros_like(body)
cv2.drawContours(body, cs, -1, 255, -1)
body = cv2.GaussianBlur(cv2.erode(body, np.ones((3, 3))), (0, 0), 1.0)
near = cv2.dilate((body > 0).astype(np.uint8), np.ones((61, 61))) * roi
dark = np.array([60, 50, 50], np.float32)
shadow = np.clip((lum(clean) - lum(img)) / (lum(clean) - dark.mean()), 0, 1) * near
shadow = cv2.GaussianBlur(shadow.astype(np.float32), (0, 0), 1.5)
outside = body < 128
sc_rgb = f32.copy()
sc_rgb[outside] = dark
sc_a = np.where(outside, shadow, body / 255.0).astype(np.float32)
body_a = (body / 255.0).astype(np.float32)
SX0, SY0, SX1, SY1 = 560, 0, 1060, 1230  # boîte de travail du scanner
sc_rgb, sc_a, body_a = sc_rgb[SY0:SY1, SX0:SX1], sc_a[SY0:SY1, SX0:SX1], body_a[SY0:SY1, SX0:SX1]
SC_CX, SC_CY = 780 - SX0, 600 - SY0  # pivot

# 3) Calques texte : dé-mélange couleur/alpha par rapport au fond propre.
diff = np.abs(f32 - cleanf).max(2)


def layer(x0, y0, x1, y1):
    d = diff[y0:y1, x0:x1]
    ref = max(np.percentile(d, 99.5), 1.0)
    a = np.clip((d - 4) / (ref - 4), 0, 1)
    a = np.where(a < 0.02, 0, a).astype(np.float32)
    c = cleanf[y0:y1, x0:x1] + (f32[y0:y1, x0:x1] - cleanf[y0:y1, x0:x1]) / np.maximum(a, 0.02)[..., None]
    return dict(x=x0, y=y0, rgb=np.clip(c, 0, 255).astype(np.float32), a=a)


L = {
    "logo": layer(50, 45, 450, 160),
    "kicker": layer(55, 195, 650, 250),
    "l1": layer(55, 285, 530, 375),
    "l2": layer(55, 390, 390, 475),
    "l3": layer(55, 490, 320, 578),
    "model": layer(55, 582, 510, 630),
    "f1": layer(55, 685, 400, 735),
    "bar1": layer(55, 736, 400, 752),
    "f2": layer(55, 765, 400, 815),
    "bar2": layer(55, 828, 400, 846),
    "f3": layer(55, 862, 400, 915),
    "f4": layer(55, 925, 500, 975),
    "cta": layer(55, 1000, 505, 1105),
    "wave": layer(0, 1180, W, 1300),
    "cdl": layer(125, 1246, 420, 1350),
}
# séparation vague / logo CDL (ils se touchent presque)
L["wave"]["a"][1246 - 1180:, 125:262] = 0
L["wave"]["a"][1272 - 1180:, 125:420] = 0
L["cdl"]["a"][: 1272 - 1246, 262 - 125:] = 0
# ---------------------------------------------------------------- utilitaires
def clamp(v, a=0.0, b=1.0):
    return max(a, min(b, v))


def prog(t, start, dur):
    return clamp((t - start) / dur)


def ease_out(p):
    return 1 - (1 - p) ** 3


def ease_in_out(p):
    return 3 * p * p - 2 * p * p * p


def ease_back(p, s=1.7):
    p -= 1
    return 1 + p * p * ((s + 1) * p + s)


def blend(canvas, rgb, a, x, y, opacity=1.0):
    """Compose rgb/a (float) sur canvas à la position entière (x, y)."""
    h, w = a.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0 or opacity <= 0:
        return
    sa = a[y0 - y:y1 - y, x0 - x:x1 - x, None] * opacity
    canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - sa) + rgb[y0 - y:y1 - y, x0 - x:x1 - x] * sa


def put(canvas, ly, dx=0.0, dy=0.0, opacity=1.0, clip=None, a=None):
    """Pose un calque texte avec décalage ; clip = (x0, y0, x1, y1) absolus."""
    al = ly["a"] if a is None else a
    if clip is not None:
        h, w = al.shape
        m = np.zeros_like(al)
        cx0, cy0 = int(clip[0] - ly["x"] - dx), int(clip[1] - ly["y"] - dy)
        cx1, cy1 = int(clip[2] - ly["x"] - dx), int(clip[3] - ly["y"] - dy)
        m[max(cy0, 0):max(cy1, 0), max(cx0, 0):max(cx1, 0)] = 1
        al = al * m
    blend(canvas, ly["rgb"], al, int(round(ly["x"] + dx)), int(round(ly["y"] + dy)), opacity)


def sweep_mask(shape, pos, width, angle_deg=20):
    """Bande lumineuse diagonale (0..1) centrée sur pos (en px le long de x)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx + yy * math.tan(math.radians(angle_deg))
    return np.exp(-((u - pos) / width) ** 2).astype(np.float32)


YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def radial(cx, cy, r):
    return np.exp(-(((XX - cx) ** 2 + (YY - cy) ** 2) / (r * r))).astype(np.float32)


GLOW_PURPLE = radial(820, 520, 300)
GLOW_BLUE = radial(760, 760, 360)
PURPLE = np.array([200, 70, 150], np.float32)  # BGR
BLUE = np.array([235, 170, 40], np.float32)
WHITE = np.full(3, 255, np.float32)

rng = np.random.default_rng(7)
PARTICLES = [dict(x=rng.uniform(560, 1060), y0=rng.uniform(0, H), r=rng.uniform(2, 6),
                  sp=rng.uniform(25, 70), ph=rng.uniform(0, 6.28), col=[BLUE, PURPLE, WHITE][i % 3])
             for i in range(26)]


# ---------------------------------------------------------------- rendu image
def frame(t):
    c = cleanf.copy()

    # fond : légère respiration + halos colorés
    pulse = 0.5 + 0.5 * math.sin(t * 2 * math.pi / 3.2)
    g_in = ease_out(prog(t, 0.0, 1.2))
    gp = (0.16 + 0.10 * pulse) * g_in * GLOW_PURPLE
    gb = (0.12 + 0.08 * (1 - pulse)) * g_in * GLOW_BLUE
    c = c * (1 - gp[..., None]) + PURPLE * gp[..., None]
    c = c * (1 - gb[..., None]) + BLUE * gb[..., None]

    # particules de lumière qui montent derrière le scanner
    for p in PARTICLES:
        y = (p["y0"] - p["sp"] * t) % (H + 40) - 20
        x = p["x"] + 14 * math.sin(t * 0.9 + p["ph"])
        op = 0.55 * g_in * (0.5 + 0.5 * math.sin(t * 2.2 + p["ph"]))
        r = int(p["r"] * 4)
        x0, y0 = int(x) - r, int(y) - r
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
        a = np.exp(-(xx * xx + yy * yy) / (p["r"] ** 2)).astype(np.float32)
        blend(c, np.broadcast_to(p["col"], a.shape + (3,)), a, x0, y0, op)

    # vague : se dessine de gauche à droite
    pw = ease_in_out(prog(t, 0.2, 1.6))
    put(c, L["wave"], clip=(0, 0, pw * W, H))
    if 0 < pw < 1:  # tête lumineuse au bout du tracé
        hx = pw * W
        a = np.exp(-((XX[1180:1300] - hx) / 18) ** 2) * L["wave"]["a"]
        blend(c, np.broadcast_to(WHITE, a.shape + (3,)), a.astype(np.float32), 0, 1180, 0.9)

    # scanner : arrivée par le bas puis flottement
    pin = prog(t, 0.25, 1.4)
    e = ease_back(pin, 1.2) if pin < 1 else 1.0
    float_y = 9 * math.sin((t - 1.65) * 2 * math.pi / 4.0) * clamp((t - 1.65) / 0.8)
    rot = 0.9 * math.sin((t - 1.65) * 2 * math.pi / 5.0) * clamp((t - 1.65) / 0.8)
    dy = (1 - e) * 160 + float_y
    scale = 0.9 + 0.1 * e
    M = cv2.getRotationMatrix2D((SC_CX, SC_CY), rot, scale)
    M[1, 2] += dy
    size = (SX1 - SX0, SY1 - SY0)
    pre = sc_rgb * sc_a[..., None]
    wa = cv2.warpAffine(sc_a, M, size, flags=cv2.INTER_LINEAR)
    wrgb = cv2.warpAffine(pre, M, size, flags=cv2.INTER_LINEAR) / np.maximum(wa, 1e-4)[..., None]
    wbody = cv2.warpAffine(body_a, M, size, flags=cv2.INTER_LINEAR)
    sop = ease_out(clamp(pin * 1.6))
    blend(c, wrgb, wa, SX0, SY0, sop)
    # reflets qui balayent la coque
    for ts in (1.7, 5.6, 8.6):
        ps = prog(t, ts, 0.9)
        if 0 < ps < 1:
            band = sweep_mask(wbody.shape, -300 + ps * 900, 55, -25) * wbody * 0.75
            blend(c, np.broadcast_to(WHITE, band.shape + (3,)), band, SX0, SY0, sop)
    # anneau de scan bleu qui descend le long du scanner à l'arrivée
    ps = prog(t, 1.2, 1.3)
    if 0 < ps < 1:
        sy = 60 + ease_in_out(ps) * 1080
        band = np.exp(-((np.arange(size[1], dtype=np.float32)[:, None] - sy) / 10) ** 2) * wbody
        band = band * math.sin(ps * math.pi)
        blend(c, np.broadcast_to(BLUE, band.shape + (3,)), band.astype(np.float32), SX0, SY0, 0.7)

    # logo Owandy
    p = ease_out(prog(t, 0.5, 0.8))
    put(c, L["logo"], dx=-40 * (1 - p), opacity=p)

    # sur-titre : révélation par volet
    p = ease_in_out(prog(t, 0.9, 0.7))
    put(c, L["kicker"], clip=(0, 0, 55 + p * 600, H))

    # titre : lignes qui montent derrière un masque
    for key, ts in (("l1", 1.2), ("l2", 1.4), ("l3", 1.6)):
        p = ease_out(prog(t, ts, 0.7))
        ly = L[key]
        h = ly["a"].shape[0]
        put(c, ly, dy=h * (1 - p), opacity=clamp(p * 1.5), clip=(0, ly["y"], W, ly["y"] + h))
    # « liberté » scintille
    for ts in (2.6, 6.4):
        ps = prog(t, ts, 0.8)
        if 0 < ps < 1:
            ly = L["l3"]
            band = sweep_mask(ly["a"].shape, -60 + ps * 380, 28, 25) * ly["a"]
            blend(c, np.broadcast_to(np.array([255, 235, 255], np.float32), band.shape + (3,)),
                  band, ly["x"], ly["y"], 0.85)

    p = ease_out(prog(t, 2.0, 0.6))
    put(c, L["model"], dy=14 * (1 - p), opacity=p)

    # caractéristiques + filets qui se tracent
    for key, ts in (("f1", 2.4), ("f2", 2.65), ("f3", 2.9), ("f4", 3.15)):
        p = ease_out(prog(t, ts, 0.6))
        put(c, L[key], dx=-50 * (1 - p), opacity=p)
    for key, ts in (("bar1", 2.55), ("bar2", 2.8)):
        p = ease_in_out(prog(t, ts, 0.6))
        put(c, L[key], clip=(0, 0, 55 + p * 345, H))

    # bouton : rebond, pulsation, éclat
    p = prog(t, 3.5, 0.7)
    if p > 0:
        ly = L["cta"]
        s = ease_back(p, 2.2) * (1 + 0.025 * math.sin(max(0, t - 4.2) * 2 * math.pi / 1.6) * clamp(t - 4.2))
        h, w = ly["a"].shape
        cx, cy = w / 2, h / 2
        M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
        pad = 40
        M[:, 2] += pad
        size = (w + 2 * pad, h + 2 * pad)
        a = cv2.warpAffine(ly["a"], M, size)
        rgb = cv2.warpAffine(ly["rgb"] * ly["a"][..., None], M, size) / np.maximum(a, 1e-4)[..., None]
        # halo sous le bouton
        glow = cv2.GaussianBlur(a, (0, 0), 14) * (0.25 + 0.2 * pulse) * clamp(p * 2)
        blend(c, np.broadcast_to(np.array([200, 80, 140], np.float32), glow.shape + (3,)), glow,
              ly["x"] - pad, ly["y"] - pad + 6)
        blend(c, rgb, a, ly["x"] - pad, ly["y"] - pad, clamp(p * 2))
        for ts in (4.3, 7.4):
            ps = prog(t, ts, 0.7)
            if 0 < ps < 1:
                band = sweep_mask(a.shape, -80 + ps * 640, 30, 25) * a * 0.6
                blend(c, np.broadcast_to(WHITE, a.shape + (3,)), band, ly["x"] - pad, ly["y"] - pad)

    # signature CDL
    p = ease_out(prog(t, 3.9, 0.8))
    put(c, L["cdl"], dy=16 * (1 - p), opacity=p)

    # fondu d'ouverture depuis le blanc
    fo = 1 - ease_out(prog(t, 0, 0.4))
    if fo > 0:
        c = c * (1 - fo) + 255 * fo
    return np.clip(c, 0, 255).astype(np.uint8)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
    n = int(DUR * FPS)
    for i in range(n):
        proc.stdin.write(frame(i / FPS).tobytes())
        if i % 30 == 0:
            print(f"  {i / FPS:4.1f}s / {DUR:.0f}s", flush=True)
    proc.stdin.close()
    proc.wait()
    print("OK ->", OUT)


if __name__ == "__main__":
    if "--still" in sys.argv:  # aperçu : python3 build_owandy.py --still 4.5 out.png
        i = sys.argv.index("--still")
        cv2.imwrite(sys.argv[i + 2], frame(float(sys.argv[i + 1])))
    else:
        main()
