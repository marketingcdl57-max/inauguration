#!/usr/bin/env python3
"""Variante « éclatée » du post conférence Scanner intra-oral.

Format 1080x1350, 30 i/s, ~6 s. Reprend le découpage en calques de
build_post_anime.py : au départ tous les éléments apparaissent éparpillés
autour du centre (décalés, tournés, un peu réduits), puis ils se rassemblent ensemble
à leur place avec un léger « coup » au moment de l'assemblage. Reflet
lumineux sur les encarts, LED du scanner qui pulse, et l'image se pose
exactement sur le visuel d'origine.
"""
import math
import os
import random
import subprocess
import sys

from PIL import Image

from build_post_anime import (H, LAYERS, SHINE, SPLIT, SRC, W, FPS,
                              build_layers, ease_out, led_glow, prog, shine)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "video/conference-sidi-mani-eclate.mp4")
DUR = 6.0
CENTER = (W / 2, 620)
APPEAR = 0.3                            # apparition, déjà éclaté
START, FLY, STAGGER = 0.45, 1.0, 0.035  # rassemblement : début, durée, décalage
IMPACT = START + FLY + STAGGER * len(LAYERS)
SHINE_AT = IMPACT + 0.6
SETTLE = 5.4


def scatter():
    """Position de départ de chaque pièce : éloignée du centre, tournée."""
    rnd = random.Random(7)
    pieces = [dict(name=L["name"], box=L["box"]) for L in LAYERS]
    pieces.append(dict(name="bandeau", box=(0, SPLIT, W, H)))
    pieces[-1]["fixed_rot"] = 4
    for i, P in enumerate(pieces):
        x0, y0, x1, y1 = P["box"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        vx, vy = cx - CENTER[0], cy - CENTER[1]
        n = math.hypot(vx, vy) or 1
        dist = 110 + 0.35 * n
        P["dx"], P["dy"] = vx / n * dist, vy / n * dist
        P["rot"] = rnd.choice((-1, 1)) * rnd.uniform(8, 18)
        P["scale"] = rnd.uniform(0.85, 0.95)
        P["rot"] = P.get("fixed_rot", P["rot"])
        P["t"] = START + STAGGER * i
    return pieces


def place(canvas, piece, P, t):
    alpha = prog(t, 0, APPEAR)
    if alpha <= 0:
        return
    k = 1 - ease_out(prog(t, P["t"], FLY))
    x0, y0, x1, y1 = P["box"]
    cx, cy = (x0 + x1) / 2 + P["dx"] * k, (y0 + y1) / 2 + P["dy"] * k
    if k > 0.002:
        s = 1 + (P["scale"] - 1) * k
        w, h = piece.size
        piece = piece.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
        piece = piece.rotate(P["rot"] * k, Image.BICUBIC, expand=True)
    if alpha < 1:
        piece = piece.copy()
        piece.putalpha(piece.getchannel("A").point(lambda v: int(v * alpha)))
    w, h = piece.size
    canvas.alpha_composite(piece, (int(round(cx - w / 2)), int(round(cy - h / 2))))


def frame(t, bg, layers, pieces, final):
    canvas = bg.copy()
    # Le bandeau du bas d'abord : les pièces du haut passent par-dessus en vol.
    for P in pieces[-1:] + pieces[:-1]:
        place(canvas, layers[P["name"]], P, t)
    boxes = {L["name"]: L["box"] for L in LAYERS}
    for i, (name, _) in enumerate(SHINE):
        shine(canvas, layers[name], boxes[name], t, SHINE_AT + 0.25 * i)
    led_glow(canvas, t)

    img = canvas.convert("RGB")
    # Petit « coup » à l'assemblage : zoom 2 % qui retombe.
    q = prog(t, IMPACT - 0.1, 0.5)
    z = 1 + 0.02 * math.sin(math.pi * q) * (1 - q) if 0 < q < 1 else 1
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
    bg, layers, panel = build_layers(final)
    layers["bandeau"] = panel
    pieces = scatter()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", OUT],
        stdin=subprocess.PIPE)
    for i in range(int(DUR * FPS)):
        ff.stdin.write(frame(i / FPS, bg, layers, pieces, final).tobytes())
    ff.stdin.close()
    if ff.wait():
        sys.exit("ffmpeg a échoué")
    print(f"→ {OUT} ({DUR:.1f}s)")


if __name__ == "__main__":
    main()
