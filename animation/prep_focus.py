#!/usr/bin/env python3
"""Découpe l'affiche « Focus Impression » pour l'animation « éclatée ».

Produit dans img/focus/ :
  - fond.png : l'affiche sans la carte Conférence, l'horaire et les imprimantes
  - carte.png, heure.png, midas.png, pro.png, nano.png : éléments détourés
  - pieces.json : position de chaque élément dans l'affiche
Le fond de la zone de droite est uniforme horizontalement : la couleur de fond
de chaque ligne est lue en x=1077.
"""
import json
import os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "img/focus")
P = np.asarray(Image.open(os.path.join(ROOT, "affiche-focus.png")).convert("RGB")).astype(int)
H, W, _ = P.shape
bg = P[:, 1077, :]
diff = np.abs(P - bg[:, None, :]).max(2)


def soft(d, lo, hi):
    return np.clip((d - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)


pieces = {}


def save(name, alpha, box):
    x0, y0, x1, y1 = box
    rgba = np.zeros((y1 - y0, x1 - x0, 4), np.uint8)
    rgba[..., :3] = P[y0:y1, x0:x1]
    rgba[..., 3] = alpha[y0:y1, x0:x1]
    Image.fromarray(rgba).save(os.path.join(OUT, name + ".png"))
    pieces[name] = {"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0)}


# --- Carte « Conférence Focus Impression » et horaire ---------------------
CARD = (490, 272, 1026, 472)
TIME = (568, 476, 920, 548)
save("carte", soft(diff, 2, 30), CARD)
save("heure", soft(diff, 4, 30), TIME)

# --- Imprimantes : silhouettes (bords doux, lignes remplies) ---------------
ys, ye = 549, 990
edge = soft(diff, 5, 18)
mask = diff > 9
X = np.arange(W)[None, :]
Y = np.arange(H)[:, None]
zones = {
    # Midas à gauche, NanoPrint en haut à droite, la grande imprimante au centre
    "midas": (X >= 345) & (X < 545) & (Y < 862),
    "nano": (X >= 843) & (Y < 842),
}
zones["pro"] = (X >= 556) & ~zones["nano"]
BOX = {"midas": (345, 505, 545, 862), "nano": (843, 528, 1066, 842), "pro": (556, ys, 1066, ye)}
in_time = (X >= TIME[0]) & (X < TIME[2]) & (Y >= TIME[1]) & (Y < TIME[3])
sil_all = np.zeros((H, W), bool)
for name, zone in zones.items():
    x0, y0, x1, y1 = BOX[name]
    zone = zone & ~in_time & (Y >= y0) & (Y < y1)
    z = zone & mask
    alpha = np.where(zone, edge, 0).astype(np.uint8)
    for y in range(y0, y1):
        xs = np.nonzero(z[y, x0:x1])[0]
        if len(xs) < 2 or (name == "midas" and y < 580):   # poignée ajourée de la Midas
            continue
        alpha[y, x0 + xs[0] + 2: x0 + xs[-1] - 1] = 255
    xs = np.nonzero(alpha[y0:y1, x0:x1].any(0))[0]
    yy = np.nonzero(alpha[y0:y1, x0:x1].any(1))[0]
    box = (x0 + xs[0], y0 + yy[0], x0 + xs[-1] + 1, y0 + yy[-1] + 1)
    save(name, alpha, box)
    sil_all |= alpha > 0

# --- Fond nettoyé -------------------------------------------------------------
fond = P.copy()
for x0, y0, x1, y1 in (CARD, TIME):
    fond[y0:y1, x0:x1] = bg[y0:y1, None, :]
grow = sil_all.copy()
for dy in (-2, -1, 1, 2):
    grow |= np.roll(sil_all, dy, 0)
for dx in (-3, -2, -1, 1, 2, 3):
    grow |= np.roll(sil_all, dx, 1)
fond[grow] = np.broadcast_to(bg[:, None, :], P.shape)[grow]
Image.fromarray(fond.astype(np.uint8)).save(os.path.join(OUT, "fond.png"))

with open(os.path.join(OUT, "pieces.json"), "w") as fh:
    json.dump(pieces, fh, indent=1)
print(json.dumps(pieces))
