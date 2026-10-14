#!/usr/bin/env python3
"""Découpe l'affiche Maxim Di FEDERICO pour l'animation « éclatée ».

Produit dans img/federico/ :
  - fond.png      : l'affiche sans la carte Conférence, l'heure et l'imprimante
  - carte.png, heure.png : ces éléments détourés
  - poignee.png, flacon1-3.png, ecran.png, corps.png, cartouche.png :
    les pièces de l'imprimante Midas (vue éclatée)
  - pieces.json   : position de chaque élément dans l'affiche
Le fond de l'affiche est uniforme horizontalement à droite : la couleur de
fond de chaque ligne est lue en x=1075.
"""
import json
import os
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "img/federico")
P = Image.open(os.path.join(ROOT, "affiche-federico.png")).convert("RGB")
W, H = P.size
px = P.load()
bg = [px[1075, y] for y in range(H)]


def diff(x, y):
    c, b = px[x, y], bg[y]
    return max(abs(c[i] - b[i]) for i in range(3))


def soft(d, lo=5, hi=18):
    return max(0, min(255, int((d - lo) * 255 / (hi - lo))))


def cutout(box, alpha_fn):
    x0, y0, x1, y1 = box
    im = Image.new("RGBA", (x1 - x0, y1 - y0))
    o = im.load()
    for y in range(y0, y1):
        for x in range(x0, x1):
            a = alpha_fn(x, y)
            if a:
                o[x - x0, y - y0] = (*px[x, y], a)
    return im


pieces = {}


def save(name, im, x, y):
    im.save(os.path.join(OUT, name + ".png"))
    pieces[name] = {"x": x, "y": y, "w": im.width, "h": im.height}


# --- Silhouette de l'imprimante (lignes remplies entre les bords) ---------
PB = (660, 612, 862, 986)
sil = {}
for y in range(PB[1], PB[3]):
    xs = [x for x in range(PB[0], PB[2]) if diff(x, y) > 10]
    if not xs:
        continue
    for x in range(PB[0], PB[2]):
        d = soft(diff(x, y))
        if y >= 690 and xs[0] + 2 <= x <= xs[-1] - 2:
            d = 255                      # corps plein
        if d:
            sil[(x, y)] = d

BOTTLES = [(712, 633, 744, 692), (744, 633, 774, 692), (774, 633, 806, 692)]
SCREEN = (697, 690, 817, 898)
CART = (781, 902, 850, 982)
inside = lambda r, x, y: r[0] <= x < r[2] and r[1] <= y < r[3]


def part(name, box, keep):
    im = cutout(box, lambda x, y: sil.get((x, y), 0) if keep(x, y) else 0)
    save(name, im, box[0], box[1])


for i, r in enumerate(BOTTLES):
    part(f"flacon{i + 1}", r, lambda x, y, r=r: inside(r, x, y))
part("poignee", (PB[0], PB[1], PB[2], 692),
     lambda x, y: not any(inside(r, x, y) for r in BOTTLES))
part("ecran", SCREEN, lambda x, y: True)
part("cartouche", CART, lambda x, y: True)

# Corps : trous de l'écran et de la cartouche rebouchés pour la vue éclatée.
body_box = (PB[0], 688, PB[2], PB[3])
body = cutout(body_box, lambda x, y: sil.get((x, y), 0)
              if not (inside(SCREEN, x, y) or inside(CART, x, y)) else 0)
b = body.load()
for y in range(SCREEN[1], SCREEN[3]):            # logement d'écran vide
    for x in range(SCREEN[0], SCREEN[2]):
        b[x - body_box[0], y - body_box[1]] = (8, 8, 10, 255)
BOTTOM = max(y for y in range(900, PB[3]) if diff(700, y) > 10)
for y in range(CART[1], BOTTOM + 1):              # flanc du corps derrière la cartouche
    c = px[700, y]
    for x in range(CART[0], 848):
        b[x - body_box[0], y - body_box[1]] = (*c, 255)
save("corps", body, body_box[0], body_box[1])

# --- Carte « Conférence » et horaire ---------------------------------------
CARD = (474, 314, 1026, 516)
card = cutout(CARD, lambda x, y: soft(diff(x, y), 3, 40))
save("carte", card, CARD[0], CARD[1])
TIME = (566, 522, 926, 602)
save("heure", cutout(TIME, lambda x, y: soft(diff(x, y), 4, 30)), TIME[0], TIME[1])

# --- Fond nettoyé -------------------------------------------------------------
fond = P.copy()
f = fond.load()
for box in (CARD, TIME):
    for y in range(box[1], box[3]):
        for x in range(box[0], box[2]):
            f[x, y] = bg[y]
for box in (PB, CART):                            # zone de l'imprimante (que du fond autour)
    for y in range(box[1] - 4, box[3] + 4):
        for x in range(box[0], box[2]):
            f[x, y] = bg[y]
fond.save(os.path.join(OUT, "fond.png"))

with open(os.path.join(OUT, "pieces.json"), "w") as fh:
    json.dump(pieces, fh, indent=1)
print(json.dumps(pieces))
