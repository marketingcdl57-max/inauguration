#!/usr/bin/env python3
"""Génère la vidéo de présentation du cabinet ("après" travaux).

Format vertical 1080x1920 (Reels / Stories / TikTok), 30 i/s.
Chaque photo n'apparaît qu'une seule fois :
couloir (travelling avant, comme si on marchait) -> on entre dans la salle
de soins avec le comptoir -> autres salles -> salle d'attente -> carton final.
"""
import os
import subprocess
import sys
import tempfile

W, H, FPS = 1080, 1920, 30
SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"
SANS = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
GOLD = "0xE9D7B5"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "video/cabinet-apres.mp4")

# Progression 0..1 du plan (variable `on` de zoompan) et courbes d'accélération.
P = "(on/{n})"
SMOOTH = "({p}*{p}*(3-2*{p}))"          # ease-in-out

# Chaque plan : photo, durée, décalage du recadrage 9:16, zoom (expression de
# la progression p), point visé (fx, fy) en fractions du cadre, texte,
# transition vers le plan suivant.
SHOTS = [
    dict(img="couloir.jpg", dur=5.6, off=0.5, scale=3,
         # Marche vers le fond du couloir : le zoom accélère naturellement
         # (perspective), avec un léger balancement de pas.
         zoom="1/(1-0.58*{p})", fx="0.5+0.12*{p}", fy="0.5-0.07*{p}",
         bob=True, blur=True,
         kicker="INAUGURATION", title="Découvrez notre nouveau cabinet", text_end=3.6,
         trans=("fade", 0.45)),
    dict(img="soins-2.jpg", dur=4.4, off=0.42,
         # On « entre » dans la pièce : le mouvement se pose en douceur.
         zoom="1+0.32*pow(1-{p},3)", fx="0.5", fy="0.52",
         kicker="SALLES DE SOINS", title="Des espaces lumineux et apaisants",
         trans=("smoothleft", 1.0)),
    dict(img="soins-1.jpg", dur=4.2, off=0.45,
         zoom="1.12-0.12*" + SMOOTH, fx="0.46+0.08*" + SMOOTH, fy="0.5",
         kicker="TECHNOLOGIE", title="Équipements de dernière génération",
         trans=("fade", 1.0)),
    dict(img="soins-3.jpg", dur=4.2, off=0.55,
         zoom="1+0.10*" + SMOOTH, fx="0.52", fy="0.55-0.05*" + SMOOTH,
         kicker="CONFORT", title="Un cadre apaisant pour chaque soin",
         trans=("smoothright", 1.0)),
    dict(img="soins-5.jpg", dur=4.2, off=0.62,
         zoom="1.04+0.10*" + SMOOTH, fx="0.5+0.12*" + SMOOTH, fy="0.45",
         kicker="PRÉCISION", title="Microscope opératoire et imagerie",
         trans=("fade", 1.0)),
    dict(img="salle-attente.jpg", dur=4.8, off=0.35,
         zoom="1.14-0.14*" + SMOOTH, fx="0.5", fy="0.5",
         kicker="SALLE D’ATTENTE", title="Un espace pensé pour votre confort",
         trans=("fadeblack", 1.2)),
]
END_DUR = 4.5


def esc(s):
    return s.replace("\\", "\\\\").replace("'", "’").replace(":", "\\:").replace("%", "\\%")


def spaced(s):
    # Lettres espacées, mots séparés par une espace cadratin.
    return " ".join(" ".join(w) for w in s.split())


def alpha(t0, t1):
    return (f"if(lt(t,{t0}),0,if(lt(t,{t0}+0.7),(t-{t0})/0.7,"
            f"if(lt(t,{t1}-0.5),1,max(0,({t1}-t)/0.5))))")


def text_filters(kicker, title, t0, t1):
    a = alpha(t0, t1)
    y = "h*0.76"
    return ",".join([
        f"drawtext=fontfile={SANS}:text='{esc(spaced(kicker))}':fontsize=32:fontcolor={GOLD}:"
        f"alpha='{a}':x=(w-tw)/2:y={y}:shadowcolor=black@0.5:shadowx=0:shadowy=2",
        f"drawbox=x=(iw-120)/2:y=ih*0.76+54:w=120:h=2:color={GOLD}@0.9:t=fill:"
        f"enable='between(t,{t0}+0.3,{t1}-0.3)'",
        f"drawtext=fontfile={SERIF}:text='{esc(title)}':fontsize=58:fontcolor=white:"
        f"alpha='{a}':x=(w-tw)/2:y={y}+86:shadowcolor=black@0.6:shadowx=0:shadowy=3",
    ])


def run(args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def render_shot(s, path):
    n = int(round(s["dur"] * FPS))
    p = P.format(n=n - 1)
    zoom, fx, fy = (s[k].format(p=p) for k in ("zoom", "fx", "fy"))
    k = s.get("scale", 2)
    cw, ch = W * k, H * k
    y = f"ih*({fy})-ih/zoom/2"
    if s.get("bob"):
        # Balancement de pas discret (~1,8 pas/s).
        y += f"+ih*0.0025*sin(2*PI*1.8*on/{FPS})"
    base = (f"scale=-2:{ch},crop={cw}:{ch}:(iw-ow)*{s['off']}:0,"
            f"zoompan=z='{zoom}':x='iw*({fx})-iw/zoom/2':y='{y}':d={n}:s={W}x{H}:fps={FPS},"
            "eq=contrast=1.04:saturation=1.06:brightness=0.01")
    if s.get("blur"):
        # Flou de mouvement léger quand on accélère vers la porte.
        base += ",tmix=frames=3:weights='1 2 1'"
    xf = s["trans"][1]
    txt = text_filters(s["kicker"], s["title"], 0.5, s.get("text_end", s["dur"] - xf))
    grad = ("color=black:s={W}x{H}:d={d},format=rgba,geq=r=0:g=0:b=0:"
            "a='if(gt(Y,H*0.55),255*0.8*pow((Y-H*0.55)/(H*0.45),1.2),0)'").format(W=W, H=H, d=s["dur"])
    run(["-loop", "1", "-i", os.path.join(ROOT, "photos", s["img"]), "-f", "lavfi", "-i", grad,
         "-filter_complex", f"[0:v]{base}[b];[b][1:v]overlay=format=auto,{txt},format=yuv420p,setsar=1[o]",
         "-map", "[o]", "-t", str(s["dur"]), "-r", str(FPS),
         "-c:v", "libx264", "-preset", "medium", "-crf", "15", path])


def render_end(path):
    vf = ",".join([
        f"drawtext=fontfile={SANS}:text='{esc(spaced('BIENVENUE'))}':fontsize=36:fontcolor={GOLD}:"
        "x=(w-tw)/2:y=h/2-130:alpha='min(1,max(0,(t-0.3)/0.8))'",
        f"drawbox=x=(iw-120)/2:y=ih/2-60:w=120:h=2:color={GOLD}@0.9:t=fill:enable='gte(t,0.6)'",
        f"drawtext=fontfile={SERIF}:text='{esc('Nous avons hâte')}':fontsize=78:fontcolor=white:"
        "x=(w-tw)/2:y=h/2-10:alpha='min(1,max(0,(t-0.7)/0.8))'",
        f"drawtext=fontfile={SERIF}:text='{esc('de vous accueillir')}':fontsize=78:fontcolor=white:"
        "x=(w-tw)/2:y=h/2+80:alpha='min(1,max(0,(t-0.9)/0.8))'",
        "format=yuv420p",
    ])
    run(["-f", "lavfi", "-i", f"color=0x1b1714:s={W}x{H}:d={END_DUR}:r={FPS}", "-vf", vf,
         "-c:v", "libx264", "-preset", "medium", "-crf", "15", path])


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        clips = []
        for i, s in enumerate(SHOTS):
            path = os.path.join(tmp, f"s{i}.mp4")
            render_shot(s, path)
            clips.append(path)
            print(f"plan {i} ok ({s['img']})", flush=True)
        end = os.path.join(tmp, "end.mp4")
        render_end(end)
        clips.append(end)

        inputs, fc, prev, t = [], [], "[0:v]", 0.0
        for c in clips:
            inputs += ["-i", c]
        for i, s in enumerate(SHOTS, start=1):
            name, xf = s["trans"]
            t += s["dur"] - xf
            fc.append(f"{prev}[{i}:v]xfade=transition={name}:duration={xf}:offset={t:.3f}[x{i}]")
            prev = f"[x{i}]"
        total = t + END_DUR
        fc.append(f"{prev}fade=t=in:st=0:d=0.6,fade=t=out:st={total - 0.8:.3f}:d=0.8[out]")
        run([*inputs, "-filter_complex", ";".join(fc), "-map", "[out]",
             "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", OUT])
    print(f"→ {OUT} ({total:.1f}s)")


if __name__ == "__main__":
    main()
