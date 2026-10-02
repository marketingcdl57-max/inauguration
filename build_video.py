#!/usr/bin/env python3
"""Génère la vidéo de présentation du cabinet ("après" travaux).

Format vertical 1080x1920 (Reels / Stories / TikTok), 30 i/s, sans texte
sur les photos. Chaque photo n'apparaît qu'une seule fois :
couloir (travelling avant) -> salles de soins -> salle d'attente, qui se
floute progressivement pour laisser place à la signature
« Installation réalisée par » + logo Comptoir Dentaire Lorrain.

Transitions : uniquement des fondus enchaînés, rendus fluides par la
continuité du mouvement de caméra :
  - « zoom-through » : le plan sortant accélère vers l'avant, le plan
    entrant arrive zoomé et se pose en douceur ;
  - « match-pan » : les deux plans glissent dans le même sens pendant le fondu.
"""
import os
import subprocess
import sys
import tempfile

W, H, FPS = 1080, 1920, 30
SANS = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "video/cabinet-apres.mp4")
LOGO = os.path.join(ROOT, "assets/logo-cdl.png")
SIGNATURE = "Installation réalisée par"

# Courbes de mouvement en fonction de la progression p (0..1) du plan.
ARRIVE = "0.24*pow(1-{p},3)"                          # arrive zoomé, se pose
LEAVE = "0.9*pow(max(0,({p}-0.72)/0.28),3)"           # accélère vers l'avant

ZOOM_XF = 0.5    # fondu court pendant un zoom-through
PAN_XF = 1.0     # fondu long pendant un match-pan

SHOTS = [
    # Marche dans le couloir : zoom qui accélère (perspective), léger
    # balancement de pas, flou de mouvement.
    dict(img="couloir.jpg", dur=5.6, off=0.5, scale=3, bob=True, blur=True,
         zoom="1/(1-0.58*{p})", fx="0.5+0.12*{p}", fy="0.5-0.07*{p}", xf=0.45),
    # On entre dans la salle de soins, puis panoramique lent vers la droite.
    dict(img="soins-2.jpg", dur=4.6, off=0.42,
         zoom="1.14+" + ARRIVE, fx="0.45+0.10*{p}", fy="0.52", xf=PAN_XF),
    # Même panoramique, puis poussée vers l'avant.
    dict(img="soins-1.jpg", dur=4.6, off=0.45, blur=True,
         zoom="1.14*(1+" + LEAVE + ")", fx="0.45+0.08*{p}", fy="0.5-0.04*{p}", xf=ZOOM_XF),
    dict(img="soins-3.jpg", dur=4.6, off=0.55,
         zoom="1.14+" + ARRIVE, fx="0.45+0.10*{p}", fy="0.55", xf=PAN_XF),
    dict(img="soins-5.jpg", dur=4.6, off=0.6, blur=True,
         zoom="1.14*(1+" + LEAVE + ")", fx="0.45+0.08*{p}", fy="0.47", xf=ZOOM_XF),
    # Dernier plan : se floute progressivement, puis signature + logo.
    dict(img="salle-attente.jpg", dur=9.0, off=0.35, final=True,
         zoom="1.04+" + ARRIVE + "+0.05*{p}", fx="0.5", fy="0.5", xf=0),
]

BLUR_START, BLUR_END, BLUR_MAX = 2.4, 4.6, 32
TEXT_IN, LOGO_IN = 4.2, 4.7


def run(args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def camera(s):
    n = int(round(s["dur"] * FPS))
    p = f"(on/{n - 1})"
    zoom, fx, fy = (s[k].format(p=p) for k in ("zoom", "fx", "fy"))
    k = s.get("scale", 2)
    y = f"ih*({fy})-ih/zoom/2"
    if s.get("bob"):
        y += f"+ih*0.0025*sin(2*PI*1.8*on/{FPS})"   # balancement de pas (~1,8 pas/s)
    f = (f"scale=-2:{H * k},crop={W * k}:{H * k}:(iw-ow)*{s['off']}:0,"
         f"zoompan=z='{zoom}':x='iw*({fx})-iw/zoom/2':y='{y}':d={n}:s={W}x{H}:fps={FPS},"
         "eq=contrast=1.04:saturation=1.06:brightness=0.01")
    if s.get("blur"):
        f += ",tmix=frames=3:weights='1 2 1'"         # flou de mouvement léger
    return f


def render_shot(s, path, tmp):
    src = ["-loop", "1", "-i", os.path.join(ROOT, "photos", s["img"])]
    if not s.get("final"):
        run([*src, "-vf", camera(s) + ",format=yuv420p,setsar=1", "-t", str(s["dur"]),
             "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "15", path])
        return

    # Flou progressif (sigma piloté image par image) + voile blanc,
    # puis texte et logo centrés.
    cmds = os.path.join(tmp, "blur.cmd")
    with open(cmds, "w") as fh:
        fh.write(f"{BLUR_START}-{BLUR_END} [expr] gblur@b sigma "
                 f"'{BLUR_MAX}*TI*TI*(3-2*TI)';\n"
                 f"{BLUR_END} gblur@b sigma {BLUR_MAX};\n")
    d = BLUR_END - BLUR_START
    logo_w = 820
    fc = ";".join([
        f"[0:v]{camera(s)},sendcmd=f='{cmds}',gblur@b=sigma=0:steps=3[img]",
        f"color=white:s={W}x{H}:d={s['dur']}:r={FPS},format=rgba,"
        f"colorchannelmixer=aa=0.62,fade=t=in:st={BLUR_START}:d={d}:alpha=1[veil]",
        f"[1:v]scale={logo_w}:-1,format=rgba,fade=t=in:st={LOGO_IN}:d=1.0:alpha=1[logo]",
        "[img][veil]overlay=format=auto[bg]",
        f"[bg]drawtext=fontfile={SANS}:text='{SIGNATURE}':fontsize=54:fontcolor=0x2b2b2b:"
        f"x=(w-tw)/2:y=h/2-200:alpha='min(1,max(0,(t-{TEXT_IN})/0.9))'[txt]",
        "[txt][logo]overlay=x=(W-w)/2:y=H/2-105:format=auto,format=yuv420p,setsar=1[o]",
    ])
    run([*src, "-loop", "1", "-i", LOGO, "-filter_complex", fc, "-map", "[o]",
         "-t", str(s["dur"]), "-r", str(FPS),
         "-c:v", "libx264", "-preset", "medium", "-crf", "15", path])


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        clips = []
        for i, s in enumerate(SHOTS):
            path = os.path.join(tmp, f"s{i}.mp4")
            render_shot(s, path, tmp)
            clips.append(path)
            print(f"plan {i} ok ({s['img']})", flush=True)

        inputs, fc, prev, t = [], [], "[0:v]", 0.0
        for c in clips:
            inputs += ["-i", c]
        for i in range(1, len(SHOTS)):
            xf = SHOTS[i - 1]["xf"]
            t += SHOTS[i - 1]["dur"] - xf
            fc.append(f"{prev}[{i}:v]xfade=transition=fade:duration={xf}:offset={t:.3f}[x{i}]")
            prev = f"[x{i}]"
        total = t + SHOTS[-1]["dur"]
        fc.append(f"{prev}fade=t=in:st=0:d=0.6[out]")
        run([*inputs, "-filter_complex", ";".join(fc), "-map", "[out]",
             "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", OUT])
    print(f"→ {OUT} ({total:.1f}s)")


if __name__ == "__main__":
    main()
