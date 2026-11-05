#!/usr/bin/env python3
"""Génère la vidéo de présentation du cabinet ("après" travaux).

Format vertical 1080x1920 (Reels / Stories / TikTok), 30 i/s, sans texte
sur les photos. Chaque photo n'apparaît qu'une seule fois :
couloir (travelling avant fluide) -> salles de soins -> salle d'attente, qui se
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
SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "video/cabinet-apres.mp4")
LOGO = os.path.join(ROOT, "assets/logo-cdl.png")
SIGNATURE = "Installation réalisée par"
INTRO_KICKER = "C A B I N E T\u2003D U"     # lettres espacées
INTRO_NAME = "Docteur Tapp"

# Courbes de mouvement en fonction de la progression p (0..1) du plan.
ARRIVE = "0.24*pow(1-{p},3)"                          # arrive zoomé, se pose
LEAVE = "0.9*pow(max(0,({p}-0.72)/0.28),3)"           # accélère vers l'avant

ZOOM_XF = 0.5    # fondu court pendant un zoom-through
PAN_XF = 1.0     # fondu long pendant un match-pan

SHOTS = [
    # Couloir : travelling avant fluide (type rail), puis poussée vers la
    # porte qui enchaîne sur la salle de soins.
    # Ouverture : l'image est d'abord floutée sous un voile clair avec le nom
    # du cabinet, puis redevient nette et le travelling démarre.
    dict(img="couloir.jpg", dur=4.0 + 3.2, intro=3.2, off=0.5, scale=3, blur=True,
         zoom="1+0.18*{p}+1.0*pow(max(0,({p}-0.6)/0.4),3)", fx="0.5+0.12*{p}", fy="0.5-0.07*{p}", xf=0.45),
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
INTRO_TEXT_IN, INTRO_TEXT_OUT = 0.4, 2.2       # nom du cabinet
INTRO_CLEAR = 1.8                               # début du retour au net
TEXT_IN, LOGO_IN = 4.2, 4.7


def run(args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def camera(s):
    n = int(round(s["dur"] * FPS))
    i = int(round(s.get("intro", 0) * FPS))     # caméra immobile pendant l'ouverture
    p = f"(max(0,on-{i})/{n - 1 - i})"
    zoom, fx, fy = (s[k].format(p=p) for k in ("zoom", "fx", "fy"))
    k = s.get("scale", 2)
    y = f"ih*({fy})-ih/zoom/2"
    f = (f"scale=-2:{H * k},crop={W * k}:{H * k}:(iw-ow)*{s['off']}:0,"
         f"zoompan=z='{zoom}':x='iw*({fx})-iw/zoom/2':y='{y}':d={n}:s={W}x{H}:fps={FPS},"
         "eq=contrast=1.04:saturation=1.06:brightness=0.01")
    if s.get("blur"):
        f += ",tmix=frames=3:weights='1 2 1'"         # flou de mouvement léger
    return f


def render_shot(s, path, tmp):
    src = ["-loop", "1", "-i", os.path.join(ROOT, "photos", s["img"])]
    if s.get("intro"):
        render_intro(s, src, path, tmp)
        return
    if not s.get("final"):
        run([*src, "-vf", camera(s) + ",format=yuv420p,setsar=1", "-t", str(s["dur"]),
             "-r", str(FPS), "-c:v", "libx264", "-preset", "medium", "-crf", "15", path])
        return

    # Flou progressif (sigma piloté image par image) + voile blanc,
    # puis texte et logo centrés.
    cmds = os.path.join(tmp, "blur.cmd")
    with open(cmds, "w") as fh:
        e = f"{BLUR_MAX}*TI*TI*(3-2*TI)"
        fh.write(f"{BLUR_START}-{BLUR_END} [expr] gblur@b sigma '{e}', [expr] gblur@b sigmaV '{e}';\n"
                 f"{BLUR_END} gblur@b sigma {BLUR_MAX}, gblur@b sigmaV {BLUR_MAX};\n")
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


def render_intro(s, src, path, tmp):
    # Inverse de la fin : flou + voile clair qui se dissipent pendant que le
    # nom du cabinet apparaît puis s'efface.
    end = s["intro"]
    cmds = os.path.join(tmp, "intro.cmd")
    with open(cmds, "w") as fh:
        e = f"{BLUR_MAX}*(1-TI*TI*(3-2*TI))"
        fh.write(f"{INTRO_CLEAR}-{end} [expr] gblur@b sigma '{e}', [expr] gblur@b sigmaV '{e}';\n"
                 f"{end} gblur@b sigma 0, gblur@b sigmaV 0;\n")
    a = (f"if(lt(t,{INTRO_TEXT_IN}),0,if(lt(t,{INTRO_TEXT_IN}+0.9),(t-{INTRO_TEXT_IN})/0.9,"
         f"if(lt(t,{INTRO_TEXT_OUT}-0.7),1,max(0,({INTRO_TEXT_OUT}-t)/0.7))))")
    fc = ";".join([
        f"[0:v]{camera(s)},sendcmd=f='{cmds}',gblur@b=sigma={BLUR_MAX}:steps=3[img]",
        f"color=white:s={W}x{H}:d={s['dur']}:r={FPS},format=rgba,"
        f"colorchannelmixer=aa=0.62,fade=t=out:st={INTRO_CLEAR}:d={end - INTRO_CLEAR}:alpha=1[veil]",
        "[img][veil]overlay=format=auto[bg]",
        ",".join([
            f"[bg]drawtext=fontfile={SANS}:text='{INTRO_KICKER}':fontsize=36:fontcolor=0x6b5f52:"
            f"x=(w-tw)/2:y=h/2-150:alpha='{a}'",
            f"drawbox=x=(iw-120)/2:y=ih/2-82:w=120:h=2:color=0x6b5f52@0.8:t=fill:"
            f"enable='between(t,{INTRO_TEXT_IN}+0.3,{INTRO_TEXT_OUT}-0.4)'",
            f"drawtext=fontfile={SERIF}:text='{INTRO_NAME}':fontsize=124:fontcolor=0x2b2b2b:"
            f"x=(w-tw)/2:y=h/2-40:alpha='{a}'",
            "format=yuv420p,setsar=1[o]",
        ]),
    ])
    run([*src, "-filter_complex", fc, "-map", "[o]", "-t", str(s["dur"]), "-r", str(FPS),
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
        fc.append(f"{prev}fade=t=in:st=0:d=0.6:color=white[out]")
        run([*inputs, "-filter_complex", ";".join(fc), "-map", "[out]",
             "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", OUT])
    print(f"→ {OUT} ({total:.1f}s)")


if __name__ == "__main__":
    main()
