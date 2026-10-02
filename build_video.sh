#!/usr/bin/env bash
# Génère la vidéo de présentation du cabinet ("après" travaux).
# Format vertical 1080x1920 (Reels / Stories / TikTok), 30 i/s.
set -euo pipefail
cd "$(dirname "$0")"

W=1080; H=1920; FPS=30
D=4.2        # durée de chaque plan (s)
XF=0.9       # durée des fondus enchaînés (s)
SERIF=/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf
SANS=/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf
OUT=${1:-video/cabinet-apres.mp4}
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
mkdir -p "$(dirname "$OUT")"

# photo | décalage horizontal du recadrage 9:16 (0..1) | mouvement | surtitre | titre
SHOTS=(
  "couloir.jpg|0.5|in|INAUGURATION|Découvrez notre nouveau cabinet"
  "salle-attente.jpg|0.35|out|SALLE D’ATTENTE|Un espace pensé pour votre confort"
  "couloir.jpg|0.5|in|ACCUEIL|Lumière, douceur et sérénité"
  "soins-1.jpg|0.45|out|SALLES DE SOINS|Équipements de dernière génération"
  "soins-2.jpg|0.4|in|PRÉCISION|Microscope opératoire et imagerie"
  "soins-3.jpg|0.55|out|CONFORT|Un cadre apaisant pour chaque soin"
  "soins-5.jpg|0.4|in||"
)

spaced() { python3 -c 'import sys; print("\u2003".join(" ".join(w) for w in sys.argv[1].split()))' "$1"; }   # espacement des lettres
esc() { printf '%s' "$1" | sed "s/\\\\/\\\\\\\\/g; s/'/\\\\\\\\\\\\'/g; s/:/\\\\:/g; s/%/\\\\%/g"; }

FR=$(awk "BEGIN{print int($D*$FPS)}")
i=0
for s in "${SHOTS[@]}"; do
  IFS='|' read -r img off mv kicker title <<<"$s"
  if [ "$mv" = in ]; then Z="1+0.10*on/$FR"; else Z="1.10-0.10*on/$FR"; fi
  # Recadrage 9:16 en haute résolution (2x) pour un zoom fluide, puis zoompan.
  BASE="scale=-2:$((H*2)),crop=$((W*2)):$((H*2)):(iw-ow)*$off:0,
      zoompan=z='$Z':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=$FR:s=${W}x${H}:fps=$FPS,
      eq=contrast=1.04:saturation=1.06:brightness=0.01"
  TXT="null"
  if [ -n "$title" ]; then
    T0=0.5; A="if(lt(t,$T0),0,if(lt(t,$T0+0.8),(t-$T0)/0.8,if(lt(t,$D-0.6),1,max(0,($D-t)/0.6))))"
    K=$(esc "$(spaced "$kicker")"); TT=$(esc "$title")
    TXT="drawtext=fontfile=$SANS:text='$K':fontsize=32:fontcolor=0xE9D7B5:alpha='$A':x=(w-tw)/2:y=h*0.76:shadowcolor=black@0.5:shadowx=0:shadowy=2,
        drawbox=x=(iw-120)/2:y=ih*0.76+54:w=120:h=2:color=0xE9D7B5@0.9:t=fill:enable='between(t,$T0+0.3,$D-0.4)',
        drawtext=fontfile=$SERIF:text='$TT':fontsize=58:fontcolor=white:alpha='$A':x=(w-tw)/2:y=h*0.76+86:shadowcolor=black@0.6:shadowx=0:shadowy=3"
  fi
  flat() { echo "$1" | tr -d '\n' | sed 's/  */ /g'; }
  # Dégradé sombre en bas (sous le texte) pour la lisibilité
  ffmpeg -loglevel error -y -loop 1 -i "photos/$img" \
    -f lavfi -i "color=black:s=${W}x${H}:d=$D,format=rgba,geq=r=0:g=0:b=0:a='if(gt(Y,H*0.55),255*0.8*pow((Y-H*0.55)/(H*0.45),1.2),0)'" \
    -filter_complex "[0:v]$(flat "$BASE")[b];[b][1:v]overlay=format=auto,$(flat "$TXT"),format=yuv420p,setsar=1[o]" \
    -map "[o]" -t $D -r $FPS -c:v libx264 -preset medium -crf 16 "$TMP/s$i.mp4"
  echo "plan $i ok ($img)"
  i=$((i+1))
done

# Carton final
END=4.5
ffmpeg -loglevel error -y -f lavfi -i "color=0x1b1714:s=${W}x${H}:d=$END:r=$FPS" -vf "
  drawtext=fontfile=$SANS:text='$(esc "$(spaced "BIENVENUE")")':fontsize=36:fontcolor=0xE9D7B5:x=(w-tw)/2:y=h/2-130:alpha='min(1,t/0.8)',
  drawbox=x=(iw-120)/2:y=ih/2-60:w=120:h=2:color=0xE9D7B5@0.9:t=fill,
  drawtext=fontfile=$SERIF:text='$(esc "Nous avons hâte")':fontsize=78:fontcolor=white:x=(w-tw)/2:y=h/2-10:alpha='min(1,max(0,(t-0.4)/0.8))',
  drawtext=fontfile=$SERIF:text='$(esc "de vous accueillir")':fontsize=78:fontcolor=white:x=(w-tw)/2:y=h/2+80:alpha='min(1,max(0,(t-0.6)/0.8))',
  format=yuv420p" -c:v libx264 -preset medium -crf 16 "$TMP/s$i.mp4"
N=$((i+1))

# Fondus enchaînés entre tous les plans
INPUTS=(); for ((k=0;k<N;k++)); do INPUTS+=(-i "$TMP/s$k.mp4"); done
FC=""; prev="[0:v]"; t=0
for ((k=1;k<N;k++)); do
  t=$(awk "BEGIN{print $t+$D-$XF}")
  FC+="${prev}[$k:v]xfade=transition=fade:duration=$XF:offset=$t[x$k];"
  prev="[x$k]"
done
FC+="${prev}fade=t=in:st=0:d=0.6,fade=t=out:st=$(awk "BEGIN{print $t+$END-0.8}"):d=0.8[out]"
ffmpeg -loglevel error -y "${INPUTS[@]}" -filter_complex "$FC" -map "[out]" \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart "$OUT"
echo "→ $OUT ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT")s)"
