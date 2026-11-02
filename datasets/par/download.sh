#!/bin/bash
# Download the PAR cohort (BioImage Archive S-BIAD2323) and the per-slide AI grades.
#   bash datasets/par/download.sh <destination> [parallel downloads]
# Resumable: complete files are skipped, partial ones resume (curl -C -). Slides: ~1 TB over the three scanners.
set -u
DEST=${1:?destination}; NPAR=${2:-8}
BASE=https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/323/S-BIAD2323/Files/bia_prostate_wsi
mkdir -p "$DEST"
# file lists and annotations (slide_labels.tsv)
for d in file_lists annotations; do
  wget -q -r -np -nH --cut-dirs=8 -R "index.html*" -P "$DEST" "$BASE/$d/"
done
# AI grades of the three AI graders on the three scanners: ESM 1 of the PAR validation paper (Virchows Arch 2026, CC BY 4.0)
curl -L -o "$DEST/ai_grades.csv" \
  "https://static-content.springer.com/esm/art%3A10.1007%2Fs00428-026-04742-5/MediaObjects/428_2026_4742_MOESM1_ESM.csv"
for comp in component_C_leica_svs:file_list_component_C.tsv component_B_hamamatsu_ndpi:file_list_component_B.tsv \
            component_A_grundium_svs:file_list_component_A_v2.tsv; do
  dir=${comp%%:*}; list=${comp##*:}
  mkdir -p "$DEST/$dir"
  tail -n +2 "$DEST/file_lists/$list" | sed 's#.*/##' | grep -E '\.(svs|ndpi)$' | sort -u |
    xargs -P "$NPAR" -I{} curl -s -C - --retry 8 --retry-delay 3 -o "$DEST/$dir/{}" "$BASE/$dir/{}"
done
