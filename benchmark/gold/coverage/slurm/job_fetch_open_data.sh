#!/bin/bash
#SBATCH --job-name=fetchdata
#SBATCH --partition=generic
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fetch_open_data_%j.out
# Step 5 (benchmark/gold/coverage/PREREG_step5_detector_finetune.md): openly licensed sets from their official sources
# only (no YouTube). Resumable (wget -c); md5 checked for Zenodo; archives deleted after a good unpack.
set -euo pipefail
D="$HOME/open_data"; mkdir -p "$D"/{fsd50k,esc50,musan,librispeech}
cd "$D/fsd50k"
curl -s https://zenodo.org/api/records/4060432 > record.json
python3 - <<'PY' > files.txt
import json
for f in json.load(open("record.json"))["files"]:
    print(f["key"], f["checksum"].split(":")[-1], f["links"]["self"])
PY
while read -r key md5 url; do
  if [ -f "$key.ok" ]; then continue; fi
  wget -q -c -O "$key" "$url"
  echo "$md5  $key" | md5sum -c - && touch "$key.ok"
done < files.txt
for part in dev_audio eval_audio; do
  if [ ! -d "FSD50K.$part" ]; then
    zip -q -s 0 "FSD50K.$part.zip" --out "unsplit_$part.zip"
    unzip -q "unsplit_$part.zip" && rm -f "unsplit_$part.zip" FSD50K.$part.z0* "FSD50K.$part.zip"
  fi
done
for z in FSD50K.metadata.zip FSD50K.ground_truth.zip FSD50K.doc.zip; do [ -f "$z" ] && unzip -q -o "$z" && rm -f "$z"; done
cd "$D/esc50"
[ -d ESC-50-master ] || { wget -q -c -O master.zip https://github.com/karolpiczak/ESC-50/archive/master.zip && unzip -q master.zip && rm -f master.zip; }
cd "$D/musan"
[ -d musan ] || { wget -q -c https://www.openslr.org/resources/17/musan.tar.gz && tar xzf musan.tar.gz && rm -f musan.tar.gz; }
cd "$D/librispeech"
[ -d LibriSpeech ] || { wget -q -c https://www.openslr.org/resources/12/train-clean-100.tar.gz && tar xzf train-clean-100.tar.gz && rm -f train-clean-100.tar.gz; }
du -sh "$D"/*
echo FETCH_DONE
