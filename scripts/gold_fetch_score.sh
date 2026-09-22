#!/bin/bash
# Fetch the small render artefacts of the gold rows from BIU and score everything locally
# (amendment 5, docs/prereg_v4.md). Usage: bash scripts/gold_fetch_score.sh [tag ...]
set -euo pipefail
cd "$(dirname "$0")/.."
TAGS="${*:-v4b4 v4ab4}"
H=adamg@slurm-login1.lnx.biu.ac.il
W=data/work
mkdir -p "$W"
for t in $TAGS; do
    ssh -o BatchMode=yes $H "cd ~/MscProj/data/work && tar czf /tmp/gold_$t.tgz \$(find protocol_*_$t -maxdepth 2 \( -name augmentations.json -o -name media.json -o -name events.json -o -name gate_votes.json \))" 2>/dev/null
    scp -q $H:/tmp/gold_$t.tgz "$W/gold_$t.tgz"
    tar xzf "$W/gold_$t.tgz" -C "$W"
    for s in proposed blind_a2i audio_caption; do echo "$t $s: $(ls $W/protocol_${s}_$t/*/augmentations.json 2>/dev/null | wc -l) clips"; done
done
# gate-accuracy vote files
ssh -o BatchMode=yes $H "cd ~/MscProj && tar czf /tmp/gate_gold.tgz benchmark/gold/gate_gold" 2>/dev/null && scp -q $H:/tmp/gate_gold.tgz "$W/gate_gold.tgz" && tar xzf "$W/gate_gold.tgz" -C . || true
A=benchmark/gold/annotations/gold_AG.json
for t in $TAGS; do
    echo "=================== $t (headline: late 1.0; sensitivity: late 3.0)"
    python benchmark/gold/score_per_sound.py --annotations $A --tag $t --systems proposed blind_a2i audio_caption silence --late 1.0 3.0 | tee benchmark/gold/per_sound_$t.txt
    echo "=================== $t old rule (level-1 needed sounds scored)"
    python benchmark/gold/score_per_sound.py --annotations $A --tag $t --systems proposed blind_a2i silence --old-rule --subsets bench --out benchmark/gold/per_sound_${t}_oldrule.json | tee benchmark/gold/per_sound_${t}_oldrule.txt
    echo "=================== $t gate re-decided (unanimous), dry"
    python benchmark/gold/gate_redecide.py --tag $t --rule unanimous | tee benchmark/gold/gate_redecide_$t.txt || true
done
echo "=================== gate accuracy on gold sounds"
python benchmark/gold/gate_gold.py --score | tee benchmark/gold/gate_gold/summary.txt || true
