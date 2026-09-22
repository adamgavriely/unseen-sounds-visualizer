import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,r"P:/MscProj")
import config; config.use_v4("59")
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import GOLD
from benchmark.gold.veto_sweep import keep, fx
from src.labels import canonical
gold=S.load_gold([GOLD]); subs=S.subsets_of(gold)
PA=Path("benchmark/gold/panns_fw"); BE=Path("benchmark/gold/beats_fw")
def peaks(dirp, stem):
    p=dirp/(stem+".npz")
    if not p.exists(): return None
    try: z=np.load(p,allow_pickle=False)
    except Exception: return None
    d=defaultdict(float); fw=z["fw"]
    mx=fw.max(axis=0)
    for j,l in enumerate(z["labels"]):
        c=canonical(str(l))
        if mx[j]>d[c]: d[c]=float(mx[j])
    return d
PK={}
def pk(dirp,stem,lab):
    k=(str(dirp),stem)
    if k not in PK: PK[k]=peaks(dirp,stem)
    d=PK[k]
    return None if d is None else d.get(canonical(lab))
def run(stems, root, tau2):
    out=[]
    for st in stems:
        pics=S.load_pictures(root,st,"proposed")
        if pics is None: continue
        pics=[p for p in pics if keep(st,p[0],0.3,None)]
        if tau2 is not None:
            kept=[]
            for lab,x,y in pics:
                f=fx(st,lab); fxpk=f[1] if f else 0.0
                bpk=pk(BE,st,lab); ppk=pk(PA,st,lab)
                # FlexSED raised it and BEATs did not -> require PANNs support
                if fxpk>=0.8 and (bpk is None or bpk<0.35) and ppk is not None and ppk<tau2:
                    continue
                kept.append((lab,x,y))
            pics=kept
        out.append(S.score_clip(gold[st],pics))
    return out
stems=sorted(set(subs["dev"])&set(subs["bench"]))
root=Path("data/work/protocol_proposed_v4b6")
print("== DEV (%d clips), selection only"%len(stems))
print("%-30s %6s %6s %6s %8s %6s %5s %5s"%("cell","F1","P","R","FA/clip","cost","hits","miss"))
for tau2,name in ((None,"veto only (current best)"),(0.02,"+ PANNs veto tau2 0.02"),(0.05,"+ PANNs veto tau2 0.05"),(0.10,"+ PANNs veto tau2 0.10")):
    r=run(stems,root,tau2); a=S.aggregate(r)
    print("%-30s %6.3f %6.3f %6.3f %8.2f %6.2f %5d %5d"%(name,a["F1"],a["P"],a["R"],S.fa_per_clip(r),S.viewer_cost(r),sum(x["hit"] for x in r),sum(x["miss"] for x in r)))
print("%-30s %6s %6s %6s %8.2f %6.2f"%("silence","-","-","-",0.0,S.viewer_cost([S.score_clip(gold[s],[]) for s in stems])))
