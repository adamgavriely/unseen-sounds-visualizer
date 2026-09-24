"""Does OUR render move the audio or the video relative to the source? Compare each stream of the
composite with the same stream of the original: audio waveform to audio waveform, brightness to
brightness. Both lags zero = the composite is in sync with whatever the source was."""
import glob, subprocess
import numpy as np
FPS = 25.0

def bright(path, crop=None):
    vf = f"{crop+',' if crop else ''}fps={FPS},scale=32:32"
    v = subprocess.run(["ffmpeg","-v","error","-i",path,"-vf",vf,"-f","rawvideo","-pix_fmt","gray","-"],
                       capture_output=True).stdout
    n=len(v)//1024
    return np.frombuffer(v,dtype=np.uint8)[:n*1024].reshape(n,1024).mean(axis=1).astype(np.float64)

def env(path):
    a = subprocess.run(["ffmpeg","-v","error","-i",path,"-ac","1","-ar","16000","-f","f32le","-"],
                       capture_output=True).stdout
    y=np.frombuffer(a,dtype=np.float32).astype(np.float64); k=int(16000/FPS); m=len(y)//k
    return np.abs(y[:m*k].reshape(m,k)).max(axis=1)

def lag(a,b,maxs=3.0):
    n=min(len(a),len(b)); a,b=a[:n],b[:n]
    a=(a-a.mean())/(a.std()+1e-9); b=(b-b.mean())/(b.std()+1e-9)
    best,bl=-9,0
    for s in range(-int(maxs*FPS), int(maxs*FPS)+1):
        x,y=(a[:n-s],b[s:]) if s>=0 else (a[-s:],b[:n+s])
        if len(x)<20: continue
        c=float((x*y).mean())
        if c>best: best,bl=c,s
    return bl/FPS, best

for comp in sorted(glob.glob("data/work/error_site/clips/*.mp4"))[:8]:
    import os; stem=os.path.basename(comp)[:-4]
    g=glob.glob(f"data/input/benchmark/*/{stem}.*")+glob.glob(f"data/input/gold139/all/{stem}.*")
    if not g: continue
    o=g[0]
    la,ra = lag(env(o), env(comp))
    lv,rv = lag(bright(o), bright(comp, crop="crop=in_h*16/9:in_h:0:0"))
    print(f"{stem[:32]:32s} audio shift {la:+.2f}s (r {ra:.2f})   video shift {lv:+.2f}s (r {rv:.2f})")
