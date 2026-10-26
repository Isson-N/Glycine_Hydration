import sys; sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parents[3] / "second_member"))
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import acf
from scipy.signal import detrend
import autocorrelation as A
d=pd.read_pickle("d.pkl")
fc=lambda r: int(np.flatnonzero(r<np.exp(-1))[0]) if (r<np.exp(-1)).any() else None
def show(nm,x,L=None):
    L=L or len(x)//4; r=acf(x,nlags=L,fft=True)
    print(f"{nm:35s} N={len(x)} r1..5={r[1:6].round(3)} r[20]={r[min(20,L)]:.3f} r[100]={r[min(100,L)]:.3f} cross={fc(r)}")
    return r
x=A.segment_values(d,"rep2",0,5200,"n_gly_wat")
show("rep2 site full",x); show("rep2 site lin-detrended",detrend(x))
r=acf(x,nlags=1300,fft=True); print("rep2 site ACF at lags 50,100,200,400,800,1300:",r[[50,100,200,400,800,1300]].round(3))
print("rep2 site: rolling mean(400 fr) range", pd.Series(x).rolling(400,center=True).mean().agg(["min","max"]).round(2).tolist(), "SD",x.std().round(2))
# rep0
for v in ["n_gly_wat","cn_gly_wat"]:
    y=A.segment_values(d,"rep0",0,8276,v)
    rf=show(f"rep0 full site {v}",y); show(f"rep0 full site detr {v}",detrend(y))
    W=144; ws=[y[i:i+W] for i in range(0,len(y)-W+1,W)]
    rs=np.array([acf(w,nlags=36,fft=True) for w in ws])
    crosses=[fc(r) for r in rs]
    print(f"  {len(ws)} windows of 144: mean r1..5={rs.mean(0)[1:6].round(3)}, cross lag distribution:",pd.Series(crosses).value_counts().sort_index().to_dict())
    w=A.segment_values(d,"rep0",100,243,v); show(f"  rep0 A window 100-243 {v}",w)
    # also global-mean-centered windows
    yc=y-y.mean(); num=np.mean([np.dot(w_[:-1]-y.mean(),w_[1:]-y.mean()) for w_ in ws])/(W); den=np.mean([np.mean((w_-y.mean())**2) for w_ in ws])
    print("  windows but centered by global mean: r1≈",round(num/den,3))
