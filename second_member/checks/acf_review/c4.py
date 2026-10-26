import numpy as np, pandas as pd
from statsmodels.tsa.stattools import acf
rng=np.random.default_rng(0)
fc=lambda r: int(np.flatnonzero(r<np.exp(-1))[0]) if (r<np.exp(-1)).any() else None
def ar(n,phi):
    e=rng.normal(size=n); x=np.empty(n); x[0]=e[0]/np.sqrt(1-phi**2)
    for i in range(1,n): x[i]=phi*x[i-1]+e[i]
    return x
n=200000
# two-timescale: fast white + slow AR1 with ~64% variance share -> r1~0.64 like rep0
slow=ar(n,np.exp(-1/40)); slow/=slow.std(); fast=rng.normal(size=n)
x=np.sqrt(.65)*slow+np.sqrt(.35)*fast
rt=acf(x,nlags=200,fft=True); print("synthetic true r1..3",rt[1:4].round(3),"true cross lag",fc(rt))
for W in [144,500,2600,8277]:
    ws=[x[i:i+W] for i in range(0,n-W+1,W)]
    rs=[acf(w,nlags=W//4,fft=True) for w in ws]; c=pd.Series([fc(r) for r in rs])
    print(f"W={W}: mean r1={np.mean([r[1] for r in rs]):.3f}, median cross={c.median()}, share cross==1: {(c==1).mean():.2f}")
