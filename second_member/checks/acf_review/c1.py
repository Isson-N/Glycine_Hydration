import sys; sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parents[3] / "second_member"))
import numpy as np, pandas as pd, warnings
from statsmodels.tsa.stattools import acf
from segments import build_variants, phase_bounds
import autocorrelation as A
d=pd.read_pickle("d.pkl"); d=d[d.rep.isin(["rep0","rep2"])].sort_values(["rep","frame"]).reset_index(drop=True)
V=build_variants(d)
for k,v in V.items(): print(k,v)
# 1 manual acf
x=A.segment_values(d,"rep2",0,5200,"n_gly_wat"); N=len(x); y=x-x.mean()
K=N//4
man_b=np.array([np.dot(y[:N-k],y[k:]) for k in range(K+1)])/np.dot(y,y)
man_u=np.array([np.dot(y[:N-k],y[k:])/(N-k) for k in range(K+1)])/(np.dot(y,y)/N)
sm=acf(x,nlags=K,fft=True); sm2=acf(x,nlags=K,fft=False); smu=acf(x,nlags=K,fft=True,adjusted=True)
print("max|sm-manual_biased|",abs(sm-man_b).max(),"fft vs nofft",abs(sm-sm2).max(),"adjusted vs manual_unb",abs(smu-man_u).max())
print("first lags sm",sm[:8].round(3)); print("unbiased",man_u[:8].round(3))
fc=lambda r: int(np.flatnonzero(r<np.exp(-1))[0]) if (r<np.exp(-1)).any() else None
print("crossing biased",fc(sm),"unbiased",fc(man_u))
# 6 rep2 site halves
h=N//2
for nm,p in [("full",x),("h1",x[:h]),("h2",x[h:])]:
    r=acf(p,nlags=len(p)//4,fft=True); print(nm,len(p),r[:20].round(2),"cross",fc(r))
# 2 positions of crossing for halves: any value close to threshold?
