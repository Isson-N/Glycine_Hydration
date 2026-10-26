import sys; sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parents[3] / "second_member"))
import numpy as np, pandas as pd, statsmodels, inspect
from statsmodels.tsa.stattools import adfuller, kpss
from segments import phase_bounds, MEDIAN_WINDOW
d=pd.read_pickle("d.pkl")
print("statsmodels",statsmodels.__version__, "adfuller sig:", inspect.signature(adfuller))
print("NaN:",d[["n_gly_wat","cn_gly_wat","dist_ca_gly_com_bs"]].isna().sum().to_dict())
for rep in ["rep0","rep2"]:
    s=d[d.rep==rep].set_index("frame")["dist_ca_gly_com_bs"]
    med=s.rolling(MEDIAN_WINDOW,center=True,min_periods=1).median()
    lv,ex=phase_bounds(s)
    print(rep,"leave",lv,"exit_start",ex,"len",len(s))
    print("  median>5 before leave? ",(med.loc[:lv-1]>5).sum(), " median<=5 after leave:",(med.loc[lv:]<=5).sum(), " frames >10 between leave..exit:",(med.loc[lv:ex]>10).sum())
    print("  raw dist quantiles site(0..leave-41):",s.loc[:lv-41].quantile([.5,.95,.99,1]).round(2).tolist())
    print("  raw dist stop/plateau quantiles:",s.loc[lv:ex].quantile([0,.05,.5,.95,1]).round(2).tolist())
    print("  raw dist max over site, frames with raw>5 in site:",(s.loc[:lv-41]>5).sum())
# marker index check
fs=0.05; bad=[k for k in range(1,20000) if round((k*fs)/fs)!=k]; print("marker index mismatches:",len(bad))
