from common import *
from arch.bootstrap import CircularBlockBootstrap, MovingBlockBootstrap, StationaryBootstrap
from statsmodels.tsa.stattools import acf
dt = 0.05
for v in ("cn_gly_wat", "n_gly_wat"):
    for k in ("s", "p"):
        x = S["rep2", k, v]; r = acf(x, nlags=len(x)//4, fft=True)
        lag = np.flatnonzero(r < np.exp(-1))[0]
        # flatness: how many lags with r within ±0.03 of 1/e; and where r<1/e-0.05 first
        near = np.sum(np.abs(r[:3*lag+20] - np.exp(-1)) < 0.03)
        print(v, k, "tau", lag*dt, "r[lag-1..lag+1]", r[lag-1:lag+2].round(3), "lags within ±0.03 of 1/e:", near,
              "first r<0.32:", np.flatnonzero(r < 0.32)[0]*dt, "first r<0.42:", np.flatnonzero(r < 0.42)[0]*dt,
              "r at 1,5,10,20,40,56,80 lags:", r[[1,5,10,20,40,56,80]].round(3))
R = 500
def boot(cls, x, L, seed):
    out = []
    for d in cls(L, x, seed=np.random.default_rng(seed)).bootstrap(R):
        out.append(correlation_time(d[0][0], dt)[2])
    return np.array([np.nan if t is None else t for t in out], float)
for v in ("cn_gly_wat", "n_gly_wat"):
    for L in (182, 364, 728, 1302):
        for name, cls in (("CBB", CircularBlockBootstrap), ("MBB", MovingBlockBootstrap)):
            ts = boot(cls, S["rep2", "s", v], L, 1); tp = boot(cls, S["rep2", "p", v], L, 2)
            d = tp - ts
            print(f"{v} L={L} {name}: site obs {correlation_time(S['rep2','s',v],dt)[2]:.2f} boot mean {np.nanmean(ts):.2f} med {np.nanmedian(ts):.2f} q[2.5,97.5]={np.nanpercentile(ts,[2.5,97.5]).round(2)} | "
                  f"plat obs {correlation_time(S['rep2','p',v],dt)[2]:.2f} mean {np.nanmean(tp):.2f} | diff mean {np.nanmean(d):.2f} CI {np.nanpercentile(d,[2.5,97.5]).round(2)}")
