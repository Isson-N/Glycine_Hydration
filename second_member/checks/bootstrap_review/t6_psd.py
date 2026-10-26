from common import *
from scipy.signal import welch
from statsmodels.tsa.stattools import acf
fs = 20.
segs = {"site": S["rep2","s","n_gly_wat"], "win1": S["rep2","p","n_gly_wat"], "win2": S["rep2","p2","n_gly_wat"]}
N = 5209; nper = 2048; step = nper//2
print("Welch segments:", len(range(0, N - nper + 1, step)), "dropped tail frames:", N - (len(range(0, N-nper+1, step))-1)*step - nper)
for k, x in segs.items():
    f, p = welch(x - x.mean(), fs=fs, nperseg=2048)
    df = f[1]-f[0]
    var = x.var()
    integ = p.sum()*df
    r = acf(x, nlags=5, fft=True)
    hf = p[f > 5].mean()
    white_var = (1 - r[1]) * var   # crude: variance of component lost at lag 1
    print(f"{k}: var={var:.3f} int PSD={integ:.3f} (Welch, excl. scales>102 ns) | df={df:.4f} | r1={r[1]:.3f} | "
          f"HF floor (f>5)={hf:.4f} vs white model 2*var_w/fs={2*white_var/fs:.4f} | tau_c={correlation_time(x,0.05)[2]}")
    # flatness above some f: ratio PSD at 2 vs 9 cyc/ns
    print("   PSD at f≈0.01,0.1,0.5,1,2,5,9:", [round(p[np.argmin(abs(f-q))],4) for q in (0.01,0.1,0.5,1,2,5,9)])
