from common import *
# rep2 / rep0: correlation between segment means of adjacent segments under exponential ACF (and empirical ACF of full trajectory)
from statsmodels.tsa.stattools import acf
def ratio(acfun, N1, N2, gap):
    i = np.arange(N1)[:, None]; j = np.arange(N2)[None, :]
    lag = (N1 - 1 - i) + gap + 1 + j
    cov = acfun(lag).mean()
    v1 = acfun(np.abs(np.arange(N1)[:, None]-np.arange(N1)[None, :])).mean()
    v2 = acfun(np.abs(np.arange(N2)[:, None]-np.arange(N2)[None, :])).mean()
    return cov/np.sqrt(v1*v2)
for rep, N, gap in (("rep2", 5209, 81), ("rep0", 149, 40)):
    for tau_ns in (0.35, 1.4, 2.8, 10):
        t = tau_ns/0.05
        rho = ratio(lambda k: np.exp(-k/t), N, N, gap)
        print(f"{rep} gap={gap} fr, exp ACF tau={tau_ns} ns: corr(mean1,mean2)={rho:.4f} -> SE(diff) inflation under indep assumption x{1/np.sqrt(1-rho):.3f}")
    # empirical: acf of full trajectory n (heavily nonstationary; upper bound flavour)
    x = S[rep, "full_n"]; r = acf(x, nlags=N+gap+N+2, fft=True)
    rho = ratio(lambda k: r[k], N, N, gap); print(f"{rep} empirical full-traj ACF (len {len(x)}): corr(means)={rho:.3f}; r at gap={r[gap]:.3f}")
