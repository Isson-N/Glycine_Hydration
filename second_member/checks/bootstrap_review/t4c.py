exec(open("t4b.py").read().split("for nm in")[0])
from arch.bootstrap import CircularBlockBootstrap
for nm in ("s", "p"):
    z = S["rep2", nm, "n_gly_wat"]; N = len(z)
    m = np.array([[s[:N].mean(), s[N:].mean(), s[:N].var(ddof=1)] for s in surrogates(z, 2*N, 400)])
    bm = np.array([d[0][0].mean() for d in CircularBlockBootstrap(182, z, seed=rng).bootstrap(1000)])
    bv = np.array([d[0][0].var(ddof=1) for d in CircularBlockBootstrap(182, z, seed=rng).bootstrap(1000)])
    print(f"{nm}: SE(mean) surrogate {m[:,0].std():.3f} vs CBB L=182 {bm.std():.3f} | SD(var) surrogate {m[:,2].std():.3f} vs CBB {bv.std():.3f}")
