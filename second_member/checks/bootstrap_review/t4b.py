exec(open("t4_perm.py").read().split("rng = np.random")[0])
rng = np.random.default_rng(7)
def surrogates(z, M, k):
    zc = z - z.mean(); spec = np.abs(np.fft.rfft(np.concatenate([zc, zc[::-1]])))[: M//2+1] if M == 2*len(z) else None
    srt = np.sort(z)
    for _ in range(k):
        g = np.fft.irfft(spec*np.exp(1j*rng.uniform(0, 2*np.pi, len(spec))), n=M)
        yield np.sort(rng.choice(srt, M))[np.argsort(np.argsort(g))]
for nm in ("s", "p"):
    z = S["rep2", nm, "n_gly_wat"]; N = len(z); tv = []
    for surr in surrogates(z, 2*N, 300):
        a, b = surr[:N], surr[N:]; lo = min(a.min(), b.min()); tv.append(tvd_fast(a, b, lo, int(max(a.max(), b.max())-lo+1)))
    tv = np.array(tv); print(f"{nm}-spectrum surrogate: TVD between adjacent halves q50={np.median(tv):.3f} q95={np.quantile(tv,.95):.3f} max={tv.max():.3f}; P(TVD>=0.285)={np.mean(tv>=0.285):.3f}")
# white check
rej = 0
for s in range(200):
    w = rng.choice(S["rep2","s","n_gly_wat"], 2*5209); _, p, _ = perm(w[:5209], w[5209:], 182, rng, 300); rej += p < .05
print("white H0 rejection", rej/200)
# AR-like short memory surrogate: keep spectrum only above 1/(20 ns) (smooth low end)
