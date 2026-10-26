from common import *
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "second_member"))
import bootstrap as B
def tvd_fast(a, b, lo, K):
    pa = np.bincount((a - lo).astype(int), minlength=K) / len(a); pb = np.bincount((b - lo).astype(int), minlength=K) / len(b)
    return 0.5 * np.abs(pa - pb).sum()
def perm(x, y, L, rng, R=500):
    lo = min(x.min(), y.min()); K = int(max(x.max(), y.max()) - lo + 1)
    blocks = [z[i:i+L] for z in (x, y) for i in range(0, len(z)-L+1, L)]; ns = len(x)//L
    obs = tvd_fast(x, y, lo, K); null = np.empty(R)
    for r in range(R):
        o = rng.permutation(len(blocks))
        null[r] = tvd_fast(np.concatenate([blocks[i] for i in o[:ns]]), np.concatenate([blocks[i] for i in o[ns:]]), lo, K)
    return obs, (np.sum(null >= obs) + 1) / (R + 1), null
rng = np.random.default_rng(5)
# agreement with project implementation
x, y = S["rep2","s","n_gly_wat"], S["rep2","p","n_gly_wat"]
o, p, null = perm(x, y, 182, rng, 2000)
print("rep2 obs TVD", round(o,3), "p", p, "null q50/q99/max", np.quantile(null,[.5,.99]).round(3), null.max().round(3))
print("dropped frames per segment at L=182:", len(x) - (len(x)//182)*182, "| obs TVD on truncated:", round(tvd_fast(x[:28*182], y[:28*182], min(x.min(),y.min()), 20),3))
x0, y0 = S["rep0","s","n_gly_wat"], S["rep0","p","n_gly_wat"]
for L in (10, 19):
    o, p, null = perm(x0, y0, L, rng, 2000); print(f"rep0 L={L}: obs {o:.3f} p {p:.3f}, blocks/seg {149//L}, dropped {149-(149//L)*L}")
# Type-I calibration: surrogate with site spectrum & marginal, two adjacent segments of 5209 from one realisation
z = S["rep2","s","n_gly_wat"]; N = len(z); M = 2*N
spec = np.abs(np.fft.rfft(np.concatenate([z - z.mean(), (z - z.mean())[::-1]])))  # length-M even-extension spectrum
srt = np.sort(z)
for L in (182, 364, 91):
    rej = 0; S_ = 200
    for s in range(S_):
        ph = rng.uniform(0, 2*np.pi, len(spec)); g = np.fft.irfft(spec*np.exp(1j*ph), n=M)
        ranks = np.argsort(np.argsort(g)); surr = np.sort(rng.choice(srt, M))[ranks]
        _, p, _ = perm(surr[:N], surr[N:], L, rng, 300); rej += p < 0.05
    print(f"surrogate H0 (site-like spectrum), L={L}: rejection rate {rej/S_:.3f} (nominal 0.05)")
