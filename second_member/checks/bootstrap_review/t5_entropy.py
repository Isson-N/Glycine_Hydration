from common import *
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "second_member"))
from extras import shannon_entropy
from bootstrap import tvd
def H(x):
    p = np.unique(x, return_counts=True)[1] / len(x); return -(p*np.log2(p)).sum()
def HMM(x):
    return H(x) + (len(np.unique(x)) - 1) / (2 * len(x) * np.log(2))
# correctness spot checks
x = np.array([1,1,2,2,3,3,4,4.]); print("H uniform4 =", shannon_entropy(x), "(expect 2)")
print("tvd disjoint", tvd(np.array([1,1.]), np.array([3,3.])), "tvd same", tvd(np.array([1,2.]), np.array([2,1.])),
      "tvd check", tvd(np.array([1,1,2,2.]), np.array([1,2,2,2.])), "(expect 0.25)")
rng = np.random.default_rng(0)
for k in ("s", "p"):
    x = S["rep2", k, "n_gly_wat"]; N = len(x)
    print(f"\nrep2 {k}: N={N}, K={len(np.unique(x))}, H_full plug={H(x):.3f} MM={HMM(x):.3f}")
    print(" L   plug   MM    shuf_plug(mean of 20)  shuf_full   [bias part = Hshuf_full - shuf_plug]  [slow part = shuf_plug - plug]")
    for L in (50, 100, 200, 500, 1000, 2000):
        w = [x[i:i+L] for i in range(0, N-L+1, L)]
        hp = np.mean([H(v) for v in w]); hm = np.mean([HMM(v) for v in w])
        sh = []
        for _ in range(20):
            y = rng.permutation(x); sh.append(np.mean([H(y[i:i+L]) for i in range(0, N-L+1, L)]))
        sh = np.mean(sh)
        print(f"{L:5d} {hp:.3f} {hm:.3f} {sh:.3f} {H(x):.3f}  bias={H(x)-sh:.3f}  slow={sh-hp:.3f}  total_gap={H(x)-hp:.3f}")
# rep0 entropy difference — plug-in bias at N=149 equal sizes
for k in ("s", "p"):
    x = S["rep0", k, "n_gly_wat"]; print("rep0", k, "K", len(np.unique(x)), "plug", round(H(x),3), "MM", round(HMM(x),3), "counts", dict(zip(*np.unique(x, return_counts=True))))
