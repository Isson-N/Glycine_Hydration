from common import *
from arch.bootstrap import CircularBlockBootstrap, optimal_block_length
import arch; print("arch", arch.__version__)
print(S["bounds"])
# 1. joint indices
a = np.arange(1000.); b = np.arange(1000.) + 10000
rng = np.random.default_rng(1)
bs = CircularBlockBootstrap(50, a, b, seed=rng)
bp = CircularBlockBootstrap(50, a, b, seed=rng)
same = []; ident = []
for (s, kw), (p, _) in zip(bs.bootstrap(200), bp.bootstrap(200)):
    same.append(np.array_equal(s[0] + 10000, s[1])); ident.append(np.array_equal(s[0], p[0]))
print("joint same idx:", all(same), "| site==plateau draw in any rep:", any(ident), "| kw:", kw)
print("bs generator is rng:", bs.generator is rng)
# reproducibility: second run with fresh rng(1) gives same?
r1 = [x[0][0][:5] for x in CircularBlockBootstrap(50, a, seed=np.random.default_rng(1)).bootstrap(2)]
r2 = [x[0][0][:5] for x in CircularBlockBootstrap(50, a, seed=np.random.default_rng(1)).bootstrap(2)]
print("reproducible:", all(np.array_equal(x, y) for x, y in zip(r1, r2)))
# block structure: circular wrap present?
s0 = next(CircularBlockBootstrap(50, a, seed=np.random.default_rng(3)).bootstrap(1))[0][0]
d = np.diff(s0); print("jumps:", np.sum(d != 1), "wraps (999->0):", np.sum((s0[:-1] == 999) & (s0[1:] == 0)))
# 2. block length
for rep in ("rep2", "rep0"):
    for k in ("s", "p"):
        for v in ("n_gly_wat", "cn_gly_wat"):
            x = S[rep, k, v]; o = optimal_block_length(x)
            print(rep, k, v, len(x), o.round(1).to_dict("records"))
