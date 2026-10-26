import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "second_member"))
import numpy as np, pandas as pd
from autocorrelation import load_working_table, segment_values, correlation_time
from segments import build_variants, SMALL, LARGE
import pickle, os
cache = os.path.join(os.path.dirname(__file__), "segs.pkl")
if os.path.exists(cache):
    S = pickle.load(open(cache, "rb"))
else:
    data = load_working_table(__import__("pathlib").Path(__file__).resolve().parents[3] / "data" / "gly_matrix_all_reps.csv", 0.05)
    V = build_variants(data)
    S = {"bounds": {k: V[k] for k in ("C", "C2")}}
    for rep in ("rep0", "rep2"):
        full = data.loc[data.rep == rep].sort_values("frame")
        S[rep, "full_n"] = full.n_gly_wat.to_numpy(float); S[rep, "full_cn"] = full.cn_gly_wat.to_numpy(float)
        S[rep, "full_frame"] = full.frame.to_numpy()
        for nm, key in ((SMALL, "s"), (LARGE, "p")):
            for v in ("n_gly_wat", "cn_gly_wat"):
                S[rep, key, v] = segment_values(data, rep, *V["C"][rep][nm], v)
    S["rep2", "p2", "n_gly_wat"] = segment_values(data, "rep2", *V["C2"]["rep2"][LARGE], "n_gly_wat")
    pickle.dump(S, open(cache, "wb"))
