import pandas as pd
d=pd.read_csv(__import__("pathlib").Path(__file__).resolve().parents[3] / "data" / "gly_matrix_all_reps.csv",usecols=["rep","frame","n_gly_wat","cn_gly_wat","dist_ca_gly_com_bs"])
print(d.dtypes); print(d.groupby("rep").frame.agg(["min","max","count","nunique"]))
print("dups",d.duplicated(["rep","frame"]).sum())
print("order sorted within rep?", d.groupby("rep").frame.apply(lambda s:(s.diff().dropna()>0).all()))
d.to_pickle("d.pkl")
