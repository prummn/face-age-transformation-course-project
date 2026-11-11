"""Summaries per eval dir (+ per-input analysis). VLM = primary ethnicity metric (independent); FairFace/ID/age share models with losses (partly circular)."""
import json, glob, sys, os, numpy as np, collections, pandas as pd
def load(D):
    R = json.load(open(f"{D}/records.json")); V = {}
    for f in glob.glob(f"{D}/_vlm_*.json"): V.update(json.load(open(f)))
    pv = lambda f: V[f]["A"] + V[f]["B"] if f in V else np.nan
    for r in R: r["p_asian_vlm"] = pv(r["file"]); r["p_asian_vlm_in"] = pv(r["file"].rsplit("_", 1)[0] + "_input.png")
    return R
def summ(R):
    aged = [r for r in R if r["tag"] != "rec"]; t = np.array([r["target_age"] for r in aged]); e = np.array([r["age_est"] for r in aged])
    by = collections.defaultdict(dict)
    for r in R: by[r["input"]][r["tag"]] = r
    lost = [sum(d[k]["p_asian_vlm"] < .5 for k in ["a20", "a40", "a60", "a80"]) for d in by.values() if d["rec"]["p_asian_vlm_in"] > .5]
    out = dict(n_in=len(by), vlm_in=np.nanmean([d["rec"]["p_asian_vlm_in"] for d in by.values()]),
               **{f"vlm_{k}": np.nanmean([r["p_asian_vlm"] for r in R if r["tag"] == k]) for k in ["rec", "a20", "a40", "a60", "a80"]},
               vlm_rate_aged=np.nanmean(np.array([r["p_asian_vlm"] for r in aged]) > .5), collapse=int(sum(l >= 2 for l in lost)), n_asian_in=len(lost),
               ff_aged=np.mean([r["p_asian_ff"] for r in aged]), id_rec=np.mean([r["id_cos"] for r in R if r["tag"] == "rec"]), id_aged=np.mean([r["id_cos"] for r in aged]),
               age_corr=np.corrcoef(t, e)[0, 1], age_mae=np.abs(t - e).mean(), **{f"age_est_a{g}": np.mean([r["age_est"] for r in aged if r["target_age"] == g]) for g in [20, 40, 60, 80]})
    return {k: (round(float(v), 3) if not isinstance(v, int) else v) for k, v in out.items()}
if __name__ == "__main__":
    out = sys.argv[1]; res = {}
    for D in sys.argv[2:]:
        if os.path.exists(f"{D}/records.json"): res[os.path.basename(D)] = summ(load(D)); print(os.path.basename(D), json.dumps(res[os.path.basename(D)]), flush=True)
    json.dump(res, open(out, "w"), indent=1)
