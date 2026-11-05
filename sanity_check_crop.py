"""Sanity tests for the "Check my crop" suitability score.

The score has no ground truth (the dataset holds only good examples of each crop), so it cannot have an
accuracy. These tests check that it behaves sensibly instead. Ranges and the Naive Bayes reference
levels are fitted on the 1,760 training rows only; every result is measured on the 440 held-out rows
(same split as train_model.py), so the test rows never influence the score they are tested with.

Run:  python sanity_check_crop.py
Output: results/sanity_check_crop.json
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# check_crop.py needs Streamlit's cache decorators; plain functions are enough here.
import streamlit as st
st.cache_data = st.cache_resource = lambda *a, **k: (lambda f: f) if not (a and callable(a[0])) else a[0]
from pages_impl import check_crop as cc   # noqa: E402

ROOT = Path(__file__).parent
FEATURES = cc.BASE_FEATURES
SOIL, CLIMATE = cc.SOIL, cc.CLIMATE
RNG = np.random.default_rng(42)

df = pd.read_csv(ROOT / "data" / "Crop_recommendation.csv").drop_duplicates().dropna().reset_index(drop=True)
train, test = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=42)
crops = sorted(df["label"].unique())

# ---- fit the score on the training rows only ---------------------------------------------------
q = train.groupby("label")[FEATURES].quantile([cc.LOW_Q, cc.HIGH_Q]).unstack()
nb = GaussianNB().fit(train[FEATURES].values, train["label"].values)
model = {"nb": nb, "classes": list(nb.classes_), "ref": {}}
for crop in crops:
    ll = cc._log_likelihood(model, crop, train.loc[train["label"] == crop, FEATURES].values)
    edge = float(np.percentile(ll, 5))
    model["ref"][crop] = (edge, max(float(np.median(ll)) - edge, 1e-6))


def bounds(crop):
    out = {}
    for k in FEATURES:
        d = 1 if k in ("ph", "temperature", "humidity") else 0
        out[k] = (round(float(q.loc[crop][(k, cc.LOW_Q)]), d), round(float(q.loc[crop][(k, cc.HIGH_Q)]), d))
    return out


BOUNDS = {c: bounds(c) for c in crops}


def typ(crop, vals):
    edge, scale = model["ref"][crop]
    ll = float(cc._log_likelihood(model, crop, [vals[k] for k in FEATURES]))
    return 1.0 if ll >= edge else float(np.exp(-(edge - ll) / scale))


def score(crop, vals):
    """Same arithmetic as pages_impl/check_crop._check, without the page and the champion model."""
    rows, fixed = [], dict(vals)
    for k in FEATURES:
        lo, hi = BOUNDS[crop][k]
        v = vals[k]
        status = "low" if v < lo else "high" if v > hi else "ok"
        rows.append((k, cc._value_score(v, lo, hi), status))
        if k in SOIL:
            fixed[k] = min(max(v, lo), hi)
    range_now = np.mean([s for _, s, _ in rows])
    range_after = np.mean([1.0 if k in SOIL else s for k, s, _ in rows])
    now = (range_now + typ(crop, vals)) / 2
    after = (range_after + typ(crop, fixed)) / 2
    n_soil = sum(1 for k, _, st_ in rows if k in SOIL and st_ != "ok")
    n_clim = sum(1 for k, _, st_ in rows if k in CLIMATE and st_ != "ok")
    risk = "high" if n_clim >= 2 else "medium" if (n_clim == 1 or n_soil) else "low"
    return {"now": now, "after": after, "risk": risk, "n_out": n_soil + n_clim}


test_rows = [(r["label"], {k: float(r[k]) for k in FEATURES}) for _, r in test.iterrows()]
res = {"n_test": len(test_rows), "n_crops": len(crops)}

# ---- Test 1: the right crop on real held-out land --------------------------------------------
own = [score(c, v) for c, v in test_rows]
now = np.array([s["now"] for s in own])
res["t1_own_crop"] = {
    "mean_score": float(now.mean()), "median_score": float(np.median(now)),
    "share_score_ge_80": float((now >= 0.8).mean()), "share_score_ge_60": float((now >= 0.6).mean()),
    "share_risk_low": float(np.mean([s["risk"] == "low" for s in own])),
    "share_risk_high": float(np.mean([s["risk"] == "high" for s in own])),
    "min_score": float(now.min()),
}

# ---- Test 2: wrong crops should score lower than the right crop -------------------------------
top1 = top3 = 0
wins = pairs = 0
wrong_scores, wrong_low = [], 0
ranks = []
for c, v in test_rows:
    all_scores = {k: score(k, v) for k in crops}
    sc = {k: s["now"] for k, s in all_scores.items()}
    order = sorted(sc, key=sc.get, reverse=True)
    ranks.append(order.index(c) + 1)
    top1 += order[0] == c
    top3 += c in order[:3]
    for k in crops:
        if k == c:
            continue
        pairs += 1
        wins += sc[c] > sc[k]
        wrong_scores.append(sc[k])
        wrong_low += all_scores[k]["risk"] == "low"
res["t2_wrong_crops"] = {
    "mean_score_wrong_crop": float(np.mean(wrong_scores)),
    "true_beats_wrong_pairs": float(wins / pairs),
    "true_crop_ranked_1st": float(top1 / len(test_rows)),
    "true_crop_in_top3": float(top3 / len(test_rows)),
    "mean_rank_of_true_crop": float(np.mean(ranks)),
    "share_wrong_crop_risk_low": float(wrong_low / pairs),
    "share_wrong_crop_score_ge_80": float(np.mean(np.array(wrong_scores) >= 0.8)),
}

# ---- Test 3: monotonic - moving a value further from the crop's range never raises the score --
viol = checks = 0
for c, v in test_rows:
    for k in FEATURES:
        lo, hi = BOUNDS[c][k]
        width = max(hi - lo, 1e-9)
        for side in (-1, 1):
            prev = None
            for step in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5):
                w = dict(v)
                edge = lo if side < 0 else hi
                # start at the range edge itself, then move outwards in steps of the range width
                w[k] = edge + side * step * width
                s = score(c, w)["now"]
                if prev is not None:
                    checks += 1
                    viol += s > prev + 1e-9
                prev = s
res["t3_monotonic"] = {"checks": checks, "violations": int(viol)}

# ---- Test 4: fixing the soil never lowers the score -------------------------------------------
bad = tot = 0
for c, v in test_rows:
    for k in crops:
        s = score(k, v)
        tot += 1
        bad += s["after"] < s["now"] - 1e-9
res["t4_soil_fix_never_hurts"] = {"checks": tot, "violations": int(bad)}

# ---- Test 5: risk level rises with the number of values out of range --------------------------
by_out = {}
for c, v in test_rows:
    for k in crops:
        s = score(k, v)
        by_out.setdefault(min(s["n_out"], 5), []).append(s["now"])
res["t5_score_by_values_out_of_range"] = {str(n): {"n": len(a), "mean_score": float(np.mean(a))}
                                          for n, a in sorted(by_out.items())}

out = ROOT / "results" / "sanity_check_crop.json"
out.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
