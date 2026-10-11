"""
Hyperparameter tuning experiment for the Explainable Crop Recommendation System.

This is a CHECK on the fixed, benchmark-derived settings used in train_model.py,
not a replacement for them. It answers one question: would tuning each model's
hyperparameters have meaningfully changed the results?

Protocol (fixed before running):
  1. Same 80/20 stratified split (random_state=42) as train_model.py, for every model.
  2. The test set is NEVER used to choose settings. Every setting is chosen with
     5-fold stratified cross validation on the 1,760 training rows only.
  3. Scaling and hybrid feature fusion are re-fitted inside every CV fold
     (sklearn Pipeline), so the validation fold never leaks into preprocessing.
  4. Grid search tests parameter combinations together, not one at a time.
  5. Grids are centred on the benchmark settings used in train_model.py.
  6. Adoption rule: a tuned setting is adopted only if its CV mean accuracy beats
     the benchmark setting's CV mean by more than the benchmark's CV standard
     deviation. Otherwise the benchmark setting is kept.
  7. The test set is used ONCE, at the very end, to report default vs tuned
     models side by side.

Outputs go to results/tuning/:
    grid_<model>.csv     every combination tried, with CV mean and std
    summary.csv          default vs best-tuned CV per model, and the decision
    test_comparison.csv  default vs tuned test-set metrics (computed once)
    tuning.json          everything above in one file

Run:
    python tune_models.py
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils.preprocessing import HybridFeatureFusion, dataframe_to_raw

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "Crop_recommendation.csv"
OUT_DIR = ROOT / "results" / "tuning"
RANDOM_STATE = 42
CV_FOLDS = 5


class FusionStep(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper so hybrid fusion can be re-fitted inside each CV fold."""

    def __init__(self, n_pca: int = 5, n_mi: int = 5, n_rfe: int = 5):
        self.n_pca = n_pca
        self.n_mi = n_mi
        self.n_rfe = n_rfe

    def fit(self, X, y):
        self.fusion_ = HybridFeatureFusion(self.n_pca, self.n_mi, self.n_rfe).fit(X, y)
        return self

    def transform(self, X):
        return self.fusion_.transform(X)


# ------------------------------------------------------------------ #
# Benchmark settings (identical to train_model.py) and search grids.
# n_jobs=1 inside models; parallelism happens across CV folds/combos.
# ------------------------------------------------------------------ #
def xgb(**kw):
    base = dict(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="mlogloss",
                tree_method="hist", random_state=RANDOM_STATE, verbosity=0, n_jobs=1)
    base.update(kw)
    return XGBClassifier(**base)


BASE_MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE),
        {"C": [0.01, 0.1, 1, 10, 100]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE),
        {"criterion": ["gini", "entropy"], "max_depth": [None, 5, 10, 20],
         "min_samples_split": [2, 5, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=RANDOM_STATE),
        {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20],
         "min_samples_split": [2, 5]},
    ),
    "KNN": (
        KNeighborsClassifier(n_neighbors=5, metric="minkowski"),
        {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"],
         "metric": ["minkowski", "manhattan"]},  # minkowski (p=2) is Euclidean distance
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]},
    ),
    "XGBoost": (
        xgb(),
        {"n_estimators": [100, 200, 300], "max_depth": [3, 6, 9],
         "learning_rate": [0.05, 0.1, 0.2]},
    ),
    # Bagging is not a stand-alone baseline; it is tuned because it is a base
    # learner inside the Second Stacking ensemble.
    "Bagging": (
        BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                          n_estimators=10, n_jobs=1, random_state=RANDOM_STATE),
        {"n_estimators": [10, 50, 100]},
    ),
}
BASELINES = ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "Naive Bayes", "XGBoost"]


def build_ensembles(m: dict) -> dict:
    """The three ensembles from train_model.py, built from the given base learners."""
    def c(name):
        return clone(m[name])

    return {
        "First Stacking": StackingClassifier(
            estimators=[("knn", c("KNN")), ("rf", c("Random Forest")), ("nb", c("Naive Bayes"))],
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            cv=5, stack_method="predict_proba", n_jobs=1),
        "Second Stacking": StackingClassifier(
            estimators=[("knn", c("KNN")), ("bagging", c("Bagging")), ("nb", c("Naive Bayes"))],
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            cv=5, stack_method="predict_proba", n_jobs=1),
        "Voting": VotingClassifier(
            estimators=[("lr", c("Logistic Regression")), ("xgb", c("XGBoost")),
                        ("rf", c("Random Forest")), ("dt", c("Decision Tree")),
                        ("nb", c("Naive Bayes"))],
            voting="soft", n_jobs=1),
    }


def pipe(model, memory=None) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("fusion", FusionStep()), ("model", model)],
                    memory=memory)


def test_metrics(model, X_tr, y_tr, X_te, y_te) -> dict:
    p = pipe(model).fit(X_tr, y_tr)
    pred = p.predict(X_te)
    return {
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_te, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        "correct": int((pred == y_te).sum()),
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH).drop_duplicates().dropna().reset_index(drop=True)
    y = LabelEncoder().fit_transform(df["label"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        dataframe_to_raw(df), y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"[split] train={len(X_tr)} test={len(X_te)}  (test set untouched until the final step)")

    cache = tempfile.mkdtemp(prefix="fusion_cache_")
    summary, report = [], {"grids": {}}
    try:
        # ---------------- Stage 1: grid search on the training set only ---------------- #
        for name, (model, grid) in BASE_MODELS.items():
            t = time.time()
            gs = GridSearchCV(pipe(model, memory=cache), {f"model__{k}": v for k, v in grid.items()},
                              cv=skf, scoring="accuracy", n_jobs=-1, refit=False)
            gs.fit(X_tr, y_tr)
            res = pd.DataFrame(gs.cv_results_)
            params = res["params"].apply(lambda d: {k.replace("model__", ""): v for k, v in d.items()})
            table = pd.DataFrame(list(params))
            table["cv_mean"] = res["mean_test_score"]
            table["cv_std"] = res["std_test_score"]
            table = table.sort_values("cv_mean", ascending=False).reset_index(drop=True)
            table.to_csv(OUT_DIR / f"grid_{name.lower().replace(' ', '_')}.csv", index=False)

            default_params = {k: model.get_params()[k] for k in grid}
            is_default = params.apply(lambda d: d == default_params)
            d_mean = float(res.loc[is_default, "mean_test_score"].iloc[0])
            d_std = float(res.loc[is_default, "std_test_score"].iloc[0])
            best = params.iloc[int(np.argmax(res["mean_test_score"]))]
            b_mean = float(res["mean_test_score"].max())
            b_std = float(res.loc[res["mean_test_score"].idxmax(), "std_test_score"])
            adopt = (b_mean - d_mean) > d_std and best != default_params

            summary.append({
                "model": name, "combinations": len(table),
                "default_params": default_params, "default_cv_mean": d_mean, "default_cv_std": d_std,
                "best_params": best, "best_cv_mean": b_mean, "best_cv_std": b_std,
                "gain_pct_points": (b_mean - d_mean) * 100, "adopted": bool(adopt),
            })
            report["grids"][name] = table.to_dict(orient="records")
            print(f"[tune ] {name:20s} {len(table):3d} combos  default={d_mean:.4f}±{d_std:.4f}  "
                  f"best={b_mean:.4f} {best}  adopt={adopt}  ({time.time() - t:.0f}s)")

        defaults = {name: clone(model) for name, (model, _) in BASE_MODELS.items()}
        tuned_bases = {}
        for s in summary:
            m = clone(defaults[s["model"]])
            tuned_bases[s["model"]] = m.set_params(**s["best_params"]) if s["adopted"] else m

        # ---------------- Stage 2: ensembles rebuilt from tuned base learners ---------------- #
        # Ensembles are not grid-searched (too costly and they inherit their members'
        # settings); they are rebuilt with the tuned members and compared by CV.
        ens_default, ens_tuned = build_ensembles(defaults), build_ensembles(tuned_bases)
        any_member_tuned = {
            "First Stacking": any(s["adopted"] for s in summary if s["model"] in ("KNN", "Random Forest", "Naive Bayes")),
            "Second Stacking": any(s["adopted"] for s in summary if s["model"] in ("KNN", "Bagging", "Naive Bayes")),
            "Voting": any(s["adopted"] for s in summary if s["model"] in
                          ("Logistic Regression", "XGBoost", "Random Forest", "Decision Tree", "Naive Bayes")),
        }
        for name in ens_default:
            t = time.time()
            d = cross_val_score(pipe(ens_default[name], memory=cache), X_tr, y_tr, cv=skf, n_jobs=-1)
            if any_member_tuned[name]:
                tu = cross_val_score(pipe(ens_tuned[name], memory=cache), X_tr, y_tr, cv=skf, n_jobs=-1)
            else:
                tu = d
            summary.append({
                "model": name, "combinations": 1 + int(any_member_tuned[name]),
                "default_params": "benchmark members", "default_cv_mean": float(d.mean()),
                "default_cv_std": float(d.std()),
                "best_params": "tuned members" if any_member_tuned[name] else "no member changed",
                "best_cv_mean": float(tu.mean()), "best_cv_std": float(tu.std()),
                "gain_pct_points": float(tu.mean() - d.mean()) * 100,
                "adopted": bool(any_member_tuned[name] and (tu.mean() - d.mean()) > d.std()),
            })
            print(f"[ens  ] {name:20s} default={d.mean():.4f}±{d.std():.4f}  "
                  f"tuned-members={tu.mean():.4f}±{tu.std():.4f}  ({time.time() - t:.0f}s)")

        # ---------------- Stage 3: test set, used once ---------------- #
        print("[test ] evaluating default and tuned models on the held-out test set (once)")
        rows = []
        all_default = {**{k: defaults[k] for k in BASELINES}, **ens_default}
        all_tuned = {**{k: tuned_bases[k] for k in BASELINES}, **ens_tuned}
        for name in all_default:
            dm = test_metrics(all_default[name], X_tr, y_tr, X_te, y_te)
            changed = next(s for s in summary if s["model"] == name)["adopted"] or (
                name in any_member_tuned and any_member_tuned[name])
            tm = test_metrics(all_tuned[name], X_tr, y_tr, X_te, y_te) if changed else dm
            rows.append({"model": name,
                         **{f"default_{k}": v for k, v in dm.items()},
                         **{f"tuned_{k}": v for k, v in tm.items()},
                         "settings_changed": bool(changed)})
            print(f"        {name:20s} default={dm['accuracy']:.4f} ({dm['correct']}/440)  "
                  f"tuned={tm['accuracy']:.4f} ({tm['correct']}/440)")
    finally:
        shutil.rmtree(cache, ignore_errors=True)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT_DIR / "test_comparison.csv", index=False)
    report.update({"summary": summary, "test_comparison": rows, "cv_folds": CV_FOLDS,
                   "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
                   "seconds": float(time.time() - t0)})
    with open(OUT_DIR / "tuning.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[done ] results in {OUT_DIR}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
