"""
Train the Explainable Crop Recommendation System.

This script implements the full offline pipeline described in Chapter 3,
replicating the methodology of Ahmed et al. (2025).

Pipeline (the eleven-phase methodology, condensed to code):
  1.  Load and clean the Kaggle dataset
  2.  Encode the 22 crop labels (LabelEncoder)
  3.  80/20 stratified split (random_state=42)
  4.  Standardize features (StandardScaler, fit on train only)
  5.  Hybrid feature fusion (PCA + MI + RFE -> 22 fused features)
  6.  Train 6 baseline classifiers
  7.  Train 3 ensemble models (First Stacking, Second Stacking, Voting)
  8.  Evaluate all 9 models (accuracy, precision, recall, F1, 5-fold CV)
  9.  Build SHAP TreeExplainer on a dedicated Random Forest proxy
  10. Persist every artefact to models/

Run:
    python train_model.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils.preprocessing import BASE_FEATURES, HybridFeatureFusion, dataframe_to_raw

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "Crop_recommendation.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 5


# ------------------------------------------------------------------ #
# Model builders — every model matches the Chapter 3 specification.
# ------------------------------------------------------------------ #
def build_baselines() -> dict:
    """The 6 baseline classifiers."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="minkowski"),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", tree_method="hist",
            random_state=RANDOM_STATE, verbosity=0,
        ),
    }


def build_first_stacking() -> StackingClassifier:
    """First Stacking: KNN + Random Forest + Naive Bayes -> Logistic Regression."""
    return StackingClassifier(
        estimators=[
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("rf", RandomForestClassifier(
                n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)),
            ("nb", GaussianNB()),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )


def build_second_stacking() -> StackingClassifier:
    """Second Stacking: KNN + Bagging + Naive Bayes -> Logistic Regression."""
    return StackingClassifier(
        estimators=[
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("bagging", BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
                n_estimators=10, n_jobs=-1, random_state=RANDOM_STATE)),
            ("nb", GaussianNB()),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
    )


def build_voting() -> VotingClassifier:
    """Voting (soft): LR + XGBoost + RF + DT + NB. KNN deliberately excluded."""
    return VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ("xgb", XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                eval_metric="mlogloss", tree_method="hist",
                random_state=RANDOM_STATE, verbosity=0)),
            ("rf", RandomForestClassifier(
                n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)),
            ("dt", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ("nb", GaussianNB()),
        ],
        voting="soft",
        n_jobs=-1,
    )


def evaluate(model, X_te, y_te) -> dict:
    """Compute the four macro-averaged metrics on the held-out test set."""
    y_pred = model.predict(X_te)
    return {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
    }


def main() -> None:
    t0 = time.time()

    # 1. Load and clean
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH).drop_duplicates().dropna().reset_index(drop=True)
    print(f"[data ] {len(df)} rows, {df['label'].nunique()} crop classes")

    # 2. Encode target
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])

    # 3. Split BEFORE fitting anything (prevent data leakage)
    X_raw = dataframe_to_raw(df)
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[split] train={len(X_tr_raw)}  test={len(X_te_raw)}")

    # 4. Scale (fit on train only)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_raw)
    X_te_s = scaler.transform(X_te_raw)

    # 5. Hybrid feature fusion (fit on train only)
    print("[fusion] fitting PCA + MI + RFE ...")
    fusion = HybridFeatureFusion()
    X_tr_fused = fusion.fit_transform(X_tr_s, y_tr)
    X_te_fused = fusion.transform(X_te_s)
    print(f"[fusion] {len(BASE_FEATURES)} base -> {X_tr_fused.shape[1]} fused features")
    print(f"[fusion] PCA explained variance = {fusion.pca_explained_variance_:.3f}")
    print(f"[fusion] MI selected  : {[BASE_FEATURES[i] for i in fusion.mi_indices]}")
    print(f"[fusion] RFE selected : {[BASE_FEATURES[i] for i in fusion.rfe_indices]}")

    # 6 + 7. Train all 9 models
    all_results = {}
    trained_models = {}

    print("[train] 6 baseline classifiers ...")
    for name, model in build_baselines().items():
        model.fit(X_tr_fused, y_tr)
        all_results[name] = evaluate(model, X_te_fused, y_te)
        trained_models[name] = model
        print(f"        {name:20s} acc={all_results[name]['accuracy']:.4f}")

    print("[train] 3 ensemble models ...")
    ensembles = {
        "First Stacking": build_first_stacking(),
        "Second Stacking": build_second_stacking(),
        "Voting": build_voting(),
    }
    for name, model in ensembles.items():
        model.fit(X_tr_fused, y_tr)
        all_results[name] = evaluate(model, X_te_fused, y_te)
        trained_models[name] = model
        print(f"        {name:20s} acc={all_results[name]['accuracy']:.4f}")

    # 8. 5-fold cross validation for every model
    print(f"[cv   ] running {CV_FOLDS}-fold cross validation ...")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_builders = {
        "Logistic Regression": lambda: build_baselines()["Logistic Regression"],
        "Decision Tree": lambda: build_baselines()["Decision Tree"],
        "Random Forest": lambda: build_baselines()["Random Forest"],
        "KNN": lambda: build_baselines()["KNN"],
        "Naive Bayes": lambda: build_baselines()["Naive Bayes"],
        "XGBoost": lambda: build_baselines()["XGBoost"],
        "First Stacking": build_first_stacking,
        "Second Stacking": build_second_stacking,
        "Voting": build_voting,
    }
    for name, builder in cv_builders.items():
        scores = cross_val_score(
            builder(), X_tr_fused, y_tr, cv=skf, scoring="accuracy", n_jobs=-1
        )
        all_results[name]["cv_mean"] = float(scores.mean())
        all_results[name]["cv_std"] = float(scores.std())
        print(f"        {name:20s} cv={scores.mean():.4f} +/- {scores.std():.4f}")

    # Champion = the model with the highest test-set accuracy, breaking ties
    # with the higher 5-fold CV mean (more stable across folds).
    champion_name = max(
        all_results,
        key=lambda name: (all_results[name]["accuracy"], all_results[name]["cv_mean"]),
    )
    production_model = trained_models[champion_name]
    print(f"[champ] winner = {champion_name} "
          f"(acc={all_results[champion_name]['accuracy']:.4f}, "
          f"cv={all_results[champion_name]['cv_mean']:.4f})")

    # Confusion matrix for the champion
    y_pred_champ = production_model.predict(X_te_fused)
    cm = confusion_matrix(y_te, y_pred_champ).tolist()
    class_report = classification_report(
        y_te, y_pred_champ, target_names=encoder.classes_,
        output_dict=True, zero_division=0,
    )

    # 9. SHAP proxy — dedicated Random Forest on the same 22 fused features
    print("[shap ] training Random Forest proxy for SHAP ...")
    import shap
    shap_rf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
    )
    shap_rf.fit(X_tr_fused, y_tr)
    shap_explainer = shap.TreeExplainer(shap_rf)

    # 10. Persist everything
    joblib.dump(production_model, MODELS_DIR / "production_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(encoder, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(fusion, MODELS_DIR / "fusion_pipeline.joblib")
    joblib.dump(shap_rf, MODELS_DIR / "shap_rf_model.joblib")
    joblib.dump(shap_explainer, MODELS_DIR / "shap_explainer.joblib")

    metrics = {
        "models": all_results,
        "champion": champion_name,
        "confusion_matrix": cm,
        "classification_report": class_report,
        "class_names": encoder.classes_.tolist(),
        "n_train": int(len(X_tr_raw)),
        "n_test": int(len(X_te_raw)),
        "n_base_features": len(BASE_FEATURES),
        "n_fused_features": int(X_tr_fused.shape[1]),
        "fused_feature_names": fusion.fused_feature_names,
        "pca_explained_variance": fusion.pca_explained_variance_,
        "mi_selected": [BASE_FEATURES[i] for i in fusion.mi_indices],
        "rfe_selected": [BASE_FEATURES[i] for i in fusion.rfe_indices],
        "training_seconds": float(time.time() - t0),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump({"fused_feature_names": fusion.fused_feature_names}, f, indent=2)

    print(f"[done ] artefacts saved to {MODELS_DIR}")
    print(f"[done ] champion ({champion_name}) accuracy = "
          f"{all_results[champion_name]['accuracy']:.4f}")
    print(f"[done ] total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
