"""
Inference + SHAP explanation helpers used by the Streamlit app.

Loads all saved artefacts once (cached), then exposes a single
`predict_and_explain()` function that runs the full live pipeline:

    raw 7 values -> scale -> fuse (22) -> production model -> SHAP

Everything here APPLIES already-fitted objects. Nothing is re-trained.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from utils.preprocessing import BASE_FEATURES

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"


class CropRecommender:
    """Bundles the loaded artefacts and runs the live prediction pipeline."""

    def __init__(self):
        self.scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        self.encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
        self.fusion = joblib.load(MODELS_DIR / "fusion_pipeline.joblib")
        self.model = joblib.load(MODELS_DIR / "production_model.joblib")
        self.shap_explainer = joblib.load(MODELS_DIR / "shap_explainer.joblib")
        self.shap_rf = joblib.load(MODELS_DIR / "shap_rf_model.joblib")
        self.feature_names = self.fusion.fused_feature_names
        self.class_names = list(self.encoder.classes_)

    # ------------------------------------------------------------------ #
    def _fuse(self, X_raw: np.ndarray) -> np.ndarray:
        """raw (n,7) -> scaled -> fused (n,22)."""
        X_scaled = self.scaler.transform(X_raw)
        return self.fusion.transform(X_scaled)

    # ------------------------------------------------------------------ #
    def predict(self, X_raw: np.ndarray):
        """Return (predicted_labels, probability_matrix) for a batch of rows."""
        X_fused = self._fuse(X_raw)
        proba = self.model.predict_proba(X_fused)
        idx = np.argmax(proba, axis=1)
        labels = self.encoder.inverse_transform(idx)
        return labels, proba

    # ------------------------------------------------------------------ #
    def predict_and_explain(self, X_raw: np.ndarray, top_k: int = 5, top_feats: int = 10):
        """
        Full single-row pipeline.

        Returns a dict with:
            crop            -> predicted crop name
            confidence      -> probability of the predicted crop (0-1)
            top_crops       -> list of (crop, prob) top-k alternatives
            shap_factors    -> list of (feature, shap_value) top features
        """
        X_fused = self._fuse(X_raw)                      # (1, 22)
        proba = self.model.predict_proba(X_fused)[0]     # (22,)

        pred_idx = int(np.argmax(proba))
        crop = self.class_names[pred_idx]
        confidence = float(proba[pred_idx])

        order = np.argsort(proba)[::-1][:top_k]
        top_crops = [(self.class_names[i], float(proba[i])) for i in order]

        # SHAP values from the dedicated Random Forest proxy
        shap_factors = self._shap_for_class(X_fused, pred_idx, top_feats)

        return {
            "crop": crop,
            "confidence": confidence,
            "top_crops": top_crops,
            "shap_factors": shap_factors,
        }

    # ------------------------------------------------------------------ #
    def _shap_for_class(self, X_fused: np.ndarray, class_idx: int, top_feats: int):
        """Compute top SHAP contributions for the predicted class, sorted by impact."""
        raw = self.shap_explainer.shap_values(X_fused)

        # Different SHAP versions return different shapes; normalise to (n_features,)
        vals = None
        if isinstance(raw, list):
            # list of arrays, one per class
            vals = np.asarray(raw[class_idx])[0]
        else:
            arr = np.asarray(raw)
            if arr.ndim == 3:          # (samples, features, classes)
                vals = arr[0, :, class_idx]
            elif arr.ndim == 2:        # (samples, features)
                vals = arr[0]
            else:
                vals = arr

        vals = np.asarray(vals).flatten()
        n = min(len(vals), len(self.feature_names))
        pairs = list(zip(self.feature_names[:n], vals[:n]))
        # sort by absolute impact, keep the strongest
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return [(name, float(v)) for name, v in pairs[:top_feats]]


@st.cache_resource(show_spinner=False)
def load_recommender() -> "CropRecommender":
    """Load and cache the CropRecommender once per server process, shared by every page."""
    return CropRecommender()
