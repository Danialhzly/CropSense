"""
Preprocessing and Hybrid Feature Fusion.

This module implements the hybrid feature fusion approach described in
Chapter 3, replicating the methodology of Ahmed et al. (2025).

The fusion combines THREE feature engineering / selection methods that each
work on a different principle:

    1. PCA  (feature ENGINEERING)  -> creates 5 brand-new components
    2. MI   (feature SELECTION)    -> selects the 5 most informative originals
    3. RFE  (feature SELECTION)    -> selects the 5 best originals via a model

Final fused set:
    7 original (scaled) + 5 PCA + 5 MI + 5 RFE = 22 fused features

All three methods are FITTED ONCE on the training data during
train_model.py, then bundled into fusion_pipeline.joblib. At prediction
time the SAME fitted objects are re-applied — nothing is re-fitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif

# The 7 original input features, in a fixed canonical order.
BASE_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

RANDOM_STATE = 42
N_PCA = 5
N_MI = 5
N_RFE = 5


class HybridFeatureFusion:
    """
    Fits and applies the PCA + MI + RFE hybrid feature fusion.

    Usage (training):
        fusion = HybridFeatureFusion()
        X_fused = fusion.fit_transform(X_scaled_train, y_train)

    Usage (inference):
        X_fused = fusion.transform(X_scaled_new)

    The fitted instance is what gets saved as fusion_pipeline.joblib.
    """

    def __init__(self, n_pca: int = N_PCA, n_mi: int = N_MI, n_rfe: int = N_RFE):
        self.n_pca = n_pca
        self.n_mi = n_mi
        self.n_rfe = n_rfe

        self.pca: PCA | None = None
        self.mi_indices: list[int] = []
        self.rfe_indices: list[int] = []
        self.fused_feature_names: list[str] = []
        self.pca_explained_variance_: float = 0.0

    # ------------------------------------------------------------------ #
    def fit(self, X_scaled: np.ndarray, y: np.ndarray) -> "HybridFeatureFusion":
        """Learn PCA rotation, MI ranking, and RFE selection from training data."""
        # --- PCA: feature engineering ---
        self.pca = PCA(n_components=self.n_pca, random_state=RANDOM_STATE)
        self.pca.fit(X_scaled)
        self.pca_explained_variance_ = float(self.pca.explained_variance_ratio_.sum())

        # --- Mutual Information: feature selection ---
        mi_scores = mutual_info_classif(X_scaled, y, random_state=RANDOM_STATE)
        # indices of the top-n MI features (highest score first)
        self.mi_indices = list(np.argsort(mi_scores)[::-1][: self.n_mi])

        # --- RFE with Random Forest (100 trees): feature selection ---
        rfe = RFE(
            estimator=RandomForestClassifier(
                n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
            ),
            n_features_to_select=self.n_rfe,
            step=1,
        )
        rfe.fit(X_scaled, y)
        self.rfe_indices = list(np.where(rfe.support_)[0])

        self._build_feature_names()
        return self

    # ------------------------------------------------------------------ #
    def transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Apply the already-fitted PCA / MI / RFE to new scaled data."""
        if self.pca is None:
            raise RuntimeError("HybridFeatureFusion must be fitted before transform().")

        X_pca = self.pca.transform(X_scaled)            # 5 new components
        X_mi = X_scaled[:, self.mi_indices]             # 5 selected originals
        X_rfe = X_scaled[:, self.rfe_indices]           # 5 selected originals

        # 7 original + 5 PCA + 5 MI + 5 RFE = 22
        return np.hstack([X_scaled, X_pca, X_mi, X_rfe])

    # ------------------------------------------------------------------ #
    def fit_transform(self, X_scaled: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X_scaled, y).transform(X_scaled)

    # ------------------------------------------------------------------ #
    def _build_feature_names(self) -> None:
        """Produce readable names for all 22 fused features (for SHAP display)."""
        names = list(BASE_FEATURES)                             # 7 original
        names += [f"PCA_{i + 1}" for i in range(self.n_pca)]    # 5 PCA
        names += [f"MI_{BASE_FEATURES[i]}" for i in self.mi_indices]   # 5 MI
        names += [f"RFE_{BASE_FEATURES[i]}" for i in self.rfe_indices] # 5 RFE
        self.fused_feature_names = names


def build_raw_input(
    N: float, P: float, K: float,
    temperature: float, humidity: float, ph: float, rainfall: float,
) -> np.ndarray:
    """Turn a single farmer's 7 slider values into a (1, 7) array in canonical order."""
    return np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)


def dataframe_to_raw(df: pd.DataFrame) -> np.ndarray:
    """Extract the 7 base features from a DataFrame in canonical order (for batch)."""
    return df[BASE_FEATURES].values.astype(float)
