# CropSense — Explainable Crop Recommendation System

An Explainable Crop Recommendation System Using Hybrid Feature Fusion and Stacking Ensemble Learning.

This web application recommends the most suitable crop for a plot of land based on seven soil and climate measurements, and explains the reasoning behind every recommendation using SHAP.

## What it does

- Takes 7 inputs: Nitrogen, Phosphorus, Potassium, temperature, humidity, pH, rainfall
- Expands them into 22 features using **hybrid feature fusion** (PCA + Mutual Information + RFE)
- Predicts the best crop using a **Second Stacking ensemble** (KNN + Bagging + Naive Bayes with a Logistic Regression meta-learner)
- Explains the recommendation using **SHAP** (via a dedicated Random Forest proxy)
- Compares 9 machine learning models on a performance dashboard

## Project structure

```
FYP WEBSITE/
├── app.py                      # Streamlit entry point (routing + navbar)
├── train_model.py              # Offline training pipeline (run this first)
├── requirements.txt
├── data/
│   └── Crop_recommendation.csv # Kaggle dataset (2,200 rows, 22 crops)
├── models/                     # Created by train_model.py
│   ├── production_model.joblib     # Second Stacking champion
│   ├── scaler.joblib
│   ├── label_encoder.joblib
│   ├── fusion_pipeline.joblib      # Fitted PCA + MI + RFE
│   ├── shap_rf_model.joblib        # Random Forest proxy for SHAP
│   ├── shap_explainer.joblib       # SHAP TreeExplainer
│   ├── metrics.json                # All 9 models' results
│   └── feature_names.json
├── utils/
│   ├── preprocessing.py        # HybridFeatureFusion (PCA + MI + RFE)
│   ├── explainer.py            # Inference + SHAP at prediction time
│   └── styles.py               # Shared CSS + constants
└── pages_impl/
    ├── home.py
    ├── single_prediction.py
    ├── batch_prediction.py
    ├── performance.py
    └── about.py
```

## Setup

1. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Train the models (this creates everything in `models/`):

   ```
   python train_model.py
   ```

   This runs the full pipeline: cleaning, encoding, an 80/20 stratified split, scaling,
   hybrid feature fusion, training all 9 models, 5-fold cross-validation, and building the
   SHAP explainer. It takes a few minutes and only needs to be done once.

3. Launch the web app:

   ```
   streamlit run app.py
   ```

   The app opens in your browser. If the models haven't been trained yet, the app will
   remind you to run `train_model.py` first.

## The 5 pages

1. **Home** — project overview
2. **Predict** — enter 7 values, get a crop recommendation with a probability chart and SHAP explanation
3. **Batch** — upload a CSV to get recommendations for many rows at once
4. **Dashboard** — accuracy comparison of all 9 models, confusion matrix, full metrics table
5. **About** — methodology and technology explanation

## Notes

- All 9 models receive the same 22 fused features, which keeps the comparison fair.
- `random_state=42` is fixed throughout so results are reproducible.
- The scaler and fusion pipeline are fitted on the training set only, to prevent data leakage.
