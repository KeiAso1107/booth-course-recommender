"""
Enhanced Bid Price Prediction Model v2.
Improvements:
- Log-transform target (prices are right-skewed)
- Separate models for zero vs non-zero prices
- Quantile regression for confidence intervals
- Better feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, classification_report
import pickle
from pathlib import Path

from data_processing import load_bid_data, load_evaluation_data, DATA_DIR
from bid_model import prepare_model_data


def train_two_stage_model(bid_df, eval_df, target_phase="Phase 2"):
    """
    Two-stage model:
    Stage 1: Classify whether price > 0 (demand vs no demand)
    Stage 2: Predict price for courses with price > 0 (using log transform)
    """
    df = prepare_model_data(bid_df, eval_df, target_phase=target_phase)

    feature_cols = [
        "hist_price_mean", "hist_price_max", "hist_price_std",
        "hist_price_last", "hist_price_last2", "hist_count", "price_trend",
        "est_capacity", "fill_ratio", "is_evening", "is_weekend",
        "day_monday", "day_tuesday", "day_wednesday", "day_thursday", "day_friday",
        "eval_clarity", "eval_interesting", "eval_useful",
        "eval_got_out", "eval_recommend", "eval_hours", "eval_count",
        "instructor_avg_price", "instructor_max_price", "instructor_course_count",
        "quarter_sin", "quarter_cos", "year_norm",
        "dept_frequency",
    ]
    if target_phase == "Phase 2":
        feature_cols.append("p1_price")

    # Drop rows without history
    model_df = df.dropna(subset=["hist_price_mean"]).copy()

    X = model_df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median())
    y = model_df["clearing_price"]

    # ===== STAGE 1: Classify zero vs non-zero =====
    y_binary = (y > 0).astype(int)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1, class_weight="balanced"
    )

    tscv = TimeSeriesSplit(n_splits=5)
    clf_scores = cross_val_score(clf, X, y_binary, cv=tscv, scoring="accuracy")
    print(f"\n{'='*60}")
    print(f"Stage 1: Zero vs Non-Zero Classification ({target_phase})")
    print(f"{'='*60}")
    print(f"Accuracy: {clf_scores.mean():.3f} ± {clf_scores.std():.3f}")

    clf.fit(X, y_binary)
    print(f"Train accuracy: {accuracy_score(y_binary, clf.predict(X)):.3f}")

    # ===== STAGE 2: Predict price for non-zero courses =====
    nonzero_mask = y > 0
    X_nz = X[nonzero_mask]
    y_nz = y[nonzero_mask]
    y_log = np.log1p(y_nz)  # log(1+price) transform

    print(f"\n{'='*60}")
    print(f"Stage 2: Price Prediction for Non-Zero Courses ({target_phase})")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_nz)}")
    print(f"Price stats: mean={y_nz.mean():.0f}, median={y_nz.median():.0f}, max={y_nz.max():.0f}")

    # Random Forest on log-transformed prices
    rf_log = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )

    # CV on log scale
    tscv_nz = TimeSeriesSplit(n_splits=5)
    log_mae_scores = cross_val_score(rf_log, X_nz, y_log, cv=tscv_nz, scoring="neg_mean_absolute_error")

    # CV on original scale
    original_mae_scores = []
    for train_idx, test_idx in tscv_nz.split(X_nz):
        X_tr, X_te = X_nz.iloc[train_idx], X_nz.iloc[test_idx]
        y_tr, y_te = y_log.iloc[train_idx], y_nz.iloc[test_idx]
        rf_temp = RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        )
        rf_temp.fit(X_tr, y_tr)
        y_pred_log = rf_temp.predict(X_te)
        y_pred = np.expm1(y_pred_log)
        original_mae_scores.append(mean_absolute_error(y_te, y_pred))

    print(f"Log-scale MAE (CV): {-log_mae_scores.mean():.3f} ± {log_mae_scores.std():.3f}")
    print(f"Original-scale MAE (CV): {np.mean(original_mae_scores):.0f} ± {np.std(original_mae_scores):.0f}")

    # Fit final model
    rf_log.fit(X_nz, y_log)
    y_pred_train = np.expm1(rf_log.predict(X_nz))
    print(f"Train MAE: {mean_absolute_error(y_nz, y_pred_train):.0f}")
    print(f"Train R²:  {r2_score(y_nz, y_pred_train):.3f}")

    # Gradient Boosting for comparison
    gb_log = GradientBoostingRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    )
    gb_original_mae = []
    for train_idx, test_idx in tscv_nz.split(X_nz):
        X_tr, X_te = X_nz.iloc[train_idx], X_nz.iloc[test_idx]
        y_tr, y_te = y_log.iloc[train_idx], y_nz.iloc[test_idx]
        gb_temp = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            min_samples_leaf=5, subsample=0.8, random_state=42,
        )
        gb_temp.fit(X_tr, y_tr)
        y_pred = np.expm1(gb_temp.predict(X_te))
        gb_original_mae.append(mean_absolute_error(y_te, y_pred))

    print(f"\nGradient Boosting MAE (CV): {np.mean(gb_original_mae):.0f} ± {np.std(gb_original_mae):.0f}")

    # Choose best
    if np.mean(gb_original_mae) < np.mean(original_mae_scores):
        gb_log.fit(X_nz, y_log)
        best_reg = gb_log
        best_name = "Gradient Boosting"
        best_mae = np.mean(gb_original_mae)
    else:
        best_reg = rf_log
        best_name = "Random Forest"
        best_mae = np.mean(original_mae_scores)

    print(f"\nBest regressor: {best_name} (MAE={best_mae:.0f})")

    # Feature importance
    importances = pd.Series(best_reg.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print(f"\nTop 10 feature importances:")
    for feat, imp in importances.head(10).items():
        print(f"  {feat:30s} {imp:.4f}")

    return clf, best_reg, feature_cols


def predict_two_stage(clf, reg, feature_cols, course_features):
    """
    Predict bid price using two-stage model.
    Returns dict with prediction details.
    """
    X = pd.DataFrame([course_features])[feature_cols].fillna(0)

    # Stage 1: Will it have a non-zero price?
    prob_nonzero = clf.predict_proba(X)[0][1]

    # Stage 2: If non-zero, what price?
    pred_log = reg.predict(X)[0]
    pred_price = np.expm1(pred_log)

    # Uncertainty from RF trees
    if hasattr(reg, "estimators_"):
        X_vals = X.values
        tree_preds = np.array([np.expm1(tree.predict(X_vals)[0]) for tree in reg.estimators_])
        std = np.std(tree_preds)
        percentile_25 = np.percentile(tree_preds, 25)
        percentile_75 = np.percentile(tree_preds, 75)
    else:
        std = pred_price * 0.3  # rough estimate
        percentile_25 = pred_price * 0.7
        percentile_75 = pred_price * 1.3

    # Expected price (probability-weighted)
    expected_price = prob_nonzero * pred_price

    return {
        "prob_nonzero": prob_nonzero,
        "predicted_price_if_nonzero": max(0, pred_price),
        "expected_price": max(0, expected_price),
        "price_p25": max(0, percentile_25),
        "price_p75": percentile_75,
        "price_std": std,
        "bid_recommendation": _bid_recommendation(prob_nonzero, pred_price),
    }


def _bid_recommendation(prob_nonzero, pred_price):
    """Generate a bid strategy recommendation."""
    if prob_nonzero < 0.3:
        return "FREE — Likely no bidding needed. Save your points."
    elif prob_nonzero < 0.6:
        return f"LOW DEMAND — Consider minimal bid (~{int(pred_price * 0.3):,} pts)."
    elif pred_price < 2000:
        return f"MODERATE — Bid around {int(pred_price * 1.1):,} pts to be safe."
    elif pred_price < 8000:
        return f"HIGH DEMAND — Budget {int(pred_price * 1.2):,}+ pts. Popular course."
    else:
        return f"VERY HIGH DEMAND — Premium course. Budget {int(pred_price * 1.3):,}+ pts or try alternate section/quarter."


def evaluate_latest_quarter(clf, reg, feature_cols, bid_df, eval_df, target_phase="Phase 2"):
    """Evaluate on latest quarter."""
    df = prepare_model_data(bid_df, eval_df, target_phase=target_phase)
    latest_term = df["term_order"].max()
    latest_df = df[df["term_order"] == latest_term].dropna(subset=["hist_price_mean"]).copy()

    if len(latest_df) == 0:
        print("No data for evaluation.")
        return

    X = latest_df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median())

    y_actual = latest_df["clearing_price"]

    # Two-stage prediction
    prob_nz = clf.predict_proba(X)[:, 1]
    pred_log = reg.predict(X)
    pred_price = np.expm1(pred_log) * prob_nz  # expected price

    latest_df = latest_df.copy()
    latest_df["predicted"] = pred_price
    latest_df["prob_nonzero"] = prob_nz
    latest_df["error"] = pred_price - y_actual

    # Overall metrics
    mae = mean_absolute_error(y_actual, pred_price)
    r2 = r2_score(y_actual, pred_price)

    # Metrics for non-zero only
    nz_mask = y_actual > 0
    if nz_mask.sum() > 0:
        mae_nz = mean_absolute_error(y_actual[nz_mask], pred_price[nz_mask])
        r2_nz = r2_score(y_actual[nz_mask], pred_price[nz_mask])
    else:
        mae_nz, r2_nz = 0, 0

    quarter = latest_df["quarter"].iloc[0]
    year = latest_df["year"].iloc[0]

    print(f"\n{'='*60}")
    print(f"Two-Stage Model Evaluation: {quarter} {year} ({target_phase})")
    print(f"{'='*60}")
    print(f"Total courses: {len(latest_df)}")
    print(f"Non-zero prices: {nz_mask.sum()}")
    print(f"Overall MAE: {mae:.0f}")
    print(f"Overall R²:  {r2:.3f}")
    print(f"Non-zero MAE: {mae_nz:.0f}")
    print(f"Non-zero R²:  {r2_nz:.3f}")

    # Top predictions vs actual
    print(f"\nTop 20 (by actual price):")
    top = latest_df.nlargest(20, "clearing_price")[
        ["course_code", "title", "clearing_price", "predicted", "prob_nonzero", "error"]
    ]
    for _, row in top.iterrows():
        print(f"  {row['course_code']:12s} {row['title'][:45]:45s} "
              f"actual={row['clearing_price']:8,.0f}  pred={row['predicted']:8,.0f}  "
              f"p(nz)={row['prob_nonzero']:.2f}  err={row['error']:+8,.0f}")

    return latest_df


if __name__ == "__main__":
    print("Loading data...")
    bid_df = load_bid_data()
    eval_df = load_evaluation_data()

    # Train Phase 2 two-stage model
    clf_p2, reg_p2, feat_p2 = train_two_stage_model(bid_df, eval_df, target_phase="Phase 2")

    # Evaluate on latest quarter
    evaluate_latest_quarter(clf_p2, reg_p2, feat_p2, bid_df, eval_df, target_phase="Phase 2")

    # Train Phase 1 two-stage model
    clf_p1, reg_p1, feat_p1 = train_two_stage_model(bid_df, eval_df, target_phase="Phase 1")

    # Save enhanced models
    model_artifacts = {
        "clf_p2": clf_p2, "reg_p2": reg_p2, "features_p2": feat_p2,
        "clf_p1": clf_p1, "reg_p1": reg_p1, "features_p1": feat_p1,
    }
    with open(DATA_DIR / "bid_model_v2.pkl", "wb") as f:
        pickle.dump(model_artifacts, f)
    print("\nEnhanced models saved to bid_model_v2.pkl")
