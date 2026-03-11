"""
Bid Price Prediction Model for Chicago Booth Course Recommender.
Predicts clearing prices using historical bid data and course evaluations.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path

from data_processing import load_bid_data, load_evaluation_data, DATA_DIR


def prepare_model_data(bid_df, eval_df, target_phase="Phase 2"):
    """
    Prepare feature matrix for bid price prediction.

    Target: clearing_price for the specified phase.
    Features: course characteristics, historical prices, evaluation scores,
              time features, instructor effects.
    """
    # Filter to target phase with valid prices
    df = bid_df[bid_df["phase"] == target_phase].copy()
    df = df[df["clearing_price"].notna() & (df["clearing_price"] >= 0)]

    # --- Course-level historical stats (using data BEFORE each observation) ---
    # Sort by time
    df = df.sort_values("term_order")

    # Evaluation features per dept_code
    eval_agg = eval_df.groupby("dept_code").agg(
        eval_clarity=("clarity", "mean"),
        eval_interesting=("interesting", "mean"),
        eval_useful=("useful_tools", "mean"),
        eval_got_out=("how_much_got", "mean"),
        eval_recommend=("recommend", "mean"),
        eval_hours=("hours_per_week", "mean"),
        eval_response_rate=("response_rate", "mean"),
        eval_count=("course_code_clean", "count"),
    ).reset_index()

    df = df.merge(eval_agg, on="dept_code", how="left")

    # --- Instructor popularity (average price across all their courses) ---
    instructor_stats = bid_df[
        (bid_df["phase"] == target_phase) & (bid_df["clearing_price"].notna())
    ].groupby("instructor").agg(
        instructor_avg_price=("clearing_price", "mean"),
        instructor_max_price=("clearing_price", "max"),
        instructor_course_count=("dept_code", "nunique"),
    ).reset_index()

    df = df.merge(instructor_stats, on="instructor", how="left")

    # --- Historical price features for each course ---
    # Rolling averages: for each row, compute stats from previous offerings
    hist_features = []
    for _, group in df.groupby("dept_code"):
        group = group.sort_values("term_order")
        # Expanding mean/max of previous prices (shift to avoid leakage)
        group["hist_price_mean"] = group["clearing_price"].shift().expanding().mean()
        group["hist_price_max"] = group["clearing_price"].shift().expanding().max()
        group["hist_price_std"] = group["clearing_price"].shift().expanding().std()
        group["hist_price_last"] = group["clearing_price"].shift(1)
        group["hist_price_last2"] = group["clearing_price"].shift(2)
        group["hist_count"] = group["clearing_price"].shift().expanding().count()
        # Price trend (last - previous)
        group["price_trend"] = group["hist_price_last"] - group["hist_price_last2"]
        hist_features.append(group)

    df = pd.concat(hist_features, ignore_index=True)

    # --- Phase 1 price as feature (for Phase 2 prediction) ---
    if target_phase == "Phase 2":
        p1 = bid_df[bid_df["phase"] == "Phase 1"][
            ["course_code", "quarter", "year", "clearing_price"]
        ].rename(columns={"clearing_price": "p1_price"})
        df = df.merge(p1, on=["course_code", "quarter", "year"], how="left")

    # --- Time features ---
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter_num"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter_num"] / 4)
    df["year_norm"] = (df["year"] - df["year"].min()) / max(df["year"].max() - df["year"].min(), 1)

    # --- Demand indicator: fill ratio at Phase 1 ---
    # Already in df from data_processing

    # --- Encode categorical: department ---
    # Use dept frequency as a feature instead of one-hot (too many categories)
    dept_freq = df["dept_code"].value_counts()
    df["dept_frequency"] = df["dept_code"].map(dept_freq)

    return df


def train_model(df, target_phase="Phase 2"):
    """Train and evaluate bid price prediction model."""

    # Feature columns
    feature_cols = [
        # Historical price features
        "hist_price_mean", "hist_price_max", "hist_price_std",
        "hist_price_last", "hist_price_last2", "hist_count", "price_trend",
        # Course characteristics
        "est_capacity", "fill_ratio", "is_evening", "is_weekend",
        "day_monday", "day_tuesday", "day_wednesday", "day_thursday", "day_friday",
        # Evaluation scores
        "eval_clarity", "eval_interesting", "eval_useful",
        "eval_got_out", "eval_recommend", "eval_hours", "eval_count",
        # Instructor
        "instructor_avg_price", "instructor_max_price", "instructor_course_count",
        # Time
        "quarter_sin", "quarter_cos", "year_norm",
        # Demand
        "dept_frequency",
    ]

    if target_phase == "Phase 2":
        feature_cols.append("p1_price")

    # Drop rows with no historical data (first offering of each course)
    model_df = df.dropna(subset=["hist_price_mean"]).copy()

    # Fill remaining NaN in features with median
    X = model_df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median())

    y = model_df["clearing_price"]

    print(f"\n{'='*60}")
    print(f"Model for {target_phase} Price Prediction")
    print(f"{'='*60}")
    print(f"Training samples: {len(X)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target stats: mean={y.mean():.0f}, median={y.median():.0f}, max={y.max():.0f}")

    # --- Model 1: Gradient Boosting ---
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )

    # Time-series cross-validation (respect temporal order)
    tscv = TimeSeriesSplit(n_splits=5)
    gb_scores_mae = cross_val_score(gb_model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
    gb_scores_r2 = cross_val_score(gb_model, X, y, cv=tscv, scoring="r2")

    print(f"\nGradient Boosting (5-fold time-series CV):")
    print(f"  MAE:  {-gb_scores_mae.mean():.0f} ± {gb_scores_mae.std():.0f}")
    print(f"  R²:   {gb_scores_r2.mean():.3f} ± {gb_scores_r2.std():.3f}")

    # --- Model 2: Random Forest ---
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    rf_scores_mae = cross_val_score(rf_model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
    rf_scores_r2 = cross_val_score(rf_model, X, y, cv=tscv, scoring="r2")

    print(f"\nRandom Forest (5-fold time-series CV):")
    print(f"  MAE:  {-rf_scores_mae.mean():.0f} ± {rf_scores_mae.std():.0f}")
    print(f"  R²:   {rf_scores_r2.mean():.3f} ± {rf_scores_r2.std():.3f}")

    # --- Train final model on all data ---
    # Choose best model
    if gb_scores_r2.mean() >= rf_scores_r2.mean():
        best_model = gb_model
        best_name = "Gradient Boosting"
    else:
        best_model = rf_model
        best_name = "Random Forest"

    print(f"\nBest model: {best_name}")

    best_model.fit(X, y)

    # Feature importance
    importances = pd.Series(best_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)
    print(f"\nTop 15 feature importances:")
    for feat, imp in importances.head(15).items():
        print(f"  {feat:30s} {imp:.4f}")

    # Train error
    y_pred = best_model.predict(X)
    print(f"\nTrain MAE: {mean_absolute_error(y, y_pred):.0f}")
    print(f"Train R²:  {r2_score(y, y_pred):.3f}")
    print(f"Train RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.0f}")

    return best_model, feature_cols, X, y


def predict_price(model, feature_cols, course_data):
    """
    Predict clearing price for a course.

    course_data: dict with feature values
    Returns: predicted price and confidence interval
    """
    X = pd.DataFrame([course_data])[feature_cols]
    X = X.fillna(0)

    pred = model.predict(X)[0]

    # For GBR, use staged predictions for uncertainty estimate
    if hasattr(model, "staged_predict"):
        staged = list(model.staged_predict(X))
        # Use last 50 stages for uncertainty
        recent = [s[0] for s in staged[-50:]]
        std = np.std(recent)
    else:
        # For RF, use tree predictions
        tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
        std = np.std(tree_preds)

    return {
        "predicted_price": max(0, pred),
        "price_low": max(0, pred - 1.96 * std),
        "price_high": pred + 1.96 * std,
        "confidence_std": std,
    }


def build_prediction_table(model, feature_cols, bid_df, eval_df):
    """
    Build predictions for all courses in the latest quarter.
    """
    # Get latest quarter
    latest_term = bid_df["term_order"].max()
    latest = bid_df[bid_df["term_order"] == latest_term]
    latest_quarter = latest["quarter"].iloc[0]
    latest_year = latest["year"].iloc[0]
    print(f"\nPredicting for: {latest_quarter} {latest_year}")

    # Prepare features for each course in latest quarter
    df = prepare_model_data(bid_df, eval_df, target_phase="Phase 2")

    # Get latest quarter data
    latest_df = df[df["term_order"] == latest_term].copy()

    if len(latest_df) == 0:
        print("No Phase 2 data for latest quarter yet.")
        return pd.DataFrame()

    X = latest_df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median())

    latest_df["predicted_price"] = model.predict(X)
    latest_df["predicted_price"] = latest_df["predicted_price"].clip(lower=0)

    result = latest_df[["course_code", "title", "instructor", "day_time",
                         "clearing_price", "predicted_price", "eval_recommend",
                         "est_capacity", "fill_ratio"]].copy()
    result["prediction_error"] = result["predicted_price"] - result["clearing_price"]
    result = result.sort_values("clearing_price", ascending=False)

    return result


if __name__ == "__main__":
    print("Loading data...")
    bid_df = load_bid_data()
    eval_df = load_evaluation_data()

    # Train Phase 2 model (most relevant for bidding)
    df_p2 = prepare_model_data(bid_df, eval_df, target_phase="Phase 2")
    model_p2, features_p2, X_p2, y_p2 = train_model(df_p2, target_phase="Phase 2")

    # Train Phase 1 model
    df_p1 = prepare_model_data(bid_df, eval_df, target_phase="Phase 1")
    model_p1, features_p1, X_p1, y_p1 = train_model(df_p1, target_phase="Phase 1")

    # Save models
    model_artifacts = {
        "model_p2": model_p2,
        "features_p2": features_p2,
        "model_p1": model_p1,
        "features_p1": features_p1,
    }
    with open(DATA_DIR / "bid_model.pkl", "wb") as f:
        pickle.dump(model_artifacts, f)
    print("\nModels saved to bid_model.pkl")

    # Build prediction table for latest quarter
    print("\n" + "=" * 60)
    print("Prediction vs Actual for latest quarter")
    print("=" * 60)
    pred_table = build_prediction_table(model_p2, features_p2, bid_df, eval_df)
    if len(pred_table) > 0:
        print(pred_table.head(30).to_string(index=False))
        mae = mean_absolute_error(
            pred_table["clearing_price"], pred_table["predicted_price"]
        )
        print(f"\nLatest quarter MAE: {mae:.0f}")
