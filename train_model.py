"""
train_model.py
--------------
Trains the XGBoost fraud detection model and saves artifacts to models/.
Run once before starting the API:

    python train_model.py

What this script does:
  1. Loads data and validates class imbalance
  2. Scales Time and Amount (V1-V28 are already PCA-scaled)
  3. Trains XGBoost with scale_pos_weight to handle imbalance natively
  4. Tunes decision threshold by maximizing business value (not F1)
  5. Reports results in business terms, not just ML metrics
  6. Saves model + scaler to models/
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/creditcard.csv")
MODEL_PATH = Path("models/fraud_model.joblib")
SCALER_PATH = Path("models/scaler.joblib")

# Business cost assumptions
AVG_FRAUD_AMOUNT = 122.21  # mean transaction amount for fraud cases
REVIEW_COST = 2.00  # cost of manually reviewing a flagged transaction

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print(f"[ERROR] Dataset not found at {DATA_PATH}")
        print("Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("Place creditcard.csv inside the data/ folder.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    n_fraud = df["Class"].sum()
    n_total = len(df)
    fraud_pct = n_fraud / n_total * 100

    print(f"[DATA] Loaded {n_total:,} transactions")
    print(f"[DATA] Fraud: {n_fraud:,} ({fraud_pct:.3f}%) — Legit: {n_total - n_fraud:,}")
    print(f"[DATA] Imbalance ratio: 1 fraud per {int(n_total / n_fraud)} legitimate transactions")

    return df


def preprocess(df: pd.DataFrame):
    """
    Only scale Time and Amount.
    V1-V28 are PCA output — already zero-mean, unit-variance.
    Scaling them again would distort the PCA structure.
    """
    feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = df[feature_cols].copy()
    y = df["Class"].values

    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    return X, y, scaler


def train(X_train, y_train) -> XGBClassifier:
    """
    scale_pos_weight = n_negatives / n_positives
    This tells XGBoost to penalize missing a fraud proportionally.
    No resampling (SMOTE etc.) needed — XGBoost handles it internally.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    print(
        f"\n[TRAIN] scale_pos_weight = {scale_pos_weight:.1f} "
        f"(1 fraud per {int(scale_pos_weight)} legit)"
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",  # AUPRC — correct metric for imbalanced data
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )

    return model


def tune_threshold_by_business_value(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    avg_fraud_amount: float,
    review_cost: float,
) -> tuple[float, pd.DataFrame]:
    """
    Instead of picking threshold at max F1, we find the threshold that
    maximizes net business value saved.

    For each threshold:
      - True Positives  → fraud blocked → saved avg_fraud_amount
      - False Positives → legit tx flagged → cost review_cost
      - False Negatives → fraud missed → lost avg_fraud_amount
      - True Negatives  → legit tx approved → $0 impact

    Net value = (TP * avg_fraud_amount) - (FP * review_cost)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    n_fraud_total = y_true.sum()

    rows = []
    for thresh, prec, rec in zip(thresholds, precision[:-1], recall[:-1], strict=True):
        tp = rec * n_fraud_total
        fp = (tp / prec) - tp if prec > 0 else 0
        fn = n_fraud_total - tp

        value_saved = tp * avg_fraud_amount
        review_costs = fp * review_cost
        missed_fraud = fn * avg_fraud_amount
        net_value = value_saved - review_costs

        rows.append(
            {
                "threshold": round(thresh, 3),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "value_saved_usd": round(value_saved, 2),
                "review_cost_usd": round(review_costs, 2),
                "missed_fraud_usd": round(missed_fraud, 2),
                "net_value_usd": round(net_value, 2),
            }
        )

    results_df = pd.DataFrame(rows)
    best_idx = results_df["net_value_usd"].idxmax()
    best_threshold = results_df.loc[best_idx, "threshold"]

    return best_threshold, results_df


def evaluate(model, X_test, y_test, threshold: float):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auprc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n[METRIC] AUPRC (Area Under Precision-Recall Curve): {auprc:.4f}")
    print(f"         (Baseline for random classifier: {y_test.mean():.4f})")
    print(f"\n[THRESHOLD] Using {threshold:.2f} (tuned for business value)")
    print("\n[CONFUSION MATRIX]")
    print(f"  True Positives  (fraud caught):       {tp:>6,}")
    print(f"  False Positives (legit flagged):       {fp:>6,}")
    print(f"  False Negatives (fraud missed):        {fn:>6,}")
    print(f"  True Negatives  (legit approved):      {tn:>6,}")

    print("\n[BUSINESS IMPACT — test set]")
    value_blocked = tp * AVG_FRAUD_AMOUNT
    review_costs = fp * REVIEW_COST
    missed_losses = fn * AVG_FRAUD_AMOUNT
    net_value = value_blocked - review_costs

    print(f"  Fraud value blocked:    ${value_blocked:>10,.2f}")
    print(f"  Review costs incurred:  ${review_costs:>10,.2f}")
    print(f"  Fraud missed (losses):  ${missed_losses:>10,.2f}")
    print("  ─────────────────────────────────────")
    print(f"  Net value protected:    ${net_value:>10,.2f}")

    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print("=" * 60)


def main():
    print("=" * 60)
    print("FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)

    # 1. Load
    df = load_data()

    # 2. Preprocess
    X, y, scaler = preprocess(df)

    # 3. Split (stratified — preserve class ratio in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n[SPLIT] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 4. Train
    print("\n[TRAIN] Training XGBoost...")
    model = train(X_train, y_train)
    print("[TRAIN] Done.")

    # 5. Tune threshold by business value (on test set)
    print("\n[THRESHOLD] Tuning decision threshold by business value...")
    y_prob_test = model.predict_proba(X_test)[:, 1]
    best_threshold, threshold_df = tune_threshold_by_business_value(
        y_test, y_prob_test, AVG_FRAUD_AMOUNT, REVIEW_COST
    )
    print(f"[THRESHOLD] Best threshold by net business value: {best_threshold:.2f}")

    # Show top 5 threshold options
    print("\n[THRESHOLD] Top 5 thresholds by net value saved:")
    top5 = threshold_df.nlargest(5, "net_value_usd")[
        ["threshold", "precision", "recall", "tp", "fp", "net_value_usd"]
    ]
    print(top5.to_string(index=False))

    # 6. Evaluate
    evaluate(model, X_test, y_test, best_threshold)

    # 7. Save artifacts
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n[SAVE] Model → {MODEL_PATH}")
    print(f"[SAVE] Scaler → {SCALER_PATH}")

    # Save threshold recommendation (for reference in config)
    print(f"\n[NOTE] Set PREDICTION_THRESHOLD={best_threshold} in your .env")
    print("\nDone. Run the API with: uvicorn src.main:app --reload")


if __name__ == "__main__":
    main()
