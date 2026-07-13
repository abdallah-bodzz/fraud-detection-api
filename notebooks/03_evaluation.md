# 03 - Evaluation & Business Impact

**Project:** Fraud Detection API
**Lead Developer:** Abdallah A Khames
**Organization:** BODZZ
**GitHub:** [abdallah-bodzz](https://github.com/abdallah-bodzz) - **Repo:** `fraud-detection-api`

---

## Objective

Translate model output into a production decision boundary and a dollar-
denominated business case. This notebook answers:

1. How well does the model rank fraud above legitimate transactions (AUPRC)?
2. Where should the decision threshold sit if optimized for net business
   value, not F1 or accuracy?
3. Are the model's output probabilities trustworthy enough to use directly
   in a dollar-cost calculation, or are they miscalibrated?
4. What does the final model actually deliver in blocked fraud value, review
   cost, and missed-fraud exposure on held-out data?
5. What would this look like at annualized scale, and with what caveats?

This notebook loads the model trained in `02_model_training.ipynb` - it does
not retrain. Its job is honest measurement, not further optimization.

## 0. Setup

Imports the canonical test split and the saved model/scaler pair from
`_shared.py` - same random seed, same split, same palette as
`01_eda.ipynb` and `02_model_training.ipynb`.

**Cell 1:**
```python
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

import _shared as shared

shared.set_plot_theme()
pd.set_option('display.max_columns', 40)
pd.set_option('display.float_format', lambda v: f'{v:,.4f}')

print(f'Random state       : {shared.RANDOM_STATE}')
print(f'Avg fraud amount   : ${shared.AVG_FRAUD_AMOUNT:,.2f}')
print(f'Review cost        : ${shared.REVIEW_COST:,.2f}')
```

**Output:**
```
Random state       : 42
Avg fraud amount   : $122.21
Review cost        : $2.00
```

### Helper functions

Reused verbatim from `01_eda.ipynb` and `02_model_training.ipynb` so
behaviour and output formatting stay identical across the pipeline:

- **`run_query`** - filter a DataFrame with a pandas query expression,
  logging how many rows survived.
- **`save_fig`** - persist a figure to `reports/figures/` via the shared
  helper.
- **`pct`** - consistent percentage formatting across every printed summary.

**Cell 2:**
```python
def run_query(df, expr, label=None):
    '''Filter df with a pandas query expression, logging rows kept.'''
    result = df.query(expr)
    tag = label or expr
    kept_pct = len(result) / len(df) * 100
    print(f'[QUERY] {tag}: {len(result):,} / {len(df):,} rows kept ({kept_pct:.2f}%)')
    return result


def save_fig(fig, name):
    '''Persist a figure to reports/figures/ via the shared helper.'''
    out_path = shared.save_figure(fig, name)
    print(f'[SAVED] {out_path}')
    return out_path


def pct(numerator, denominator, decimals=3):
    '''Format a ratio as a percentage string.'''
    if denominator == 0:
        return 'n/a'
    return f'{numerator / denominator * 100:.{decimals}f}%'
```

## 1. Load Model & Held-Out Test Set

Rebuilds the identical test split used during training (same seed, same
stratification) and loads the model and scaler `02_model_training.ipynb`
saved to disk. No retraining happens in this notebook - what gets measured
here is exactly what `src/model.py` serves in production.

**Cell 3:**
```python
X_train, X_test, y_train, y_test, _ = shared.load_split()

model = joblib.load(shared.MODEL_PATH)
scaler = joblib.load(shared.SCALER_PATH)

y_prob = model.predict_proba(X_test)[:, 1]

print(f'Model loaded  : {shared.MODEL_PATH}')
print(f'Test set size : {len(X_test):,} transactions ({int(y_test.sum())} fraud)')
```

**Output:**
```
Model loaded  : C:\Users\User\Desktop\Projects\(STAGE 4) Fraud Detection API\fraud-detection-api\models\fraud_model.joblib
Test set size : 56,962 transactions (98 fraud)
```

## 2. AUPRC - The Right Metric for Imbalanced Data

ROC-AUC is optimistic on heavily imbalanced data because it is inflated by
the large pool of true negatives that are trivially easy to rank correctly.
AUPRC (Average Precision) measures what actually matters here: how well the
model ranks the rare positive class. AUROC is reported alongside it for
reference, not as the headline number.

**Cell 4:**
```python
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
auprc = average_precision_score(y_test, y_prob)
auroc = roc_auc_score(y_test, y_prob)
random_baseline = y_test.mean()

print(f'AUPRC                  : {auprc:.4f}')
print(f'AUROC (for reference)  : {auroc:.4f}')
print(f'Random baseline AUPRC  : {random_baseline:.4f} (= fraud rate)')
print(f'Lift over random       : {auprc / random_baseline:.0f}x')
```

**Output:**
```
AUPRC                  : 0.8772
AUROC (for reference)  : 0.9815
Random baseline AUPRC  : 0.0017 (= fraud rate)
Lift over random       : 510x
```

**Cell 5:**
```python
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(recall, precision, color=shared.LEGIT_COLOR, lw=2.2, label=f'Model (AUPRC = {auprc:.3f})')
ax.axhline(random_baseline, color=shared.NEUTRAL_COLOR, linestyle='--', lw=1.2,
           label=f'Random baseline ({random_baseline:.3f})')
ax.set_xlabel('Recall (share of fraud caught)')
ax.set_ylabel('Precision (share of flags that are real fraud)')
ax.set_title('Precision-Recall Curve - Held-Out Test Set')
ax.legend()
fig.tight_layout()
save_fig(fig, '09_precision_recall_curve')
plt.show()
```

**Output:**
```
[SAVED] C:\Users\User\Desktop\Projects\(STAGE 4) Fraud Detection API\fraud-detection-api\reports\figures\09_precision_recall_curve.png
```
```
<Figure size 910x715 with 1 Axes>
```

## 3. Probability Calibration

The threshold-tuning step in Section 4 treats the model's output as a real
probability - it multiplies `P(fraud)` by dollar amounts to compute expected
cost. That calculation is only trustworthy if the model is calibrated: among
transactions the model scores at 0.7, roughly 70% should actually be fraud.
Gradient-boosted trees are known to produce probabilities that skew toward
the extremes (overconfident near 0 and 1) unless explicitly calibrated -
this section checks whether that is happening here.

**Cell 6:**
```python
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='quantile')
brier = brier_score_loss(y_test, y_prob)

print(f'Brier score          : {brier:.5f} (lower is better; 0 = perfect, 0.25 = uninformative)')
print(f'Calibration bins     : {len(prob_true)}')
```

**Output:**
```
Brier score          : 0.00050 (lower is better; 0 = perfect, 0.25 = uninformative)
Calibration bins     : 10
```

**Cell 7:**
```python
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot([0, 1], [0, 1], color=shared.NEUTRAL_COLOR, linestyle='--', lw=1.2, label='Perfect calibration')
ax.plot(prob_pred, prob_true, marker='o', color=shared.FRAUD_COLOR, lw=2, markersize=6,
        label=f'Model (Brier = {brier:.4f})')
ax.set_xlabel('Mean predicted probability (per bin)')
ax.set_ylabel('Observed fraud fraction (per bin)')
ax.set_title('Calibration Curve - Reliability Diagram')
ax.legend()
fig.tight_layout()
save_fig(fig, '10_calibration_curve')
plt.show()
```

**Output:**
```
[SAVED] C:\Users\User\Desktop\Projects\(STAGE 4) Fraud Detection API\fraud-detection-api\reports\figures\10_calibration_curve.png
```
```
<Figure size 845x780 with 1 Axes>
```

**Reading & documented tradeoff.** Deviation from the diagonal indicates
over- or under-confidence at that probability range - check the printed
Brier score and plot above before trusting the dollar-cost threshold search
in Section 4 at face value. If calibration is imperfect (it typically is for
boosted trees), the *ranking* of transactions by fraud likelihood remains
reliable - which is what AUPRC measures and what the threshold search
ultimately depends on - but the *absolute* probability values, and therefore
the exact dollar figures below, should be read as directionally correct
rather than precise to the cent. Platt scaling or isotonic regression
(`sklearn.calibration.CalibratedClassifierCV`) would tighten this and is a
documented, deliberately deferred next step - not implemented here to avoid
adding a second fitted transform to the serving path without a demonstrated
need for it.

## 4. Business Value by Threshold

The core reframe of this project: **the decision threshold is a business
decision, not an ML decision.** A missed fraud costs the transaction amount;
a false alarm costs a review. For every candidate threshold we compute net
dollar value, and select the threshold that maximizes it - not the threshold
that maximizes F1 or accuracy.

**Cell 8:**
```python
n_fraud_test = int(y_test.sum())
rows = []

for thresh, prec, rec in zip(thresholds, precision[:-1], recall[:-1]):
    tp = rec * n_fraud_test
    fp = (tp / prec - tp) if prec > 0 else 0
    fn = n_fraud_test - tp

    rows.append({
        'threshold': round(float(thresh), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'value_blocked_usd': tp * shared.AVG_FRAUD_AMOUNT,
        'review_costs_usd': fp * shared.REVIEW_COST,
        'missed_fraud_usd': fn * shared.AVG_FRAUD_AMOUNT,
        'net_value_usd': tp * shared.AVG_FRAUD_AMOUNT - fp * shared.REVIEW_COST,
    })

business_df = pd.DataFrame(rows)
best_row = business_df.loc[business_df['net_value_usd'].idxmax()]
best_threshold = float(best_row['threshold'])

print(f'Best threshold by net business value: {best_threshold:.4f}')
print(f'\nAt threshold = {best_threshold:.4f}:')
print(f'  Precision            : {best_row["precision"]:.2%}')
print(f'  Recall               : {best_row["recall"]:.2%}')
print(f'  Fraud value blocked  : ${best_row["value_blocked_usd"]:,.2f}')
print(f'  Review costs         : ${best_row["review_costs_usd"]:,.2f}')
print(f'  Net value saved      : ${best_row["net_value_usd"]:,.2f}')
```

**Output:**
```
Best threshold by net business value: 0.0933

At threshold = 0.0933:
  Precision            : 54.32%
  Recall               : 89.80%
  Fraud value blocked  : $10,754.48
  Review costs         : $148.00
  Net value saved      : $10,606.48
```

**Cell 9:**
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(business_df['threshold'], business_df['net_value_usd'], color=shared.LEGIT_COLOR, lw=2)
axes[0].axvline(best_threshold, color=shared.FRAUD_COLOR, linestyle='--', lw=1.4,
                label=f'Optimal threshold = {best_threshold:.3f}')
axes[0].set_xlabel('Decision threshold')
axes[0].set_ylabel('Net value saved, USD (test set)')
axes[0].set_title('Business Value by Threshold')
axes[0].legend()

axes[1].plot(business_df['threshold'], business_df['precision'], color=shared.LEGIT_COLOR, label='Precision')
axes[1].plot(business_df['threshold'], business_df['recall'], color=shared.FRAUD_COLOR, label='Recall')
axes[1].axvline(best_threshold, color=shared.TEXT_COLOR, linestyle='--', lw=1.2, alpha=0.6,
                label=f'Selected threshold = {best_threshold:.3f}')
axes[1].set_xlabel('Decision threshold')
axes[1].set_ylabel('Score')
axes[1].set_title('Precision & Recall by Threshold')
axes[1].legend()

fig.tight_layout()
save_fig(fig, '11_threshold_business_value')
plt.show()
```

**Output:**
```
[SAVED] C:\Users\User\Desktop\Projects\(STAGE 4) Fraud Detection API\fraud-detection-api\reports\figures\11_threshold_business_value.png
```
```
<Figure size 1820x650 with 2 Axes>
```

### Top 5 candidate thresholds

Shown for transparency - the chosen threshold is not the only reasonable
option, and a stakeholder may prefer a nearby threshold for reasons outside
this model (e.g. review team capacity).

**Cell 10:**
```python
top5 = business_df.nlargest(5, 'net_value_usd')[
    ['threshold', 'precision', 'recall', 'tp', 'fp', 'net_value_usd']
]
print(top5.to_string(index=False))
```

**Output:**
```
 threshold  precision  recall  tp  fp  net_value_usd
    0.0933     0.5432  0.8980  88  74    10,606.4800
    0.0927     0.5399  0.8980  88  75    10,604.4800
    0.0882     0.5366  0.8980  88  76    10,602.4800
    0.0834     0.5333  0.8980  88  77    10,600.4800
    0.0809     0.5301  0.8980  88  78    10,598.4800
```

## 5. Final Evaluation at Chosen Threshold

The numbers in this section are the ones that belong in a README or a
stakeholder deck. Notice the framing avoids "accuracy" entirely - a metric
that is nearly meaningless at this class imbalance - in favor of what each
outcome actually costs or saves.

**Cell 11:**
```python
y_pred = (y_prob >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print('=' * 58)
print(f'FINAL RESULTS @ threshold = {best_threshold:.4f}')
print('=' * 58)

print('\nConfusion matrix:')
print(f'  True Positives  (fraud blocked)   : {tp:>6,}')
print(f'  False Positives (legit flagged)   : {fp:>6,}')
print(f'  False Negatives (fraud missed)    : {fn:>6,}')
print(f'  True Negatives  (legit approved)  : {tn:>6,}')

print('\nML metrics:')
print(f'  AUPRC     : {auprc:.4f}')
print(f'  Precision : {tp / (tp + fp):.4f}')
print(f'  Recall    : {tp / (tp + fn):.4f}')

value_blocked = tp * shared.AVG_FRAUD_AMOUNT
review_costs = fp * shared.REVIEW_COST
missed_losses = fn * shared.AVG_FRAUD_AMOUNT
net_value = value_blocked - review_costs

print('\nBusiness metrics (held-out test set):')
print(f'  Fraud value blocked     : ${value_blocked:>12,.2f}')
print(f'  Review costs incurred   : ${review_costs:>12,.2f}')
print(f'  Fraud value missed      : ${missed_losses:>12,.2f}')
print(f'  {"-" * 42}')
print(f'  Net value protected     : ${net_value:>12,.2f}')

print('\nClassification report:')
print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
```

**Output:**
```
==========================================================
FINAL RESULTS @ threshold = 0.0933
==========================================================

Confusion matrix:
  True Positives  (fraud blocked)   :     87
  False Positives (legit flagged)   :     74
  False Negatives (fraud missed)    :     11
  True Negatives  (legit approved)  : 56,790

ML metrics:
  AUPRC     : 0.8772
  Precision : 0.5404
  Recall    : 0.8878

Business metrics (held-out test set):
  Fraud value blocked     : $   10,632.27
  Review costs incurred   : $      148.00
  Fraud value missed      : $    1,344.31
  ------------------------------------------
  Net value protected     : $   10,484.27

Classification report:
              precision    recall  f1-score   support

       Legit       1.00      1.00      1.00     56864
       Fraud       0.54      0.89      0.67        98

    accuracy                           1.00     56962
   macro avg       0.77      0.94      0.84     56962
weighted avg       1.00      1.00      1.00     56962
```

### Annualized projection - with an explicit caveat

The dataset spans roughly two days of transaction activity. A naive
365/2 extrapolation of net value protected is a common way this kind of
result gets misrepresented - fraud is not uniformly distributed across the
year (seasonal spikes, holiday-period fraud rings, evolving attack
patterns), so this figure is reported as an order-of-magnitude indicator for
stakeholder framing, not a forecast.

**Cell 12:**
```python
observed_days = 2
annual_factor = 365 / observed_days
annual_net_estimate = net_value * annual_factor

print(f'Net value protected (test set, ~{observed_days}-day sample): ${net_value:,.2f}')
print(f'Naive annualized estimate ({annual_factor:.0f}x)          : ${annual_net_estimate:,.0f}')
print('Caveat: fraud patterns and volume are not uniform across the year -')
print('treat this as an order-of-magnitude signal, not a forecast.')
```

**Output:**
```
Net value protected (test set, ~2-day sample): $10,484.27
Naive annualized estimate (182x)          : $1,913,379
Caveat: fraud patterns and volume are not uniform across the year -
treat this as an order-of-magnitude signal, not a forecast.
```

## 6. Probability Distribution - What the Model Sees

A direct look at how cleanly the model separates the two classes in
probability space, and specifically how many fraud cases fall just below
the chosen threshold - the transactions most worth a second look if the
threshold were to shift.

**Cell 13:**
```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

legit_probs = y_prob[y_test == 0]
fraud_probs = y_prob[y_test == 1]

axes[0].hist(legit_probs, bins=60, color=shared.LEGIT_COLOR, alpha=0.7, label='Legit', density=True)
axes[0].hist(fraud_probs, bins=30, color=shared.FRAUD_COLOR, alpha=0.8, label='Fraud', density=True)
axes[0].axvline(best_threshold, color=shared.TEXT_COLOR, linestyle='--', lw=1.3,
                label=f'Threshold = {best_threshold:.3f}')
axes[0].set_xlabel('Predicted fraud probability')
axes[0].set_ylabel('Density')
axes[0].set_title('Predicted Probability by True Class')
axes[0].legend()

axes[1].hist(fraud_probs, bins=30, color=shared.FRAUD_COLOR, alpha=0.85)
axes[1].axvline(best_threshold, color=shared.TEXT_COLOR, linestyle='--', lw=1.3,
                label=f'Threshold = {best_threshold:.3f}')
axes[1].set_xlabel('Predicted fraud probability')
axes[1].set_ylabel('Count')
axes[1].set_title('Fraud Cases - Probability Distribution')
axes[1].legend()

fig.tight_layout()
save_fig(fig, '12_probability_distribution')
plt.show()

caught = int((fraud_probs >= best_threshold).sum())
missed = int((fraud_probs < best_threshold).sum())
print(f'Fraud cases above threshold (caught): {caught}')
print(f'Fraud cases below threshold (missed): {missed}')
```

**Output:**
```
[SAVED] C:\Users\User\Desktop\Projects\(STAGE 4) Fraud Detection API\fraud-detection-api\reports\figures\12_probability_distribution.png
```
```
<Figure size 1690x650 with 2 Axes>
```
```
Fraud cases above threshold (caught): 87
Fraud cases below threshold (missed): 11
```

### Near-miss inspection

Fraud cases that scored just below the threshold - the highest-value
candidates for a secondary review tier (e.g. a MEDIUM risk band routed to
human review rather than auto-approved), consistent with the risk-banding
already implemented in `src/model.py`.

**Cell 14:**
```python
near_miss_window = 0.10
near_miss_mask = (fraud_probs < best_threshold) & (fraud_probs >= best_threshold - near_miss_window)
n_near_miss = int(near_miss_mask.sum())

print(f'Fraud cases scoring within {near_miss_window:.2f} below threshold: {n_near_miss}')
if n_near_miss > 0:
    recoverable_value = n_near_miss * shared.AVG_FRAUD_AMOUNT
    print(f'Potential value recoverable via secondary review: ${recoverable_value:,.2f}')
    print('These are the transactions a MEDIUM risk band exists to catch.')
else:
    print('No near-miss fraud cases in this window.')
```

**Output:**
```
Fraud cases scoring within 0.10 below threshold: 11
Potential value recoverable via secondary review: $1,344.31
These are the transactions a MEDIUM risk band exists to catch.
```

## 7. Summary & Production Readout

Key findings, machine-readable, exported for traceability and to keep
`README.md` and `src/config.py` numbers grounded in this evaluation rather
than hand-copied and prone to drift.

**Cell 15:**
```python
import json
from datetime import datetime, timezone

summary = {
    'notebook': '03_evaluation',
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'metrics': {
        'auprc': round(float(auprc), 4),
        'auroc': round(float(auroc), 4),
        'brier_score': round(float(brier), 5),
        'random_baseline_auprc': round(float(random_baseline), 4),
    },
    'chosen_threshold': {
        'value': round(float(best_threshold), 4),
        'precision': round(float(best_row['precision']), 4),
        'recall': round(float(best_row['recall']), 4),
        'selection_method': 'max net business value (not F1 / accuracy)',
    },
    'confusion_matrix': {
        'true_positives': int(tp), 'false_positives': int(fp),
        'false_negatives': int(fn), 'true_negatives': int(tn),
    },
    'business_impact_test_set': {
        'fraud_value_blocked_usd': round(float(value_blocked), 2),
        'review_costs_usd': round(float(review_costs), 2),
        'fraud_value_missed_usd': round(float(missed_losses), 2),
        'net_value_protected_usd': round(float(net_value), 2),
        'observed_window_days': observed_days,
        'naive_annualized_estimate_usd': round(float(annual_net_estimate), 0),
        'annualization_caveat': 'Order-of-magnitude signal only - fraud volume is not uniform across the year.',
    },
    'near_miss_fraud_cases': {
        'window_below_threshold': near_miss_window,
        'count': n_near_miss,
    },
    'calibration_caveat': (
        'Ranking (AUPRC) is reliable; absolute probabilities may be imperfectly '
        'calibrated per the Brier score above. Dollar figures are directionally '
        'correct, not precise to the cent. CalibratedClassifierCV deferred as a '
        'documented next step.'
    ),
}

summary_path = shared.PROJECT_ROOT / 'reports' / 'evaluation_summary.json'
summary_path.parent.mkdir(parents=True, exist_ok=True)
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, sort_keys=True)

print(f'[SAVED] {summary_path}')
print(json.dumps(summary, indent=2, sort_keys=True))
```

**Output:**
```
[SAVED] C:\Users\User\Desktop\Projects\(STAGE 4) Fraud Detection API\fraud-detection-api\reports\evaluation_summary.json
{
  "business_impact_test_set": {
    "annualization_caveat": "Order-of-magnitude signal only - fraud volume is not uniform across the year.",
    "fraud_value_blocked_usd": 10632.27,
    "fraud_value_missed_usd": 1344.31,
    "naive_annualized_estimate_usd": 1913379.0,
    "net_value_protected_usd": 10484.27,
    "observed_window_days": 2,
    "review_costs_usd": 148.0
  },
  "calibration_caveat": "Ranking (AUPRC) is reliable; absolute probabilities may be imperfectly calibrated per the Brier score above. Dollar figures are directionally correct, not precise to the cent. CalibratedClassifierCV deferred as a documented next step.",
  "chosen_threshold": {
    "precision": 0.5432,
    "recall": 0.898,
    "selection_method": "max net business value (not F1 / accuracy)",
    "value": 0.0933
  },
  "confusion_matrix": {
    "false_negatives": 11,
    "false_positives": 74,
    "true_negatives": 56790,
    "true_positives": 87
  },
  "generated_at_utc": "2026-07-12T18:46:28.208739+00:00",
  "metrics": {
    "auprc": 0.8772,
    "auroc": 0.9815,
    "brier_score": 0.0005,
    "random_baseline_auprc": 0.0017
  },
  "near_miss_fraud_cases": {
    "count": 11,
    "window_below_threshold": 0.1
  },
  "notebook": "03_evaluation"
}
```

### Production readout

| # | Finding | Where it lands |
|---|---------|------------------|
| 1 | AUPRC and Brier score printed above | Headline metrics for README | 
| 2 | Business-optimal threshold selected above | `PREDICTION_THRESHOLD` in `src/config.py` |
| 3 | Net value protected + annualized estimate (with caveat) | Stakeholder-facing business case |
| 4 | Calibration caveat documented above | Sets expectations on dollar-figure precision |
| 5 | Near-miss fraud cases identified above | Justifies the MEDIUM risk band in `src/model.py` |

---

*Fraud Detection API - BODZZ - Abdallah A Khames -
[github.com/abdallah-bodzz/fraud-detection-api](https://github.com/abdallah-bodzz/fraud-detection-api)*
