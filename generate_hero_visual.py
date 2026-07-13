# -*- coding: utf-8 -*-
"""
generate_hero_visual.py
-------------------------
Generates the project's hero visual: a wide banner summarizing the fraud
detection model's methodology and business impact. Used as the README
header, repo social preview, and portfolio thumbnail.

Layout: a fixed-width dark identity sidebar (left) carries branding,
headline, and headline stats. A light content column (right) stacks four
evidence panels: model specification tags, precision/recall-vs-threshold
sweep with the chosen operating point marked, top feature importances,
and the business-impact / outcome breakdown. Panels are stacked with
explicit margins rather than packed into a dense grid, which removes the
coordinate-overlap risk of many axes sharing one figure.

The point of this version over earlier ones: a stat card alone reads as
a claim. A threshold sweep, a feature-importance ranking, and a model
spec line read as a modeling artifact — the kind of detail that only
exists if the work was actually done.

Two things this image deliberately does NOT claim to be, disclosed
in-image rather than only in this docstring:
  - The threshold sweep curve is a representative shape anchored to the
    real, measured operating point — `evaluation_summary.json` stores the
    chosen point, not the full precision/recall-vs-threshold arrays, so
    the curve between points is illustrative, not a re-plot of notebook
    output. See `_synthetic_threshold_sweep`.
  - The feature-importance bars encode rank order only (which the summary
    JSON does store, correctly), not the real gain magnitudes (which it
    doesn't) — bar lengths are a fixed decreasing sequence, not measured
    values.
Both are labelled on the figure itself, not just in source comments —
a viewer of the PNG alone should be able to tell what's measured and
what's illustrative.

Run once from the project root, after 02_model_training.ipynb and
03_evaluation.ipynb have produced their summary JSON files:

    python generate_hero_visual.py
    python generate_hero_visual.py --output reports/figures/preview.png --dpi 120
    python generate_hero_visual.py --strict   # exit non-zero if summaries are missing

Falls back to fixed, notebook-accurate figures if the summary files
aren't found, so this can still run standalone before the notebooks have
been executed — use `--strict` in CI/release builds where shipping a
fallback-generated banner by accident would be worse than failing loudly.

--------------------------------------------------------------------
Project   : Fraud Detection API
Lead Dev  : Abdallah A Khames
Org       : BODZZ
GitHub    : github.com/abdallah-bodzz
Repo      : fraud-detection-api
--------------------------------------------------------------------
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

# Headless-safe: this script runs in CI, in Docker image builds, and on
# dev machines without a display server. Agg must be selected before
# `pyplot` is imported anywhere, or the default backend probes for a GUI
# toolkit and fails on a display-less runner.
matplotlib.use('Agg')

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent / 'notebooks'))
import _shared as shared  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / 'reports' / 'figures' / 'hero_visual.png'
EVAL_SUMMARY_PATH = PROJECT_ROOT / 'reports' / 'evaluation_summary.json'
TRAINING_SUMMARY_PATH = PROJECT_ROOT / 'reports' / 'model_training_summary.json'

# Fallback figures mirror the actual values in reports/evaluation_summary.json
# and reports/model_training_summary.json as of the 2026-07-12 evaluation run
# (see 03_evaluation.ipynb / 02_model_training.ipynb). Keeping these in sync
# with the real reports — instead of an earlier, since-superseded run — is
# the whole point of a fallback: a standalone preview should look like the
# real thing, not contradict it the moment the real JSON is regenerated.
EVAL_FALLBACK = {
    'metrics': {'auprc': 0.8772, 'auroc': 0.9815, 'brier_score': 0.0005},
    'chosen_threshold': {'value': 0.0933, 'precision': 0.5404, 'recall': 0.8878},
    'business_impact_test_set': {
        'net_value_protected_usd': 10484.27,
        'review_costs_usd': 148.00,
        'fraud_value_blocked_usd': 10632.27,
        'naive_annualized_estimate_usd': 1913379.0,
        'observed_window_days': 2,
    },
    'confusion_matrix': {'true_positives': 87, 'false_positives': 74, 'false_negatives': 11},
    'calibration_caveat': (
        'Ranking (AUPRC) is reliable; absolute probabilities may be imperfectly '
        'calibrated per the Brier score above. Dollar figures are directionally '
        'correct, not precise to the cent.'
    ),
}
TRAINING_FALLBACK = {
    'class_balance_strategy': {'scale_pos_weight': 577.29},
    'cross_validation': {'folds': 5, 'mean_auprc': 0.851, 'std_auprc': 0.0242},
    # Model-learned ranking (final_model.feature_importances_), not the EDA
    # separation ranking — the two agree on most features but aren't the
    # same list, and this panel claims to show the former.
    'top_5_features_by_importance': ['V14', 'V10', 'V4', 'V12', 'V17'],
}

INK = '#1A1A18'
SIDEBAR_BG = '#12211E'
SIDEBAR_TEXT = '#E9E7DE'
SIDEBAR_MUTED = '#8FA39D'
ACCENT = '#3F7D6E'
DANGER = '#C1440E'
CANVAS_BG = '#FAFAF8'
RULE = '#E3E1D8'
LABEL_GRAY = '#8A8879'


class SummaryLoadError(RuntimeError):
    """Raised when a summary JSON is present but missing an expected key.

    Deliberately distinct from a missing file (which falls back silently,
    or loudly under --strict) — a malformed/renamed key means the notebook
    schema drifted out from under this script, which is a bug to fix, not
    a fallback to paper over.
    """


def load_json(path: Path, fallback: dict, name: str, strict: bool) -> dict:
    if path.exists():
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    message = f'{path} not found - using {name} fallback figures.'
    if strict:
        raise FileNotFoundError(
            f'{message} Refusing to continue under --strict: run the notebooks '
            f'first, or drop --strict for a fallback-data preview.'
        )
    print(f'[WARN] {message}')
    return fallback


def _get(d: dict, *keys, context: str):
    """Nested dict lookup with a message that names the exact missing key,
    instead of a bare KeyError three stack frames from anything useful."""
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            path = ' -> '.join(keys)
            raise SummaryLoadError(
                f"Expected key '{path}' in {context} summary JSON, but '{key}' "
                f"is missing. The notebook's summary schema may have changed — "
                f"update this script's key access to match."
            )
        cur = cur[key]
    return cur


def _synthetic_pr_curve(auprc: float):
    recall = np.linspace(0.001, 1, 200)
    precision = auprc * (1 - recall) ** 0.4 + (1 - auprc) * 0.02
    return recall, np.clip(precision, 0, 1)


def _synthetic_threshold_sweep(chosen_threshold: float, chosen_precision: float, chosen_recall: float):
    """
    Representative precision/recall-vs-threshold curves shaped to pass
    through the reported operating point. A visual proxy for the real
    sweep computed in 03_evaluation.ipynb, not a re-derivation of it —
    the summary JSON stores the chosen point, not the full arrays.
    """
    t = np.linspace(0.01, 0.99, 200)
    recall = chosen_recall + (1 - chosen_recall) * np.exp(-6 * t) - (1 - chosen_recall) * np.exp(-6 * chosen_threshold)
    recall = np.clip(recall, 0.03, 0.999)
    precision = chosen_precision * (t / chosen_threshold) ** 0.35
    precision = np.clip(precision, 0.02, 0.999)
    return t, precision, recall


def build_hero(eval_summary: dict, training_summary: dict, generated_from_live_data: bool) -> plt.Figure:
    shared.set_plot_theme()

    auprc = _get(eval_summary, 'metrics', 'auprc', context='evaluation')
    brier = eval_summary.get('metrics', {}).get('brier_score')
    threshold = _get(eval_summary, 'chosen_threshold', 'value', context='evaluation')
    precision = _get(eval_summary, 'chosen_threshold', 'precision', context='evaluation')
    recall = _get(eval_summary, 'chosen_threshold', 'recall', context='evaluation')
    biz = _get(eval_summary, 'business_impact_test_set', context='evaluation')
    net_value = _get(biz, 'net_value_protected_usd', context='evaluation.business_impact_test_set')
    review_cost = _get(biz, 'review_costs_usd', context='evaluation.business_impact_test_set')
    fraud_value_blocked = biz.get('fraud_value_blocked_usd', net_value + review_cost)
    annualized = biz.get('naive_annualized_estimate_usd')
    observed_days = biz.get('observed_window_days')
    calibration_caveat = eval_summary.get('calibration_caveat')
    cm = eval_summary.get('confusion_matrix', EVAL_FALLBACK['confusion_matrix'])
    tp, fp, fn = cm['true_positives'], cm['false_positives'], cm['false_negatives']

    scale_pos_weight = _get(
        training_summary, 'class_balance_strategy', 'scale_pos_weight', context='training'
    )
    cv_mean = _get(training_summary, 'cross_validation', 'mean_auprc', context='training')
    cv_std = _get(training_summary, 'cross_validation', 'std_auprc', context='training')
    cv_folds = _get(training_summary, 'cross_validation', 'folds', context='training')
    top_features = _get(training_summary, 'top_5_features_by_importance', context='training')

    fig = plt.figure(figsize=(16, 10.5), facecolor=CANVAS_BG)

    # ── Sidebar panel ────────────────────────────────────────────────────
    SIDEBAR_W = 0.335
    sidebar_bg_ax = fig.add_axes((0, 0, SIDEBAR_W, 1))
    sidebar_bg_ax.axis('off')
    sidebar_bg_ax.add_patch(mpatches.Rectangle(
        (0, 0), 1, 1, transform=sidebar_bg_ax.transAxes, facecolor=SIDEBAR_BG, edgecolor='none',
    ))

    pad = 0.048
    inner_w = SIDEBAR_W - 2 * pad

    mark_ax = fig.add_axes((pad, 0.925, inner_w, 0.035))
    mark_ax.axis('off')
    mark_ax.text(0, 0.5, 'FRAUD DETECTION API', fontsize=11.5, fontweight='bold',
                 color=SIDEBAR_TEXT, va='center', ha='left', family='monospace',
                 transform=mark_ax.transAxes)

    head_ax = fig.add_axes((pad, 0.735, inner_w, 0.175))
    head_ax.axis('off')
    head_ax.text(0, 1.0, 'Fraud caught by', fontsize=24, fontweight='bold',
                 color=SIDEBAR_TEXT, va='top', ha='left', transform=head_ax.transAxes)
    head_ax.text(0, 0.68, 'economics,', fontsize=24, fontweight='bold',
                 color=SIDEBAR_TEXT, va='top', ha='left', transform=head_ax.transAxes)
    head_ax.text(0, 0.36, 'not accuracy.', fontsize=24, fontweight='bold',
                 color='#7FBFA8', va='top', ha='left', transform=head_ax.transAxes)
    head_ax.text(
        0, 0.06,
        'XGBoost, threshold tuned for net dollar\nvalue. Served via FastAPI.',
        fontsize=10.5, color=SIDEBAR_MUTED, va='top', ha='left',
        transform=head_ax.transAxes, linespacing=1.6,
    )

    fig.add_artist(plt.Line2D(
        [pad, SIDEBAR_W - pad], [0.715, 0.715], color='#2B3F3A',
        linewidth=1.1, transform=fig.transFigure,
    ))

    # Model spec tag row — the ML-specific detail block
    spec_ax = fig.add_axes((pad, 0.655, inner_w, 0.05))
    spec_ax.axis('off')
    spec_ax.text(0, 1.0, 'MODEL SPEC', fontsize=9, fontweight='bold', color='#6F8880',
                  va='top', ha='left', family='monospace', transform=spec_ax.transAxes)
    spec_lines = [
        'algorithm       XGBoost (gradient-boosted trees)',
        f'imbalance       scale_pos_weight = {scale_pos_weight:.1f}',
        f'validation      {cv_folds}-fold stratified CV, AUPRC {cv_mean:.3f} +/- {cv_std:.3f}',
        f'calibration     Brier {brier:.4f}' if brier else 'calibration     see 03_evaluation.ipynb',
    ]
    for i, line in enumerate(spec_lines):
        spec_ax.text(0, 0.55 - i * 0.30, line, fontsize=8.7, color=SIDEBAR_MUTED,
                      va='top', ha='left', family='monospace', transform=spec_ax.transAxes)

    fig.add_artist(plt.Line2D(
        [pad, SIDEBAR_W - pad], [0.485, 0.485], color='#2B3F3A',
        linewidth=1.1, transform=fig.transFigure,
    ))

    # Stat rail
    net_value_sub = f'{observed_days}-day test window' if observed_days else 'held-out test set'
    if annualized:
        net_value_sub += f' · ~${annualized / 1e6:.1f}M/yr naive est.'
    rail_items = [
        ('AUPRC', f'{auprc:.2f}', 'area under precision-recall curve'),
        ('NET VALUE PROTECTED', f'${net_value:,.0f}', net_value_sub),
    ]
    row_h = 0.155
    top = 0.44
    for i, (label, value, sub) in enumerate(rail_items):
        y = top - i * row_h
        row_ax = fig.add_axes((pad, y - row_h + 0.035, inner_w, row_h - 0.02))
        row_ax.axis('off')
        row_ax.text(0, 0.86, label, fontsize=9, fontweight='bold', color='#6F8880',
                     va='top', ha='left', family='monospace', transform=row_ax.transAxes)
        row_ax.text(0, 0.50, value, fontsize=22, fontweight='bold', color=SIDEBAR_TEXT,
                     va='top', ha='left', transform=row_ax.transAxes)
        row_ax.text(0, 0.10, sub, fontsize=8.7, color=SIDEBAR_MUTED,
                     va='top', ha='left', transform=row_ax.transAxes)
        if i < len(rail_items) - 1:
            fig.add_artist(plt.Line2D(
                [pad, SIDEBAR_W - pad], [y - row_h + 0.02, y - row_h + 0.02],
                color='#233631', linewidth=0.9, transform=fig.transFigure,
            ))

    credit_ax = fig.add_axes((pad, 0.025, inner_w, 0.06))
    credit_ax.axis('off')
    credit_ax.text(0, 1.0, 'BODZZ', fontsize=9.5, fontweight='bold', color=SIDEBAR_TEXT,
                    va='top', ha='left', family='monospace', transform=credit_ax.transAxes)
    credit_ax.text(0, 0.55, 'Lead Dev: Abdallah A Khames', fontsize=9, color=SIDEBAR_MUTED,
                    va='top', ha='left', transform=credit_ax.transAxes)
    credit_ax.text(0, 0.15, 'github.com/abdallah-bodzz/fraud-detection-api', fontsize=8.5,
                    color='#6F8880', va='top', ha='left', family='monospace',
                    transform=credit_ax.transAxes)

    # ── Content column ───────────────────────────────────────────────────
    CONTENT_X0 = SIDEBAR_W + 0.04
    CONTENT_W = 1 - CONTENT_X0 - 0.04

    def section_header(y, text):
        ax = fig.add_axes((CONTENT_X0, y, CONTENT_W, 0.032))
        ax.axis('off')
        ax.text(0, 0.5, text, fontsize=9.5, fontweight='bold', color=LABEL_GRAY,
                 va='center', ha='left', family='monospace', transform=ax.transAxes)
        ax.axhline(0.02, color=RULE, linewidth=1.1, xmin=0, xmax=1)

    def caption(y, text):
        """Small, honest disclosure text placed directly on the figure —
        so a viewer of the exported PNG alone (not just this source file)
        can tell what's measured versus illustrative."""
        ax = fig.add_axes((CONTENT_X0, y, CONTENT_W, 0.026))
        ax.axis('off')
        ax.text(0, 0.5, text, fontsize=7.4, color=LABEL_GRAY, style='italic',
                 va='center', ha='left', transform=ax.transAxes)

    # Panel 1: threshold sweep (precision & recall vs threshold)
    section_header(0.900, 'THRESHOLD SWEEP - BUSINESS-OPTIMAL OPERATING POINT')
    sweep_ax = fig.add_axes((CONTENT_X0, 0.735, CONTENT_W * 0.56, 0.145))
    t, p_curve, r_curve = _synthetic_threshold_sweep(threshold, precision, recall)
    sweep_ax.plot(t, p_curve, color=ACCENT, linewidth=2.0, label='precision')
    sweep_ax.plot(t, r_curve, color=DANGER, linewidth=2.0, linestyle='--', label='recall')
    sweep_ax.axvline(threshold, color=INK, linewidth=1.0, linestyle=':', alpha=0.6)
    sweep_ax.plot(threshold, precision, marker='o', markersize=6, color=ACCENT,
                  markeredgecolor='white', markeredgewidth=1.2, zorder=5)
    sweep_ax.plot(threshold, recall, marker='o', markersize=6, color=DANGER,
                  markeredgecolor='white', markeredgewidth=1.2, zorder=5)
    sweep_ax.set_xlim(0, 1)
    sweep_ax.set_ylim(0, 1.05)
    sweep_ax.set_xlabel('decision threshold', fontsize=9, color=LABEL_GRAY)
    sweep_ax.set_xticks([0, threshold, 1.0])
    sweep_ax.set_xticklabels(['0.0', f'{threshold:.2f}', '1.0'], fontsize=8.5)
    sweep_ax.set_yticks([0, 0.5, 1.0])
    sweep_ax.tick_params(labelsize=8.5, colors=LABEL_GRAY)
    for spine in ('top', 'right'):
        sweep_ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        sweep_ax.spines[spine].set_color(RULE)
    sweep_ax.legend(
        ['precision', 'recall'],
        loc='lower right',
        fontsize=8.5,
        frameon=True,
        facecolor='white',
        edgecolor=RULE,
        framealpha=0.95,
    )

    # Panel 1b: feature importance (right of threshold sweep)
    feat_ax = fig.add_axes((CONTENT_X0 + CONTENT_W * 0.62, 0.708, CONTENT_W * 0.38, 0.172))
    feat_labels = list(reversed(top_features))
    feat_values = list(reversed(np.linspace(1.0, 0.42, len(top_features))))
    feat_ax.barh(feat_labels, feat_values, color=ACCENT, height=0.55, alpha=0.85)
    feat_ax.set_xlim(0, 1.12)
    feat_ax.set_xticks([])
    feat_ax.tick_params(axis='y', labelsize=10, colors=INK, length=0)
    for spine in feat_ax.spines.values():
        spine.set_visible(False)
    feat_ax.text(0, 1.14, 'top features by importance (rank order)', fontsize=8.0, color=LABEL_GRAY, va='bottom', ha='left', transform=feat_ax.transAxes)

    caption(
        0.66,
        'Sweep curve is illustrative, anchored to the measured operating point below · '
        'feature bars show rank order, not exact gain magnitude.',
    )

    # Panel 2: business impact
    section_header(0.615, 'BUSINESS IMPACT - HELD-OUT TEST SET')
    biz_ax = fig.add_axes((CONTENT_X0, 0.415, CONTENT_W, 0.175))
    categories = ['Fraud value blocked', 'Review cost incurred', 'Net value protected']
    values = [fraud_value_blocked, review_cost, net_value]
    colors = [ACCENT, DANGER, '#6B6B63']
    bars = biz_ax.barh(categories, values, color=colors, height=0.5, zorder=3)
    for bar, val in zip(bars, values):
        biz_ax.text(
            val + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
            f'${val:,.0f}', va='center', ha='left', fontsize=11, fontweight='bold', color=INK,
        )
    biz_ax.set_xlim(0, max(values) * 1.25)
    biz_ax.set_xticks([])
    biz_ax.invert_yaxis()
    biz_ax.tick_params(axis='y', labelsize=10, colors=INK, length=0)
    for spine in biz_ax.spines.values():
        spine.set_visible(False)

    if calibration_caveat:
        short_caveat = (
            'Dollar figures are directionally correct, not precise to the cent — '
            'ranking is reliable, calibration is a documented next step.'
        )
        caption(0.365, short_caveat)

    # Panel 3: outcome breakdown
    section_header(0.335, 'OUTCOME BREAKDOWN - CONFUSION MATRIX AT OPERATING POINT')
    cm_ax = fig.add_axes((CONTENT_X0, 0.135, CONTENT_W, 0.155))
    cm_ax.axis('off')
    cm_items = [
        ('Fraud blocked (TP)', tp, ACCENT),
        ('Legit flagged (FP)', fp, DANGER),
        ('Fraud missed (FN)', fn, '#6B6B63'),
    ]
    seg_w = 1.0 / len(cm_items)
    for i, (label, count, color) in enumerate(cm_items):
        x0 = i * seg_w
        cm_ax.add_patch(mpatches.Rectangle(
            (x0 + 0.015, 0.0), seg_w - 0.03, 0.20,
            transform=cm_ax.transAxes, facecolor=color, edgecolor='none', alpha=0.88,
        ))
        cm_ax.text(x0 + seg_w / 2, 0.92, f'{count:,}', fontsize=18, fontweight='bold',
                    color=INK, ha='center', va='top', transform=cm_ax.transAxes)
        cm_ax.text(x0 + seg_w / 2, 0.40, label, fontsize=10, color='#6B6B63',
                    ha='center', va='top', transform=cm_ax.transAxes)

    # Footer strip — tech stack + provenance
    footer_ax = fig.add_axes((CONTENT_X0, 0.03, CONTENT_W, 0.03))
    footer_ax.axis('off')
    footer_ax.axhline(1.0, color=RULE, linewidth=1.0, xmin=0, xmax=1)
    footer_ax.text(
        0, 0.0, 'XGBOOST   SCIKIT-LEARN   FASTAPI   DOCKER   PANDAS',
        fontsize=8.5, color=LABEL_GRAY, va='bottom', ha='left',
        family='monospace', transform=footer_ax.transAxes,
    )
    generated_tag = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    source_tag = 'live evaluation data' if generated_from_live_data else 'fallback data — run notebooks first'
    footer_ax.text(1.0, 0.0, f'v1.0 · GENERATED {generated_tag} · {source_tag.upper()}', fontsize=7.5,
        color=LABEL_GRAY, va='bottom', ha='right', family='monospace',
        transform=footer_ax.transAxes,
    )

    return fig


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the README hero visual from the project's evaluation summaries."
    )
    parser.add_argument(
        '--output', type=Path, default=DEFAULT_OUTPUT_PATH,
        help=f'Output PNG path (default: {DEFAULT_OUTPUT_PATH.relative_to(PROJECT_ROOT)})',
    )
    parser.add_argument(
        '--dpi', type=int, default=180,
        help='Output resolution in dots per inch (default: 180).',
    )
    parser.add_argument(
        '--strict', action='store_true',
        help='Exit with an error instead of falling back to placeholder figures '
             'when a summary JSON is missing. Recommended for CI/release builds.',
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    eval_summary = load_json(EVAL_SUMMARY_PATH, EVAL_FALLBACK, 'evaluation', args.strict)
    training_summary = load_json(TRAINING_SUMMARY_PATH, TRAINING_FALLBACK, 'training', args.strict)
    used_live_data = EVAL_SUMMARY_PATH.exists() and TRAINING_SUMMARY_PATH.exists()

    fig = build_hero(eval_summary, training_summary, generated_from_live_data=used_live_data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor=CANVAS_BG)
    plt.close(fig)

    print(f'[SAVED] {args.output}')
    if not used_live_data:
        print('[NOTE] Generated from fallback data, not the live notebook summaries.')


if __name__ == '__main__':
    main()