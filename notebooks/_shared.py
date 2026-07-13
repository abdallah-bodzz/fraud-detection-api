"""
_shared.py
----------
Single source of truth for constants, data loading, and plot styling
used across 01_eda, 02_model_training, and 03_evaluation.

Every notebook imports from here instead of redefining RANDOM_STATE,
DATA_PATH, the train/test split, or business cost assumptions inline.
That's what keeps the split reproducible and the cost assumptions
consistent between the notebook that reports results and the script
(train_model.py) that ships the model.

Usage:
    import _shared as shared

    shared.set_plot_theme()
    df = shared.load_data()
    X_train, X_test, y_train, y_test, scaler = shared.load_split()

--------------------------------------------------------------------
Project   : Fraud Detection API
Lead Dev  : Abdallah A Khames
Org       : BODZZ
GitHub    : github.com/abdallah-bodzz
Repo      : fraud-detection-api
--------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_STATE: Final[int] = 42

# ── Business cost assumptions ───────────────────────────────────────────────
# Kept identical to src/config.py and train_model.py by convention — if these
# drift, the notebook's "net value saved" figure stops matching production.
AVG_FRAUD_AMOUNT: Final[float] = 122.21  # mean Amount where Class == 1
REVIEW_COST: Final[float] = 2.00  # est. manual review cost per flagged tx

# ── Paths ────────────────────────────────────────────────────────────────────
# Notebooks run from notebooks/, one level below the project root.
NOTEBOOK_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = NOTEBOOK_DIR.parent

DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "creditcard.csv"
MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "fraud_model.joblib"
SCALER_PATH: Final[Path] = PROJECT_ROOT / "models" / "scaler.joblib"
FIGURES_DIR: Final[Path] = PROJECT_ROOT / "reports" / "figures"

# ── Feature schema ───────────────────────────────────────────────────────────
# Order matters: it must match the column order the model was trained on.
PCA_FEATURE_COLS: Final[list[str]] = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS: Final[list[str]] = ["Time", *PCA_FEATURE_COLS, "Amount"]
SCALED_COLS: Final[list[str]] = ["Time", "Amount"]  # V1-V28 are already PCA-scaled
TARGET_COL: Final[str] = "Class"

# ── Visual identity ──────────────────────────────────────────────────────────
# Muted, print-safe palette shared by every chart in the analysis. Chosen over
# seaborn/matplotlib defaults so figures read as a designed report, not
# ad-hoc notebook output — and reused as-is in any exported report deck.
FRAUD_COLOR: Final[str] = "#C1440E"  # burnt orange — signals risk, not alarm-red
LEGIT_COLOR: Final[str] = "#2E5266"  # muted navy
NEUTRAL_COLOR: Final[str] = "#6B6B63"  # warm gray, for baselines/reference lines
BACKGROUND_COLOR: Final[str] = "#FAFAF8"
GRID_COLOR: Final[str] = "#E0E0DC"
TEXT_COLOR: Final[str] = "#2B2B2B"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw transaction dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset isn't present at `path`. Raised explicitly (not
        swallowed) with the Kaggle source, so a missing-data failure is
        immediately actionable instead of surfacing as a downstream
        KeyError three cells later.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place creditcard.csv in the data/ folder."
        )
    return pd.read_csv(path)


def load_split(
    path: Path = DATA_PATH,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Load the dataset and return a reproducible, leakage-safe train/test split.

    The scaler is fit on the training set only (Time and Amount), then applied
    to both splits — the same procedure train_model.py uses in production, so
    notebook metrics and the deployed model stay comparable.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
        Scaler is returned so downstream cells (e.g. loading a saved model for
        evaluation) can reuse it instead of re-fitting — re-fitting on a
        differently-ordered split would silently shift the scaling and make
        results non-reproducible.
    """
    df = load_data(path)
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train.loc[:, SCALED_COLS] = scaler.fit_transform(X_train[SCALED_COLS])
    X_test.loc[:, SCALED_COLS] = scaler.transform(X_test[SCALED_COLS])

    return X_train, X_test, y_train, y_test, scaler


def set_plot_theme() -> None:
    """
    Apply the shared visual theme to matplotlib's rcParams.

    Call once near the top of each notebook (after the imports cell) so every
    chart in the analysis — across all three notebooks — shares one visual
    identity instead of falling back to library defaults.
    """
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND_COLOR,
            "figure.dpi": 130,
            "axes.facecolor": BACKGROUND_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.6,
            "grid.linewidth": 0.7,
            "font.family": "sans-serif",
            "font.size": 10.5,
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "savefig.facecolor": BACKGROUND_COLOR,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> Path:
    """
    Save a figure to reports/figures/<name>.png, creating the directory
    on first use.

    Centralizing this (rather than each notebook cell writing its own
    `plt.savefig(...)` with a hand-typed path) keeps figure filenames
    consistent and makes the output directory predictable for anything
    downstream that packages these into a report.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(out_path)
    return out_path
