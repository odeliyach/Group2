"""
Milestone 3 — Step 7, Module 2/5: Preprocessing & Normalization
================================================================
Dataset-agnostic. Vector-group key matches M2's unified_validation.py
exactly (rounds floats to 4dp, +0.0 to normalize -0.0, fillna('NA') for
the rest). Fold-safe scaling/class-weight helpers -- callers must fit on
the training fold only.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


def vector_group_key(df, feat_cols):
    parts = []
    for c in feat_cols:
        col = df[c]
        if col.dtype.kind in "fc":
            parts.append((col.round(4) + 0.0).astype(str))
        else:
            parts.append(col.fillna("NA").astype(str))
    return parts[0].str.cat(parts[1:], sep="|").values


def impute(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X, nan=0.0)


def log1p_continuous(X: np.ndarray, feat_cols: list, binary_features: set) -> np.ndarray:
    """log1p-transforms every column NOT in binary_features. Safe to call
    on already-non-negative count/rate/delta features (all of this
    project's continuous features are >=0 by construction). Monotonic,
    so this is a no-op for RF/XGBoost's split decisions -- see config.py's
    LOG1P_FOR_ISOFOREST/LOG1P_FOR_CNN docstring for why it matters for
    Isolation Forest and (to a lesser extent) CNN specifically."""
    X = X.copy()
    for i, col in enumerate(feat_cols):
        if col not in binary_features:
            X[:, i] = np.log1p(np.clip(X[:, i], 0, None))
    return X


def fit_scaler_on_train(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def fold_class_weights(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def xgb_scale_pos_weight(y_train: np.ndarray) -> float:
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    return (n_neg / n_pos) if n_pos > 0 else 1.0
