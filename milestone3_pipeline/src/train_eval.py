"""
Milestone 3 — Step 7, Module 5/5: Training + Evaluation
==============================================================
Dataset-agnostic. Generalizes M2's unified_validation.py `repeated_cv`
(RF-only, pooled-per-repeat StratifiedGroupKFold) to all 4 required
models, including the two M2 didn't originally cover:
  - XGBoost:        needs a fold-local scale_pos_weight (imbalance).
  - 1D-CNN:         needs a fold-fit StandardScaler + fold-local Keras
                     class_weight dict, both computed on the TRAIN fold only.
  - Isolation Forest: unsupervised -- fit ignores y, but is still scored
                     against y at evaluation time.

F1-OPTIMAL THRESHOLDING (the headline metric): every model below picks its
decision threshold by maximizing F1 on a held-out slice of the TRAINING
fold only (never the test fold) -- not the default 0.5 cutoff. This alone
raised CAM-LDS Isolation Forest's recall from 0.17 (contamination='auto'
blindly assuming ~10% outliers on a 54%-attack dataset) without touching
the model itself; AUROC (ranking quality) is unaffected since it's
threshold-independent, but F1/Precision/Recall move.

IMPORTANT DISCLOSURE FOR YOUR WRITE-UP: for Isolation Forest specifically,
this means TRAIN-FOLD LABELS are used to calibrate the decision threshold
(never to fit the trees -- fit() still never sees y). This is a legitimate,
commonly-used hybrid ("semi-supervised threshold calibration on an
unsupervised detector"), not a silent leakage bug, but it changes the
model's operating assumption from purely unsupervised and should be
stated explicitly in Ch6/Ch8 rather than left implicit.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix)

from preprocessing import impute, fit_scaler_on_train, fold_class_weights, xgb_scale_pos_weight
from models import (build_rf, build_xgb, build_cnn, cnn_fit_kwargs,
                     build_mlp, mlp_fit_kwargs, build_iforest)
from config import CV_CONFIG, MAX_FPR


def _fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def _score_fold(y_true, y_pred, y_proba):
    out = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": _fpr(y_true, y_pred),
    }
    out["auroc"] = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else np.nan
    return out


def _best_f1_threshold(y_true, proba, default=0.5, max_fpr=None):
    """Sweep every threshold implied by the sorted probabilities and return
    the one maximizing F1 -- optionally CONSTRAINED to keep FPR under
    max_fpr (an operational deployment cap, see config.MAX_FPR).

    Without a cap, this can converge to a degenerate "flag almost
    everything" threshold whenever the model can't actually discriminate
    the classes and the positive class is a large fraction of the data --
    F1 looks non-trivial (it inflates toward 2*prevalence/(1+prevalence),
    the score of an always-positive rule) while FPR silently approaches 1.
    The cap exists specifically to stop that: if no threshold satisfies
    max_fpr, we fall back to the lowest-FPR threshold available and let
    the resulting (probably poor) F1 honestly reflect that the model
    can't hit the FPR budget -- rather than silently ignoring the budget.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    if len(np.unique(y_true)) < 2:
        return default

    order = np.argsort(-proba)
    y_sorted = y_true[order]
    proba_sorted = proba[order]

    n_pos = int((y_sorted == 1).sum())
    n_neg = int((y_sorted == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return default

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    recall = tp_cum / n_pos
    fpr = fp_cum / n_neg
    precision = tp_cum / (tp_cum + fp_cum)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    if max_fpr is not None:
        eligible = np.where(fpr <= max_fpr)[0]
        if len(eligible) > 0:
            best = eligible[np.argmax(f1[eligible])]
        else:
            best = int(np.argmin(fpr))  # cap unreachable -- pick closest
    else:
        best = int(np.argmax(f1))

    return float(proba_sorted[best])


def _fit_predict_rf(X_tr, y_tr, groups_tr, X_te, hp_overrides, max_fpr):
    # Inner group-aware split of the TRAINING fold only, purely to pick an
    # F1-optimal threshold -- X_te/y_te are never touched here.
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=0)
    fit_idx, val_idx = next(gss.split(X_tr, y_tr, groups_tr))

    thresh_clf = build_rf(hp_overrides)
    thresh_clf.fit(X_tr[fit_idx], y_tr[fit_idx])
    val_proba = thresh_clf.predict_proba(X_tr[val_idx])[:, 1]
    threshold = _best_f1_threshold(y_tr[val_idx], val_proba, max_fpr=max_fpr)

    clf = build_rf(hp_overrides)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    return (proba >= threshold).astype(int), proba


def _fit_predict_xgb(X_tr, y_tr, groups_tr, X_te, hp_overrides, max_fpr):
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=0)
    fit_idx, val_idx = next(gss.split(X_tr, y_tr, groups_tr))

    spw_inner = xgb_scale_pos_weight(y_tr[fit_idx])
    thresh_clf = build_xgb(scale_pos_weight=spw_inner, overrides=hp_overrides)
    thresh_clf.fit(X_tr[fit_idx], y_tr[fit_idx])
    val_proba = thresh_clf.predict_proba(X_tr[val_idx])[:, 1]
    threshold = _best_f1_threshold(y_tr[val_idx], val_proba, max_fpr=max_fpr)

    spw = xgb_scale_pos_weight(y_tr)
    clf = build_xgb(scale_pos_weight=spw, overrides=hp_overrides)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    return (proba >= threshold).astype(int), proba


def _fit_predict_cnn(X_tr, y_tr, groups_tr, X_te, hp_overrides, max_fpr):
    scaler = fit_scaler_on_train(X_tr)          # fold-safe: fit on train only
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)[..., None]

    # Explicit group-aware validation slice (replaces Keras' own
    # validation_split, which doesn't respect groups) -- reused for BOTH
    # early stopping AND F1-threshold selection, so no extra training cost.
    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=0)
    fit_idx, val_idx = next(gss.split(X_tr_s, y_tr, groups_tr))
    cw = fold_class_weights(y_tr[fit_idx])      # fold-safe: computed on the fit slice only

    clf = build_cnn(n_features=X_tr.shape[1], overrides=hp_overrides)
    fit_kwargs = cnn_fit_kwargs(hp_overrides)
    clf.fit(X_tr_s[fit_idx][..., None], y_tr[fit_idx], class_weight=cw,
            validation_data=(X_tr_s[val_idx][..., None], y_tr[val_idx]), **fit_kwargs)

    val_proba = clf.predict(X_tr_s[val_idx][..., None], verbose=0).ravel()
    threshold = _best_f1_threshold(y_tr[val_idx], val_proba, max_fpr=max_fpr)

    proba = clf.predict(X_te_s, verbose=0).ravel()
    return (proba >= threshold).astype(int), proba


def _fit_predict_mlp(X_tr, y_tr, groups_tr, X_te, hp_overrides, max_fpr):
    # Mirrors _fit_predict_cnn exactly (same fold-safe scaler, same inner
    # group-aware validation split for threshold calibration and early
    # stopping) -- the ONLY difference from the CNN fitter is no [..., None]
    # channel-dimension reshape, since build_mlp takes a plain 2D input.
    scaler = fit_scaler_on_train(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=0)
    fit_idx, val_idx = next(gss.split(X_tr_s, y_tr, groups_tr))
    cw = fold_class_weights(y_tr[fit_idx])

    clf = build_mlp(n_features=X_tr.shape[1], overrides=hp_overrides)
    fit_kwargs = mlp_fit_kwargs(hp_overrides)
    clf.fit(X_tr_s[fit_idx], y_tr[fit_idx], class_weight=cw,
            validation_data=(X_tr_s[val_idx], y_tr[val_idx]), **fit_kwargs)

    val_proba = clf.predict(X_tr_s[val_idx], verbose=0).ravel()
    threshold = _best_f1_threshold(y_tr[val_idx], val_proba, max_fpr=max_fpr)

    proba = clf.predict(X_te_s, verbose=0).ravel()
    return (proba >= threshold).astype(int), proba


def _fit_predict_iforest(X_tr, y_tr, groups_tr, X_te, hp_overrides, max_fpr):
    # Unsupervised fit -- y_tr never reaches .fit(). y_tr IS used below,
    # AFTER fitting, purely to calibrate the anomaly-score threshold (see
    # module docstring's disclosure note -- state this explicitly in Ch8).
    clf = build_iforest(hp_overrides)
    clf.fit(X_tr)

    train_score = -clf.score_samples(X_tr)   # higher = more anomalous
    test_score = -clf.score_samples(X_te)
    smin, smax = train_score.min(), train_score.max()
    train_proba = (train_score - smin) / (smax - smin + 1e-12)
    test_proba = (test_score - smin) / (smax - smin + 1e-12)

    threshold = _best_f1_threshold(y_tr, train_proba, max_fpr=max_fpr)  # in-sample, label-calibrated
    return (test_proba >= threshold).astype(int), test_proba


_FITTERS = {
    "rf": _fit_predict_rf,
    "xgb": _fit_predict_xgb,
    "cnn": _fit_predict_cnn,
    "mlp": _fit_predict_mlp,
    "iforest": _fit_predict_iforest,
}


def repeated_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_key: str,
                repeats: int = None, n_splits: int = None, hp_overrides: dict = None,
                seed: int = None, max_fpr: float = None):
    """Repeated, leak-free, vector-group StratifiedGroupKFold, generalized
    across all 4 models, with F1-optimal thresholding inside each fitter.
    Returns one row PER REPEAT (metrics pooled across that repeat's folds),
    matching M2 unified_validation.py's pooling fix.

    max_fpr: overrides config.MAX_FPR for this call (used by
    operating_curve.py to sweep several caps without editing config.py).
    Defaults to config.MAX_FPR when not given.
    """
    repeats = repeats or CV_CONFIG["repeats"]
    n_splits = n_splits or CV_CONFIG["n_splits"]
    seed = seed if seed is not None else CV_CONFIG["seed"]
    effective_max_fpr = max_fpr if max_fpr is not None else MAX_FPR
    fitter = _FITTERS[model_key]

    X = impute(X)
    rows = []
    for rep in range(repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)

        y_true_all, y_pred_all, y_proba_all = [], [], []
        for tr, te in cv.split(X, y, groups):
            assert len(set(groups[tr]) & set(groups[te])) == 0, "Leakage: overlapping groups!"
            pred, proba = fitter(X[tr], y[tr], groups[tr], X[te], hp_overrides, effective_max_fpr)
            y_true_all.append(y[te]); y_pred_all.append(pred); y_proba_all.append(proba)

        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        y_proba_all = np.concatenate(y_proba_all)

        scores = _score_fold(y_true_all, y_pred_all, y_proba_all)
        scores.update({"repeat": rep, "model": model_key, "n_total": len(y_true_all)})
        rows.append(scores)
        print(f"[train_eval] {model_key} repeat {rep+1}/{repeats}: "
              f"F1={scores['f1']:.4f} AUROC={scores['auroc']:.4f} FPR={scores['fpr']:.4f}", flush=True)

    import pandas as pd
    return pd.DataFrame(rows)
