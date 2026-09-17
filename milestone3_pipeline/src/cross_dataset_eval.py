"""
Milestone 3 — Step 8: Cross-Dataset Generalization (train on one, test on other)
==================================================================================
Distinct from run_pipeline.py's within-dataset CV. That script answers
"how good is each model at detecting this threat, given data from THIS
environment" -- it never lets one dataset's rows influence the other's
evaluation.

THIS script answers the actual Step 8 question: "if a model gets 99% on
Dataset A and 60% on Dataset B, is that Environmental Distribution Shift
or Model Overfitting to the primary training set?" -- which is only a
coherent question if there IS a single "primary training set" being
tested elsewhere. So: train fully on Dataset A, evaluate on the ENTIRETY
of Dataset B (a truly unseen environment), and vice versa.

Constraint this forces: only the features BOTH datasets share can be
used (17/20 -- CasinoLimit lacks is_suid_exec, ephemeral_privileged,
parent_child_rarity per M2 Table 7). A model trained on 20 features can't
be evaluated on data that only has 17.

F1-threshold selection (see train_eval.py's disclosure) still only uses
TRAINING-dataset labels -- an inner held-out slice of the training
dataset picks the threshold; the test dataset is only ever touched once,
for final scoring.

Usage:
    python cross_dataset_eval.py --model rf --data-dir /path/to/csvs
    python cross_dataset_eval.py --model all --data-dir /path/to/csvs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, accuracy_score)

from ingestion import ingest
from preprocessing import (vector_group_key, impute, fit_scaler_on_train,
                            fold_class_weights, xgb_scale_pos_weight, log1p_continuous)
from models import build_rf, build_xgb, build_cnn, build_mlp, cnn_fit_kwargs, mlp_fit_kwargs, build_iforest
from train_eval import _best_f1_threshold
from config import MAX_FPR, LOG1P_CONTINUOUS_FEATURES, BINARY_FEATURES


def _shared_features(ds_a, ds_b):
    """Only features BOTH datasets actually have -- see module docstring."""
    shared = [f for f in ds_a.feat_cols if f in ds_b.feat_cols]
    dropped_a = [f for f in ds_a.feat_cols if f not in shared]
    dropped_b = [f for f in ds_b.feat_cols if f not in shared]
    if dropped_a or dropped_b:
        print(f"[cross_dataset] Restricting to {len(shared)} shared features "
              f"(dropped {dropped_a or dropped_b} -- not present in the other dataset)")
    return shared


def row_proportional_group_split(y, groups, target_frac, seed=0):
    """Splits into (calib_idx, remaining_idx) allocating approximately
    target_frac of ROWS (not groups) to calibration, while keeping every
    group entirely on one side (no leakage across the split).

    FIX for a real bug: GroupShuffleSplit's train_size allocates a
    fraction of GROUPS, which collapses to a tiny absolute row count
    under extreme duplication. Observed on Casino (356 unique groups,
    91,692 rows): train_size=0.05 -> 5% of 356 groups (~18 groups), but
    since group sizes are wildly uneven (one dominant group can hold
    thousands of duplicate rows while most hold a handful), the actual
    row count landed on just 35 -- nowhere near 5% of the dataset. This
    made the CAM-LDS->Casino recalibration test meaningless (too few
    rows to derive a threshold from, even with excellent AUROC).

    This function instead shuffles groups randomly and accumulates whole
    groups until the cumulative ROW count reaches target_frac * n_total,
    guaranteeing calibration gets a real, representative row budget.
    Trade-off: because groups are atomic (can't be split), the achieved
    fraction can overshoot the target somewhat when group sizes are
    lumpy -- reported explicitly via the returned row count, not hidden.
    """
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    n_total = len(y)
    target_n = int(round(target_frac * n_total))

    calib_mask = np.zeros(n_total, dtype=bool)
    running_total = 0
    for g in unique_groups:
        if running_total >= target_n:
            break
        g_mask = (groups == g)
        calib_mask |= g_mask
        running_total += int(g_mask.sum())

    calib_idx = np.where(calib_mask)[0]
    remaining_idx = np.where(~calib_mask)[0]
    return calib_idx, remaining_idx


def _fit_on_full_train(model_key, X_train, y_train, groups_train, hp_overrides=None):
    """Fits a model on the ENTIRE training dataset (not a CV fold) and
    returns (predict_fn, threshold) -- predict_fn(X) -> proba array.
    Threshold picked on an inner held-out slice of the training data only,
    same discipline as train_eval.py's within-dataset fitters, with the
    same MAX_FPR operational cap.
    """
    X_train = impute(X_train)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=0)
    fit_idx, val_idx = next(gss.split(X_train, y_train, groups_train))

    if model_key == "rf":
        thresh_clf = build_rf(hp_overrides)
        thresh_clf.fit(X_train[fit_idx], y_train[fit_idx])
        val_proba = thresh_clf.predict_proba(X_train[val_idx])[:, 1]
        threshold = _best_f1_threshold(y_train[val_idx], val_proba, max_fpr=MAX_FPR)

        clf = build_rf(hp_overrides)
        clf.fit(X_train, y_train)
        return (lambda X: clf.predict_proba(impute(X))[:, 1]), threshold

    if model_key == "xgb":
        spw_inner = xgb_scale_pos_weight(y_train[fit_idx])
        thresh_clf = build_xgb(scale_pos_weight=spw_inner, overrides=hp_overrides)
        thresh_clf.fit(X_train[fit_idx], y_train[fit_idx])
        val_proba = thresh_clf.predict_proba(X_train[val_idx])[:, 1]
        threshold = _best_f1_threshold(y_train[val_idx], val_proba, max_fpr=MAX_FPR)

        spw = xgb_scale_pos_weight(y_train)
        clf = build_xgb(scale_pos_weight=spw, overrides=hp_overrides)
        clf.fit(X_train, y_train)
        return (lambda X: clf.predict_proba(impute(X))[:, 1]), threshold

    if model_key == "cnn":
        scaler = fit_scaler_on_train(X_train)
        X_train_s = scaler.transform(X_train)
        cw = fold_class_weights(y_train[fit_idx])

        clf = build_cnn(n_features=X_train.shape[1], overrides=hp_overrides)
        fit_kwargs = cnn_fit_kwargs(hp_overrides)
        clf.fit(X_train_s[fit_idx][..., None], y_train[fit_idx], class_weight=cw,
                validation_data=(X_train_s[val_idx][..., None], y_train[val_idx]), **fit_kwargs)

        val_proba = clf.predict(X_train_s[val_idx][..., None], verbose=0).ravel()
        threshold = _best_f1_threshold(y_train[val_idx], val_proba, max_fpr=MAX_FPR)

        def predict_fn(X, scaler=scaler, clf=clf):
            X_s = scaler.transform(impute(X))[..., None]
            return clf.predict(X_s, verbose=0).ravel()
        return predict_fn, threshold

    if model_key == "mlp":
        scaler = fit_scaler_on_train(X_train)
        X_train_s = scaler.transform(X_train)
        cw = fold_class_weights(y_train[fit_idx])

        clf = build_mlp(n_features=X_train.shape[1], overrides=hp_overrides)
        fit_kwargs = mlp_fit_kwargs(hp_overrides)
        clf.fit(X_train_s[fit_idx], y_train[fit_idx], class_weight=cw,
                validation_data=(X_train_s[val_idx], y_train[val_idx]), **fit_kwargs)

        val_proba = clf.predict(X_train_s[val_idx], verbose=0).ravel()
        threshold = _best_f1_threshold(y_train[val_idx], val_proba, max_fpr=MAX_FPR)

        def predict_fn(X, scaler=scaler, clf=clf):
            X_s = scaler.transform(impute(X))
            return clf.predict(X_s, verbose=0).ravel()
        return predict_fn, threshold

    if model_key == "iforest":
        clf = build_iforest(hp_overrides)
        clf.fit(X_train)
        train_score = -clf.score_samples(X_train)
        smin, smax = train_score.min(), train_score.max()
        train_proba = (train_score - smin) / (smax - smin + 1e-12)
        threshold = _best_f1_threshold(y_train, train_proba, max_fpr=MAX_FPR)

        def predict_fn(X, clf=clf, smin=smin, smax=smax):
            score = -clf.score_samples(impute(X))
            return (score - smin) / (smax - smin + 1e-12)
        return predict_fn, threshold

    raise ValueError(f"Unknown model_key '{model_key}'")


def _score(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "auroc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
    }


def run_cross_dataset(model_key: str, data_dir: str, out_dir: Path, hp_overrides=None,
                       calibration_frac: float = 0.0, repeats: int = 1):
    """repeats: re-fits the model `repeats` times with a different random
    seed each time, per direction. Matters MOST for CNN -- RF/XGBoost/
    IsolationForest use a fixed random_state by default (config.py), so
    repeating them mainly confirms stability rather than revealing new
    variance. CNN has no equivalent fixed seed on its weight init, so a
    single run can't tell you whether a bad (or good) number is a stable
    property of cross-domain transfer or just one unlucky/lucky training
    run -- this is exactly the ambiguity in the CAM-LDS->Casino CNN result
    (AUROC 0.18, notably WORSE than chance) that a single run can't resolve.
    """
    ds_a = ingest("camlds", data_dir=data_dir)
    ds_b = ingest("casino", data_dir=data_dir)
    shared_feats = _shared_features(ds_a, ds_b)

    X_a = ds_a.df[shared_feats].values.astype(float)
    y_a = ds_a.y.values
    groups_a = vector_group_key(ds_a.df, shared_feats)

    X_b = ds_b.df[shared_feats].values.astype(float)
    y_b = ds_b.y.values
    groups_b = vector_group_key(ds_b.df, shared_feats)

    if LOG1P_CONTINUOUS_FEATURES:
        X_a = log1p_continuous(X_a, shared_feats, BINARY_FEATURES)
        X_b = log1p_continuous(X_b, shared_feats, BINARY_FEATURES)

    results = []
    for train_name, X_tr, y_tr, g_tr, test_name, X_te, y_te, g_te in [
        ("CAM-LDS", X_a, y_a, groups_a, "CasinoLimit", X_b, y_b, groups_b),
        ("CasinoLimit", X_b, y_b, groups_b, "CAM-LDS", X_a, y_a, groups_a),
    ]:
        for rep in range(repeats):
            seed = 42 + rep
            try:
                import numpy as _np
                _np.random.seed(seed)
                import tensorflow as _tf
                _tf.random.set_seed(seed)
            except ImportError:
                pass
            rep_overrides = {**(hp_overrides or {}), "random_state": seed}

            print(f"\n[cross_dataset] Train={train_name} -> Test={test_name}  "
                  f"(model={model_key}, repeat {rep+1}/{repeats}, seed={seed})")
            predict_fn, threshold = _fit_on_full_train(model_key, X_tr, y_tr, g_tr, rep_overrides)

            proba_train = predict_fn(X_tr)
            pred_train = (proba_train >= threshold).astype(int)
            scores_train = _score(y_tr, pred_train, proba_train)

            proba_test = predict_fn(X_te)
            pred_test = (proba_test >= threshold).astype(int)
            scores_test = _score(y_te, pred_test, proba_test)

            row = {
                "model": model_key, "train_dataset": train_name, "test_dataset": test_name,
                "repeat": rep, "seed": seed, "n_shared_features": len(shared_feats),
                **{f"insample_{k}": v for k, v in scores_train.items()},
                **{f"crossdataset_{k}": v for k, v in scores_test.items()},
                "f1_drop": scores_train["f1"] - scores_test["f1"],
            }

            if calibration_frac > 0:
                calib_idx, remaining_idx = row_proportional_group_split(
                    y_te, g_te, calibration_frac, seed=0)
                proba_calib = proba_test[calib_idx]
                recalib_threshold = _best_f1_threshold(y_te[calib_idx], proba_calib, max_fpr=MAX_FPR)

                pred_recalib = (proba_test[remaining_idx] >= recalib_threshold).astype(int)
                scores_recalib = _score(y_te[remaining_idx], pred_recalib, proba_test[remaining_idx])
                row.update({f"recalibrated_{k}": v for k, v in scores_recalib.items()})
                row["recalibration_n_labeled_samples"] = len(calib_idx)
                row["f1_recovered_by_recalibration"] = scores_recalib["f1"] - scores_test["f1"]
                print(f"  Recalibrated (using {len(calib_idx)} labeled target samples, "
                      f"{100*len(calib_idx)/len(y_te):.1f}% of target rows): "
                      f"F1={scores_recalib['f1']:.4f} FPR={scores_recalib['fpr']:.4f} "
                      f"(recovered {row['f1_recovered_by_recalibration']:+.4f} F1 vs. no recalibration)")

            results.append(row)
            print(f"  In-sample   (train==test): F1={scores_train['f1']:.4f} AUROC={scores_train['auroc']:.4f}")
            print(f"  Cross-dataset (unseen env): F1={scores_test['f1']:.4f} AUROC={scores_test['auroc']:.4f} "
                  f"FPR={scores_test['fpr']:.4f}  (F1 drop: {row['f1_drop']:+.4f})")

    df = pd.DataFrame(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cross_dataset_{model_key}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[cross_dataset] Saved: {out_path}")

    if repeats > 1:
        print(f"\n[cross_dataset] Stability across {repeats} repeats (mean +/- std of crossdataset_f1):")
        for (train_name, test_name), grp in df.groupby(["train_dataset", "test_dataset"]):
            print(f"  {train_name} -> {test_name}: "
                  f"F1={grp['crossdataset_f1'].mean():.4f}+/-{grp['crossdataset_f1'].std():.4f}  "
                  f"AUROC={grp['crossdataset_auroc'].mean():.4f}+/-{grp['crossdataset_auroc'].std():.4f}")
    return df


def main():
    p = argparse.ArgumentParser(description="Step 8 cross-dataset generalization: train on one, test on other")
    p.add_argument("--model", choices=["rf", "xgb", "cnn", "mlp", "iforest", "all"], default="all")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", default="outputs/cross_dataset")
    p.add_argument("--calibration-frac", type=float, default=0.0,
                   help="If >0, also report a before/after recalibration experiment: "
                        "picks a new threshold from this fraction of TARGET-domain labeled "
                        "data, evaluates on the rest. E.g. 0.05 = use 5%% of the target "
                        "dataset as a small labeled calibration sample.")
    p.add_argument("--repeats", type=int, default=1,
                   help="Re-fit each direction this many times with different seeds -- "
                        "use >1 (e.g. 5) especially for CNN to check whether a striking "
                        "result is stable or just one unlucky/lucky training run.")
    args = p.parse_args()

    out_dir = Path(args.out)
    models = ["rf", "xgb", "cnn", "iforest"] if args.model == "all" else [args.model]

    all_dfs = []
    for model_key in models:
        try:
            all_dfs.append(run_cross_dataset(model_key, args.data_dir, out_dir,
                                              calibration_frac=args.calibration_frac,
                                              repeats=args.repeats))
        except ImportError as e:
            print(f"[cross_dataset] SKIPPED {model_key}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(out_dir / "cross_dataset_all_models.csv", index=False)
        print(f"\n[cross_dataset] Combined summary saved: {out_dir / 'cross_dataset_all_models.csv'}")


if __name__ == "__main__":
    main()
