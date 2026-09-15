"""Milestone 3 -- Step 8 capstone: dump the rows where the hybrid cascade's two
models contradict each other.

A row qualifies iff, out-of-fold:
  1. routed  -- cascade_low_thresh <= proba_xgb <= cascade_high_thresh, and
  2. contradicting -- pred_xgb != pred_cnn (each model's own F1-optimal call).

XGBoost and the CNN are fit per fold via train_eval._FITTERS, the exact fitters
error_analysis.py and the headline pipeline use, so these calls are scored under
the identical F1-threshold / MAX_FPR logic. Nothing is imported from
hybrid_cascade.py here: the routing band (0.3/0.7) is mirrored in
`config.LLM_TRIAGE` from `hybrid_cascade.cascade_predict`'s defaults; keep the two
in sync.

Usage:
    python llm_triage_dump.py --dataset camlds --data-dir data/v3 --out results/current/llm_triage/camlds
    python llm_triage_dump.py --dataset casino --data-dir data     --out results/current/llm_triage/casino
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (LLM_TRIAGE, DATASETS, LOG1P_CONTINUOUS_FEATURES,
                    BINARY_FEATURES, MAX_FPR)

LOW = LLM_TRIAGE["cascade_low_thresh"]
HIGH = LLM_TRIAGE["cascade_high_thresh"]


def contradiction_mask(oof, low=LOW, high=HIGH):
    routed = oof["proba_xgb"].between(low, high, inclusive="both")
    disagree = oof["pred_xgb"] != oof["pred_cnn"]
    valid = (oof["pred_xgb"] >= 0) & (oof["pred_cnn"] >= 0)
    return routed & disagree & valid


def add_direction(contra):
    contra = contra.copy()
    contra["disagreement_direction"] = np.where(
        contra["pred_xgb"].astype(int) == 0,
        "xgb_benign_cnn_malicious", "xgb_malicious_cnn_benign")
    return contra


def stratified_subsample(df, max_cases, seed):
    if len(df) <= max_cases:
        return df.sort_values("row_id").reset_index(drop=True)
    strat = (df["disagreement_direction"].astype(str) + "|" + df["technique"].astype(str))
    total = len(df)
    picks = []
    for _, g in df.groupby(strat):
        k = max(1, int(round(max_cases * len(g) / total)))
        k = min(k, len(g))
        picks.append(g.sample(n=k, random_state=seed))
    result = pd.concat(picks)
    if len(result) > max_cases:
        result = result.sample(n=max_cases, random_state=seed)
    return result.sort_values("row_id").reset_index(drop=True)


def _oof_calls(dataset_key, data_dir, seed, n_splits):
    from sklearn.model_selection import StratifiedGroupKFold
    from ingestion import ingest
    from feature_selection import select_features
    from preprocessing import vector_group_key, log1p_continuous, impute
    from train_eval import _FITTERS

    ds = ingest(dataset_key, data_dir=data_dir)
    feat_cols = select_features(ds.feat_cols, dataset_key, policy="full")
    groups = vector_group_key(ds.df, feat_cols)
    X_raw = ds.df[feat_cols].values.astype(float)
    X = (log1p_continuous(X_raw, feat_cols, BINARY_FEATURES)
         if LOG1P_CONTINUOUS_FEATURES else X_raw.copy())
    X = impute(X)
    y = ds.y.values
    n = len(y)

    fold_id = np.full(n, -1, dtype=int)
    preds = {mk: np.full(n, -1, dtype=int) for mk in ("xgb", "cnn")}
    probas = {mk: np.full(n, np.nan, dtype=float) for mk in ("xgb", "cnn")}

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_i, (tr, te) in enumerate(cv.split(X, y, groups)):
        assert not (set(groups[tr]) & set(groups[te])), "group leakage"
        fold_id[te] = fold_i
        for mk in ("xgb", "cnn"):
            p, pr = _FITTERS[mk](X[tr], y[tr], groups[tr], X[te], None, MAX_FPR)
            preds[mk][te] = p
            probas[mk][te] = pr
        print(f"[llm_triage_dump] {dataset_key} fold {fold_i + 1}/{n_splits} done", flush=True)

    tech_col = DATASETS[dataset_key].get("technique_col")
    out = ds.df[feat_cols].copy().reset_index(drop=True)
    out.insert(0, "row_id", np.arange(n))
    out["fold"] = fold_id
    out["true_label"] = y
    out["technique"] = (ds.df[tech_col].values
                        if tech_col and tech_col in ds.df.columns else "unknown")
    for mk in ("xgb", "cnn"):
        out[f"pred_{mk}"] = preds[mk]
        out[f"proba_{mk}"] = probas[mk]
    # Apply fillna(0.0) to feature columns to match preprocessing.impute behavior (np.nan_to_num)
    out[feat_cols] = out[feat_cols].fillna(0.0)
    return out, feat_cols


def find_contradictions(dataset_key, data_dir, seed=42, n_splits=5):
    oof, feat_cols = _oof_calls(dataset_key, data_dir, seed, n_splits)
    contra = add_direction(oof[contradiction_mask(oof)].copy()).reset_index(drop=True)
    return contra, oof, feat_cols


def main():
    ap = argparse.ArgumentParser(description="Dump hybrid-cascade contradiction rows")
    ap.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--max-cases", type=int, default=LLM_TRIAGE["max_cases_per_dataset"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    contra, oof, _ = find_contradictions(args.dataset, args.data_dir, args.seed, args.n_splits)
    contra = stratified_subsample(contra, args.max_cases, args.seed)

    oof.to_csv(out_dir / f"{args.dataset}_cascade_oof.csv", index=False)
    contra.to_csv(out_dir / f"{args.dataset}_contradictions.csv", index=False)
    print(f"[llm_triage_dump] {args.dataset}: {len(oof):,} OOF rows, "
          f"{len(contra):,} contradiction rows after cap -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
