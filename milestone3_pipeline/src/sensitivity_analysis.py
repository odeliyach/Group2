"""
Milestone 3 — Step 7 deliverable: Hyperparameter Sensitivity Analysis.

Usage:
    python sensitivity_analysis.py --dataset camlds --model rf
    python sensitivity_analysis.py --dataset casino --model xgb --repeats 5
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ingestion import ingest
from preprocessing import vector_group_key, log1p_continuous
from feature_selection import select_features
from train_eval import repeated_cv
from config import SENSITIVITY_GRIDS, LOG1P_CONTINUOUS_FEATURES, BINARY_FEATURES


def sweep(dataset_key: str, model_key: str, data_dir: str, repeats: int, out_dir: Path):
    ds = ingest(dataset_key, data_dir=data_dir)
    feat_cols = select_features(ds.feat_cols, dataset_key, policy="full")
    groups = vector_group_key(ds.df, feat_cols)
    X = ds.df[feat_cols].values.astype(float)
    if LOG1P_CONTINUOUS_FEATURES:
        X = log1p_continuous(X, feat_cols, BINARY_FEATURES)
    y = ds.y.values

    grid = SENSITIVITY_GRIDS[model_key]
    keys, values = list(grid.keys()), list(grid.values())
    combos = list(itertools.product(*values))

    rows = []
    for combo in combos:
        overrides = dict(zip(keys, combo))
        print(f"\n[sensitivity] {model_key} / {ds.display_name} / {overrides}")
        try:
            cv_df = repeated_cv(X, y, groups, model_key, repeats=repeats, hp_overrides=overrides)
        except Exception as e:
            print(f"[sensitivity]   FAILED: {e}")
            continue
        rows.append({
            **overrides,
            "f1_mean": cv_df["f1"].mean(), "f1_std": cv_df["f1"].std(),
            "auroc_mean": cv_df["auroc"].mean(), "auroc_std": cv_df["auroc"].std(),
            "fpr_mean": cv_df["fpr"].mean(),
        })

    result_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{dataset_key}_{model_key}_sensitivity.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\n[sensitivity] Saved: {csv_path}")

    _plot(result_df, keys, dataset_key, model_key, out_dir)
    return result_df


def _plot(df, swept_keys, dataset_key, model_key, out_dir):
    if df.empty:
        return
    fig, axes = plt.subplots(1, len(swept_keys), figsize=(6 * len(swept_keys), 4.5))
    axes = [axes] if len(swept_keys) == 1 else axes

    for ax, key in zip(axes, swept_keys):
        other_keys = [k for k in swept_keys if k != key]
        if other_keys:
            first_vals = {k: df[k].iloc[0] for k in other_keys}
            mask = pd.Series(True, index=df.index)
            for k, v in first_vals.items():
                mask &= (df[k] == v)
            sub = df[mask].sort_values(key)
        else:
            sub = df.sort_values(key)
        ax.errorbar(sub[key].astype(str), sub["f1_mean"], yerr=sub["f1_std"],
                    marker="o", capsize=4, color="#1565C0", label="F1")
        ax2 = ax.twinx()
        ax2.plot(sub[key].astype(str), sub["auroc_mean"], marker="s", color="#E53935", label="AUROC")
        ax.set_xlabel(key)
        ax.set_ylabel("F1 (blue)")
        ax2.set_ylabel("AUROC (red)")
        ax.set_title(f"{model_key} / {dataset_key}: sensitivity to {key}")

    plt.tight_layout()
    path = out_dir / f"{dataset_key}_{model_key}_sensitivity.png"
    plt.savefig(path)
    plt.close()
    print(f"[sensitivity] Saved: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    p.add_argument("--model", choices=["rf", "xgb", "cnn", "iforest"], required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", default="sensitivity_results")
    args = p.parse_args()
    sweep(args.dataset, args.model, args.data_dir, args.repeats, Path(args.out))


if __name__ == "__main__":
    main()
