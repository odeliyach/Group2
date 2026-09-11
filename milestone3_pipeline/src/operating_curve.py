"""
Milestone 3 — Step 7/8: FPR Operating-Point Sweep.

Answers "what's a realistic FPR cap?" with evidence instead of a guess.
Sweeps config.py's MAX_FPR across several candidate values and reports
F1/Precision/Recall/AUROC at each, so you can see the actual trade-off
curve and pick (and justify, in your report) a specific operating point
rather than an arbitrary number.

IMPORTANT INTERPRETATION: this sweep only helps for models that actually
discriminate (real AUROC well above 0.5, e.g. RF/XGBoost on CAM-LDS). For
a non-discriminative model (e.g. Isolation Forest on CAM-LDS, AUROC~0.53),
EVERY cap on this curve will look bad -- that's not a wrong cap, that's
the honest signal that no threshold fixes a model with no separating
power. Don't chase a "good-looking" cap for those cases; report the flat,
low curve itself as the Ch8 finding.

Usage:
    python operating_curve.py --dataset camlds --model rf
    python operating_curve.py --dataset camlds --model rf --caps 0.01,0.02,0.05,0.10,0.15,0.20,0.30
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ingestion import ingest
from preprocessing import vector_group_key, log1p_continuous
from feature_selection import select_features
from train_eval import repeated_cv
from config import LOG1P_CONTINUOUS_FEATURES, BINARY_FEATURES

DEFAULT_CAPS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]  # 1.0 = uncapped


def sweep_operating_points(dataset_key, model_key, data_dir, caps, repeats, out_dir):
    ds = ingest(dataset_key, data_dir=data_dir)
    feat_cols = select_features(ds.feat_cols, dataset_key, policy="full")
    groups = vector_group_key(ds.df, feat_cols)
    X = ds.df[feat_cols].values.astype(float)
    if LOG1P_CONTINUOUS_FEATURES:
        X = log1p_continuous(X, feat_cols, BINARY_FEATURES)
    y = ds.y.values

    rows = []
    for cap in caps:
        effective_cap = None if cap >= 1.0 else cap  # 1.0 == uncapped sentinel
        print(f"\n[operating_curve] {model_key}/{ds.display_name}: max_fpr={cap}")
        cv_df = repeated_cv(X, y, groups, model_key, repeats=repeats, max_fpr=effective_cap)
        rows.append({
            "max_fpr_cap": cap,
            "f1_mean": cv_df["f1"].mean(), "f1_std": cv_df["f1"].std(),
            "precision_mean": cv_df["precision"].mean(),
            "recall_mean": cv_df["recall"].mean(),
            "actual_fpr_mean": cv_df["fpr"].mean(),  # what FPR the model actually achieved
            "auroc_mean": cv_df["auroc"].mean(),      # constant across caps -- ranking is threshold-independent
        })

    result_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{dataset_key}_{model_key}_operating_curve.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"\n[operating_curve] Saved: {csv_path}")

    _plot(result_df, dataset_key, model_key, out_dir)
    return result_df


def _plot(df, dataset_key, model_key, out_dir):
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(df["actual_fpr_mean"], df["f1_mean"], marker="o", color="#1565C0", label="F1")
    ax1.plot(df["actual_fpr_mean"], df["recall_mean"], marker="s", color="#2E7D32", label="Recall")
    ax1.plot(df["actual_fpr_mean"], df["precision_mean"], marker="^", color="#E53935", label="Precision")
    ax1.set_xlabel("Actual FPR achieved (false-alarm rate)")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    auroc = df["auroc_mean"].iloc[0]
    ax1.set_title(f"{model_key} / {dataset_key}: F1/Recall/Precision vs FPR budget\n"
                  f"(AUROC={auroc:.3f} -- flat & low here means no threshold will fix this model)")
    ax1.axvspan(0, 0.05, alpha=0.08, color="green")   # rough "low false-alarm" zone, illustrative only
    plt.tight_layout()
    path = out_dir / f"{dataset_key}_{model_key}_operating_curve.png"
    plt.savefig(path)
    plt.close()
    print(f"[operating_curve] Saved: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    p.add_argument("--model", choices=["rf", "xgb", "cnn", "mlp", "iforest"], required=True)
    p.add_argument("--caps", default=",".join(str(c) for c in DEFAULT_CAPS),
                   help="Comma-separated FPR caps to test, e.g. 0.01,0.05,0.10 (use 1.0 for uncapped)")
    p.add_argument("--repeats", type=int, default=5,
                   help="Cheaper than the headline 15 -- this is an exploratory sweep")
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", default="operating_curve_results")
    args = p.parse_args()

    caps = [float(c) for c in args.caps.split(",")]
    sweep_operating_points(args.dataset, args.model, args.data_dir, caps, args.repeats, Path(args.out))


if __name__ == "__main__":
    main()
