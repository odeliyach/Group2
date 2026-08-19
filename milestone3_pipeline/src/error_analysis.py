"""
Milestone 3 — Step 8: Sample-Level Error Analysis
====================================================================
Goes beyond aggregate metrics (F1/AUROC/FPR) to the actual rows: which
specific attack techniques does each model consistently miss, what
distinguishes the benign rows that get flagged, and why does one model
succeed on a sample another model fails on.

Method: ONE fixed-seed StratifiedGroupKFold split (not the 15-repeat
headline CV -- this needs per-row predictions, not aggregate scores), with
the SAME fold assignment reused across all 4 models so per-row comparisons
between models are apples-to-apples. Reuses train_eval.py's existing
fitters (_fit_predict_rf etc.) directly, so this is scored under the
exact same F1-threshold/MAX_FPR logic as the headline pipeline -- not a
separate, inconsistent evaluation path.

Usage:
    python error_analysis.py --dataset camlds --data-dir combined
    python error_analysis.py --dataset casino --data-dir combined --models rf,iforest
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold

from ingestion import ingest
from preprocessing import vector_group_key, log1p_continuous, impute
from feature_selection import select_features
from train_eval import _FITTERS
from config import LOG1P_CONTINUOUS_FEATURES, BINARY_FEATURES, MAX_FPR


def get_oof_predictions(dataset_key, data_dir, feature_policy="full",
                         n_splits=5, seed=42, models=None):
    """One full out-of-fold prediction pass, same fold assignment shared
    across all models -- necessary for fair per-row cross-model comparison."""
    ds = ingest(dataset_key, data_dir=data_dir)
    feat_cols = select_features(ds.feat_cols, dataset_key, policy=feature_policy)
    groups = vector_group_key(ds.df, feat_cols)
    X = ds.df[feat_cols].values.astype(float)
    if LOG1P_CONTINUOUS_FEATURES:
        X = log1p_continuous(X, feat_cols, BINARY_FEATURES)
    X = impute(X)
    y = ds.y.values
    n = len(y)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(cv.split(X, y, groups))
    for tr, te in folds:
        assert len(set(groups[tr]) & set(groups[te])) == 0, "Leakage: overlapping groups!"

    models = models or list(_FITTERS.keys())
    results = {}
    for model_key in models:
        fitter = _FITTERS[model_key]
        pred_full = np.full(n, -1, dtype=int)
        proba_full = np.full(n, np.nan, dtype=float)
        for fold_i, (tr, te) in enumerate(folds):
            pred, proba = fitter(X[tr], y[tr], groups[tr], X[te], None, MAX_FPR)
            pred_full[te] = pred
            proba_full[te] = proba
            print(f"[error_analysis] {model_key} fold {fold_i+1}/{n_splits} done", flush=True)
        results[model_key] = (pred_full, proba_full)

    return ds, feat_cols, X, y, results


def build_error_dataframe(ds, y, results):
    df = ds.df.copy()
    df["true_label"] = y
    for model_key, (pred, proba) in results.items():
        df[f"pred_{model_key}"] = pred
        df[f"proba_{model_key}"] = proba
        df[f"correct_{model_key}"] = (pred == y).astype(int)
        err = np.full(len(y), "TN", dtype=object)
        err[(y == 1) & (pred == 1)] = "TP"
        err[(y == 1) & (pred == 0)] = "FN"
        err[(y == 0) & (pred == 1)] = "FP"
        df[f"errtype_{model_key}"] = err
    return df


def technique_breakdown(df, model_key, technique_col="technique"):
    """FN rate per attack technique -- which specific techniques does this
    model consistently miss, vs. which does it catch reliably."""
    attack_rows = df[df["true_label"] == 1]
    if technique_col not in attack_rows.columns or attack_rows.empty:
        return pd.DataFrame()
    counts = attack_rows.groupby(technique_col)[f"errtype_{model_key}"].value_counts().unstack(fill_value=0)
    for col in ["TP", "FN"]:
        if col not in counts.columns:
            counts[col] = 0
    counts["total"] = counts[["TP", "FN"]].sum(axis=1)
    counts["fn_rate"] = counts["FN"] / counts["total"].replace(0, np.nan)
    return counts.sort_values("fn_rate", ascending=False)


def benign_fp_summary(df, model_key):
    """What fraction of benign rows get flagged, and how many total FPs."""
    benign_rows = df[df["true_label"] == 0]
    n_fp = (benign_rows[f"errtype_{model_key}"] == "FP").sum()
    return {"model": model_key, "n_benign": len(benign_rows), "n_fp": int(n_fp),
            "fp_rate": n_fp / len(benign_rows) if len(benign_rows) else np.nan}


def feature_comparison(df, feat_cols, model_key):
    """Feature means: FN (missed attacks) vs TP (caught attacks). A large
    ratio deviation from 1.0 suggests that feature is where the model's
    decision boundary is failing for the missed cases specifically."""
    fn = df[df[f"errtype_{model_key}"] == "FN"]
    tp = df[df[f"errtype_{model_key}"] == "TP"]
    rows = []
    for col in feat_cols:
        fn_mean = fn[col].mean() if len(fn) else np.nan
        tp_mean = tp[col].mean() if len(tp) else np.nan
        ratio = (fn_mean / tp_mean) if (tp_mean not in (0, np.nan) and not pd.isna(tp_mean) and tp_mean != 0) else np.nan
        rows.append({"feature": col, "n_FN": len(fn), "n_TP": len(tp),
                     "FN_mean": fn_mean, "TP_mean": tp_mean, "FN_over_TP_ratio": ratio})
    result = pd.DataFrame(rows)
    result["abs_dev_from_1"] = (result["FN_over_TP_ratio"] - 1).abs()
    return result.sort_values("abs_dev_from_1", ascending=False)


def fp_vs_tn_feature_comparison(df, feat_cols, model_key):
    """FP (benign flagged as attack) vs TN (benign correctly cleared) --
    THE brief's explicit question: 'is a specific type of benign
    background traffic causing a high FPR?' feature_comparison() above
    answers a DIFFERENT question (missed attacks vs caught attacks) --
    this one is the FP-side complement, using the same Cohen's-d-style
    standardized difference so features are comparable across scales."""
    fp = df[df[f"errtype_{model_key}"] == "FP"]
    tn = df[df[f"errtype_{model_key}"] == "TN"]
    rows = []
    for col in feat_cols:
        fp_mean = fp[col].mean() if len(fp) else np.nan
        tn_mean = tn[col].mean() if len(tn) else np.nan
        pooled_std = np.sqrt(((fp[col].var() if len(fp) > 1 else 0) +
                               (tn[col].var() if len(tn) > 1 else 0)) / 2)
        cohens_d = (fp_mean - tn_mean) / pooled_std if pooled_std > 0 else np.nan
        rows.append({"feature": col, "n_FP": len(fp), "n_TN": len(tn),
                     "FP_mean": fp_mean, "TN_mean": tn_mean, "cohens_d": cohens_d})
    result = pd.DataFrame(rows)
    result["abs_cohens_d"] = result["cohens_d"].abs()
    return result.sort_values("abs_cohens_d", ascending=False)


def plot_technique_fn_rates(df, models, dataset_key, out_dir):
    """Chart companion to technique_breakdown() -- one bar group per
    model, one row per technique, sorted worst-first."""
    all_rates = {}
    for model_key in models:
        tb = technique_breakdown(df, model_key)
        if not tb.empty:
            all_rates[model_key] = tb["fn_rate"]
    if not all_rates:
        return
    combined = pd.DataFrame(all_rates).fillna(0)
    combined = combined.loc[combined.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(11, max(4, 0.4 * len(combined))))
    n_models = len(all_rates)
    bar_h = 0.8 / max(n_models, 1)
    y_pos = np.arange(len(combined))
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    for i, model_key in enumerate(all_rates):
        ax.barh(y_pos + i * bar_h, combined[model_key], height=bar_h, label=model_key, color=colors[i])
    ax.set_yticks(y_pos + bar_h * (n_models - 1) / 2)
    ax.set_yticklabels(combined.index, fontsize=8)
    ax.set_xlabel("False Negative rate (fraction of this technique's attacks missed)")
    ax.set_title(f"Per-technique FN rate by model: {dataset_key}")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    path = out_dir / f"{dataset_key}_technique_fn_rates.png"
    plt.savefig(path)
    plt.close()
    print(f"[error_analysis] Saved: {path}")


def plot_fp_feature_comparison(fp_comparisons, dataset_key, out_dir):
    """fp_comparisons: dict of model_key -> fp_vs_tn_feature_comparison() result."""
    models = [m for m, df_ in fp_comparisons.items() if not df_.empty]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for i, model_key in enumerate(models):
        ax = axes[0][i]
        sub = fp_comparisons[model_key].dropna(subset=["cohens_d"]).head(8).sort_values("cohens_d")
        if sub.empty:
            continue
        colors = ["#E53935" if d > 0 else "#1E88E5" for d in sub["cohens_d"]]
        ax.barh(sub["feature"], sub["cohens_d"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{model_key}: features distinguishing FP from TN benign")
        ax.set_xlabel("Cohen's d (FP mean - TN mean)")
        ax.tick_params(labelsize=8)
    plt.tight_layout()
    path = out_dir / f"{dataset_key}_fp_feature_comparison.png"
    plt.savefig(path)
    plt.close()
    print(f"[error_analysis] Saved: {path}")


def pairwise_model_comparison(df, feat_cols, model_a, model_b):
    """Rows where model_a failed but model_b succeeded, and vice versa --
    directly answers 'why did RF fail on a sample CNN got right'."""
    a_wrong_b_right = df[(df[f"correct_{model_a}"] == 0) & (df[f"correct_{model_b}"] == 1)]
    a_right_b_wrong = df[(df[f"correct_{model_a}"] == 1) & (df[f"correct_{model_b}"] == 0)]

    rows = []
    for col in feat_cols:
        m1 = a_wrong_b_right[col].mean() if len(a_wrong_b_right) else np.nan
        m2 = a_right_b_wrong[col].mean() if len(a_right_b_wrong) else np.nan
        rows.append({"feature": col,
                      f"{model_a}_wrong_{model_b}_right_mean": m1,
                      f"{model_a}_right_{model_b}_wrong_mean": m2})
    comparison = pd.DataFrame(rows)
    summary = {"model_a": model_a, "model_b": model_b,
               "n_a_wrong_b_right": len(a_wrong_b_right),
               "n_a_right_b_wrong": len(a_right_b_wrong)}
    return comparison, summary


def main():
    parser = argparse.ArgumentParser(description="Step 8 sample-level error analysis")
    parser.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--models", default="rf,xgb,cnn,iforest")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="error_analysis_results")
    args = parser.parse_args()

    models = args.models.split(",")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[error_analysis] dataset={args.dataset} models={models}")
    ds, feat_cols, X, y, results = get_oof_predictions(
        args.dataset, args.data_dir, n_splits=args.n_splits, seed=args.seed, models=models)

    df = build_error_dataframe(ds, y, results)
    df.to_csv(out_dir / f"{args.dataset}_error_rows.csv", index=False)
    print(f"[error_analysis] Saved full row-level predictions: "
          f"{out_dir / f'{args.dataset}_error_rows.csv'}")

    fp_summaries = []
    fp_comparisons = {}
    for model_key in models:
        print(f"\n{'='*65}\n{model_key}\n{'='*65}")

        tb = technique_breakdown(df, model_key)
        if not tb.empty:
            tb.to_csv(out_dir / f"{args.dataset}_{model_key}_technique_breakdown.csv")
            print(f"FN rate by technique (worst first):")
            print(tb[["FN", "TP", "total", "fn_rate"]].to_string())

        fp = benign_fp_summary(df, model_key)
        fp_summaries.append(fp)
        print(f"\nBenign FP rate: {fp['n_fp']}/{fp['n_benign']} ({fp['fp_rate']:.2%})")

        fc = feature_comparison(df, feat_cols, model_key)
        fc.to_csv(out_dir / f"{args.dataset}_{model_key}_feature_comparison.csv", index=False)
        print(f"\nTop 5 features most different between missed (FN) and caught (TP) attacks:")
        print(fc[["feature", "FN_mean", "TP_mean", "FN_over_TP_ratio"]].head(5).to_string())

        fpc = fp_vs_tn_feature_comparison(df, feat_cols, model_key)
        fpc.to_csv(out_dir / f"{args.dataset}_{model_key}_fp_vs_tn_feature_comparison.csv", index=False)
        fp_comparisons[model_key] = fpc
        print(f"\nTop 5 features most different between false-positive and true-negative benign rows:")
        print(fpc[["feature", "FP_mean", "TN_mean", "cohens_d"]].head(5).to_string())

    pd.DataFrame(fp_summaries).to_csv(out_dir / f"{args.dataset}_fp_summary.csv", index=False)

    print(f"\n{'='*65}\nGenerating charts\n{'='*65}")
    plot_technique_fn_rates(df, models, args.dataset, out_dir)
    plot_fp_feature_comparison(fp_comparisons, args.dataset, out_dir)

    if len(models) >= 2:
        print(f"\n{'='*65}\nPairwise model comparisons\n{'='*65}")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                comp, summary = pairwise_model_comparison(df, feat_cols, models[i], models[j])
                comp.to_csv(out_dir / f"{args.dataset}_{models[i]}_vs_{models[j]}_pairwise.csv", index=False)
                print(f"\n{models[i]} wrong & {models[j]} right: {summary['n_a_wrong_b_right']} rows")
                print(f"{models[i]} right & {models[j]} wrong: {summary['n_a_right_b_wrong']} rows")

    print(f"\n[error_analysis] Done. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
