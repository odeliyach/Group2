"""
Milestone 3 -- Step 8 follow-up: Cascade confidence-band threshold sweep
=================================================================================
hybrid_cascade.py's uncertain band [0.3, 0.7] was a design choice (symmetric
around the decision boundary, half-width 0.2), never swept or validated --
noted as an explicit limitation in the report. This script closes that gap.

Sweeps the band's half-width around 0.5: narrower bands route fewer rows to
CNN (preserving more of XGBoost's own strength, but risking under-using CNN's
genuine complementary signal on truly ambiguous rows); wider bands route more
(exploiting that signal more, but also diluting XGBoost's strength on rows
that weren't actually that ambiguous -- see the original fold-1 finding).

Given everything this session found about CNN's run-to-run instability
(bimodal FP rate, batch-order/float non-determinism), THIS SWEEP IS REPEATED
PER CONFIG (not single-shot) -- unlike a naive first attempt would be. Default
3 repeats/config, matching sensitivity_analysis.py's convention for cheap
exploratory sweeps (the expensive 15-repeat protocol is reserved for the
final chosen config's headline number, not every grid point).

Usage:
    python cascade_threshold_sweep.py --dataset camlds --data-dir data --out results/current/cascade_threshold_sweep/
    python cascade_threshold_sweep.py --dataset casino --data-dir data --out results/current/cascade_threshold_sweep/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

from config import MAX_FPR
from ingestion import ingest
from feature_selection import select_features
from preprocessing import vector_group_key
from train_eval import _fpr
from hybrid_cascade import cascade_predict  # reuse the exact same cascade logic

# Half-widths around 0.5 -- 0.2 is the original, never-swept default.
HALF_WIDTHS = [0.10, 0.15, 0.20, 0.25, 0.30]


def run_one_config(dataset_key, data_dir, max_fpr, half_width, repeats, n_splits=5, base_seed=42):
    low_thresh, high_thresh = 0.5 - half_width, 0.5 + half_width
    ds = ingest(dataset_key, data_dir=data_dir)
    feat_cols = select_features(ds.feat_cols, dataset_key, policy="full")
    X = ds.df[feat_cols].fillna(0).values
    y = ds.y.values
    groups = vector_group_key(ds.df, feat_cols)

    rows = []
    for rep in range(repeats):
        seed = base_seed + rep
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_f1, fold_auc, fold_fpr, fold_routed = [], [], [], []
        for tr, te in cv.split(X, y, groups):
            if len(set(y[te])) < 2:
                continue  # same degenerate-fold guard as hybrid_cascade.py
            y_pred, y_proba, uncertain, *_ = cascade_predict(
                X[tr], y[tr], X[te], feat_cols, max_fpr,
                low_thresh=low_thresh, high_thresh=high_thresh)
            fold_f1.append(f1_score(y[te], y_pred, zero_division=0))
            fold_auc.append(roc_auc_score(y[te], y_proba))
            fold_fpr.append(_fpr(y[te], y_pred))
            fold_routed.append(100 * uncertain.mean())
        if fold_f1:
            rows.append({'repeat': rep, 'seed': seed,
                         'f1': np.mean(fold_f1), 'auroc': np.mean(fold_auc),
                         'fpr': np.mean(fold_fpr), 'pct_routed': np.mean(fold_routed)})
        print(f"[{dataset_key}] half_width={half_width:.2f} repeat {rep+1}/{repeats}: "
              f"F1={rows[-1]['f1']:.4f} AUROC={rows[-1]['auroc']:.4f} "
              f"routed={rows[-1]['pct_routed']:.1f}%")

    df = pd.DataFrame(rows)
    return {
        'half_width': half_width, 'low_thresh': low_thresh, 'high_thresh': high_thresh,
        'f1_mean': df['f1'].mean(), 'f1_std': df['f1'].std(),
        'auroc_mean': df['auroc'].mean(), 'auroc_std': df['auroc'].std(),
        'pct_routed_mean': df['pct_routed'].mean(), 'pct_routed_std': df['pct_routed'].std(),
    }


def main():
    p = argparse.ArgumentParser(description="Sweep the cascade's confidence-band half-width")
    p.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    p.add_argument("--data-dir", default=".")
    p.add_argument("--out", required=True)
    p.add_argument("--max-fpr", type=float, default=MAX_FPR)
    p.add_argument("--repeats", type=int, default=3,
                    help="Repeats per config -- kept >1 given CNN's documented instability")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    csv_path = out_dir / f"{args.dataset}_threshold_sweep.csv"

    # Resume support: skip half-widths already present in a partial CSV from
    # an earlier (e.g. timed-out) run, rather than repeating finished work.
    already_done = set()
    if csv_path.exists():
        prior = pd.read_csv(csv_path)
        results = prior.to_dict('records')
        already_done = set(prior['half_width'].round(2))
        print(f"Resuming: found {len(already_done)} completed configs in {csv_path}: {sorted(already_done)}")

    for hw in HALF_WIDTHS:
        if round(hw, 2) in already_done:
            print(f"\n=== [{args.dataset}] half_width={hw:.2f} -- already done, skipping ===")
            continue
        print(f"\n=== [{args.dataset}] half_width={hw:.2f}  (band=[{0.5-hw:.2f}, {0.5+hw:.2f}]) ===")
        results.append(run_one_config(args.dataset, args.data_dir, args.max_fpr, hw, args.repeats))

        # BUGFIX: write after every config, not just once at the end -- a
        # timed-out job previously lost all progress since to_csv() only
        # ran after the full grid finished. Same lesson already learned
        # the hard way with the XGBoost sensitivity sweep; applying it here.
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"[checkpoint] Saved {len(results)}/{len(HALF_WIDTHS)} configs to {csv_path}")

    result_df = pd.DataFrame(results)
    print(f"\nSaved: {csv_path}")

    best_row = result_df.loc[result_df['f1_mean'].idxmax()]
    current_row = result_df[result_df['half_width'] == 0.20].iloc[0]
    print(f"\nCurrent default (half_width=0.20): F1={current_row['f1_mean']:.4f}\u00b1{current_row['f1_std']:.4f}")
    print(f"Best in sweep (half_width={best_row['half_width']:.2f}): "
          f"F1={best_row['f1_mean']:.4f}\u00b1{best_row['f1_std']:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.errorbar(result_df['half_width'], result_df['f1_mean'], yerr=result_df['f1_std'],
                 marker='o', color='#1565C0', capsize=4)
    ax1.axvline(0.20, color='#999999', linestyle='--', linewidth=1, label='current default')
    ax1.set_xlabel('Band half-width')
    ax1.set_ylabel('F1')
    ax1.set_title(f'{args.dataset}: F1 vs. band half-width')
    ax1.legend(fontsize=8)

    ax2.errorbar(result_df['half_width'], result_df['pct_routed_mean'], yerr=result_df['pct_routed_std'],
                 marker='o', color='#E65100', capsize=4)
    ax2.axvline(0.20, color='#999999', linestyle='--', linewidth=1)
    ax2.set_xlabel('Band half-width')
    ax2.set_ylabel('% rows routed to CNN')
    ax2.set_title(f'{args.dataset}: routing volume vs. band half-width')

    plt.tight_layout()
    png_path = out_dir / f"{args.dataset}_threshold_sweep.png"
    plt.savefig(png_path, dpi=150, facecolor='white', bbox_inches='tight')
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
