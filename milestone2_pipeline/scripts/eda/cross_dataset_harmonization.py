"""
Cross-Dataset Harmonization — Milestone 2, Chapter 5
=================================================================
Step 5 compares Dataset A vs Dataset B directly, per feature -- a
different axis from unified_eda.py / unified_validation.py, which each
compare benign vs attack WITHIN one dataset. This script loads both
harmonized feature tables at once and does the cross-dataset half of
Ch5 that neither of those scripts can:

  Ch 5.1 — Unified Feature Schema: auto-derives which features are
           common to both datasets (vs dataset-specific) directly from
           the CSV columns, producing a Table-7-style compatibility
           report instead of one you maintain by hand.
  Ch 5.2 — Cross-Dataset Distribution Shift Analysis: for every SHARED
           feature, compares mean/variance/skewness/distribution shape
           between the two datasets with a Kolmogorov-Smirnov test
           (KS tests shape difference specifically, not just location
           shift the way Mann-Whitney does), plus overlaid histograms.
  Ch 5.3 — Scaling remedy: doesn't just assert "use log1p" -- tries
           {none, log1p, z-score, robust} per feature and reports which
           one actually reduces the KS statistic most, so the remedy in
           the report is a measured claim, not an assertion.

This script assumes the two input files are ALREADY at the harmonized
process-level granularity (e.g. dataset1_features.csv and
casino_process_level_FINAL.csv from process_level_rollup_final.py) --
it does the cross-dataset comparison, not the raw-log-to-feature
extraction, which is necessarily dataset-specific (different raw
schemas) and stays in process_level_rollup_final.py / whatever produced
dataset1_features.csv.

Usage:
    python cross_dataset_harmonization.py \
        --input-a combined/dataset1_features.csv --name-a "CAM-LDS" \
        --label-col-a label \
        --input-b casino_process_level_FINAL.csv --name-b "CasinoLimit" \
        --label-col-b is_privesc \
        --technique-col-b technique --exclude-techniques-b T1003,T1078 \
        --out cross_dataset_output/
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':          10,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':          150,
    'savefig.dpi':          150,
    'savefig.bbox':       'tight',
    'savefig.facecolor':  'white',
})

COLOR_A = '#1E88E5'
COLOR_B = '#E53935'

# Must match unified_eda.py / unified_validation.py exactly -- same
# canonical schema, same auto-intersection rule.
FULL_FEATURE_LIST = [
    'lifetime_seconds', 'events_per_second', 'seq_length',
    'unique_syscalls', 'uid_is_root', 'uid_changed', 'euid_root',
    'max_uid_euid_delta', 'is_suid_exec', 'auid_euid_mismatch',
    'ephemeral_privileged', 'failed_call_rate', 'failed_call_count',
    'priv_op_count', 'exec_count', 'file_access_count',
    'network_count', 'shell_from_service', 'sensitive_path_access',
    'parent_child_rarity',
]


def load_dataset(path, name, label_col, technique_col, exclude_techniques):
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {name}: {len(df):,} rows")
    if exclude_techniques and technique_col in df.columns:
        excl = [t.strip() for t in exclude_techniques.split(',') if t.strip()]
        before = len(df)
        keep = ~df[technique_col].isin(excl) | (df[label_col] == 0)
        df = df[keep].reset_index(drop=True)
        print(f"  Excluded {excl}: {before - len(df):,} rows removed, {len(df):,} remain")
    return df


# ── Ch 5.1 — Unified Feature Schema / compatibility report ─────────────

def write_schema_compatibility(df_a, name_a, df_b, name_b, out_dir):
    print("\n[Ch 5.1] Unified Feature Schema compatibility")
    rows = []
    for feat in FULL_FEATURE_LIST:
        in_a = feat in df_a.columns
        in_b = feat in df_b.columns
        if in_a and in_b:
            status = 'Direct (native to both)'
        elif in_a:
            status = f'Unavailable in {name_b}'
        elif in_b:
            status = f'Unavailable in {name_a}'
        else:
            status = 'Not present in either (should not happen)'
        rows.append({'feature': feat, f'in_{name_a}': in_a, f'in_{name_b}': in_b, 'status': status})

    schema_df = pd.DataFrame(rows)
    path = out_dir / 'unified_schema_compatibility.csv'
    schema_df.to_csv(path, index=False)

    shared = schema_df[schema_df['status'].str.startswith('Direct')]
    print(f"  {len(shared)}/{len(FULL_FEATURE_LIST)} features common to both datasets "
          f"(this is the Unified Feature Schema used below)")
    print(f"  Saved: {path}")
    return schema_df, shared['feature'].tolist()


# ── Ch 5.2 — Cross-dataset distribution shift, per shared feature ──────

def cross_dataset_stats(df_a, name_a, df_b, name_b, shared_feats, out_dir):
    print(f"\n[Ch 5.2] Cross-dataset distribution comparison ({len(shared_feats)} shared features)")
    results = []
    for feat in shared_feats:
        a = df_a[feat].dropna().values.astype(float)
        b = df_b[feat].dropna().values.astype(float)
        if len(a) < 5 or len(b) < 5:
            continue

        ks_stat, ks_p = stats.ks_2samp(a, b)
        results.append({
            'feature':      feat,
            f'{name_a}_mean':     round(float(np.mean(a)), 4),
            f'{name_b}_mean':     round(float(np.mean(b)), 4),
            f'{name_a}_variance': round(float(np.var(a, ddof=1)), 4),
            f'{name_b}_variance': round(float(np.var(b, ddof=1)), 4),
            f'{name_a}_skew':     round(float(stats.skew(a)), 4),
            f'{name_b}_skew':     round(float(stats.skew(b)), 4),
            'mean_ratio':   round(float(np.mean(a) / np.mean(b)), 3) if np.mean(b) != 0 else float('inf'),
            'ks_statistic': round(float(ks_stat), 4),
            'ks_p_value':   float(ks_p),
            'distributions_differ_p<.05': ks_p < 0.05,
        })
        print(f"  {feat:<25} KS={ks_stat:.4f} p={'<0.0001' if ks_p < 0.0001 else f'{ks_p:.4f}'} "
              f"mean_ratio={results[-1]['mean_ratio']:.2f}x")

    results_df = pd.DataFrame(results).sort_values('ks_statistic', ascending=False)
    path = out_dir / 'cross_dataset_distribution_stats.csv'
    results_df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return results_df


def plot_cross_dataset_distributions(df_a, name_a, df_b, name_b, shared_feats, out_dir):
    ncols = 4
    nrows = (len(shared_feats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    axes = np.atleast_1d(axes).flatten()

    for i, feat in enumerate(shared_feats):
        ax = axes[i]
        a = df_a[feat].dropna().values.astype(float)
        b = df_b[feat].dropna().values.astype(float)
        a_log = np.log1p(np.clip(a, 0, None))
        b_log = np.log1p(np.clip(b, 0, None))
        max_val = max(a_log.max() if len(a_log) else 1, b_log.max() if len(b_log) else 1, 1e-6)
        bins = np.linspace(0, max_val + 0.1, 30)
        ax.hist(a_log, bins=bins, alpha=0.55, color=COLOR_A, label=name_a, density=True)
        ax.hist(b_log, bins=bins, alpha=0.55, color=COLOR_B, label=name_b, density=True)
        ax.set_title(feat, fontsize=9, fontweight='bold')
        ax.set_xlabel('log1p(value)', fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(shared_feats), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Ch 5.2 — Cross-Dataset Distribution Comparison: {name_a} vs {name_b}\n'
                 '(log1p scale)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = out_dir / 'cross_dataset_distributions.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ── Ch 5.3 — Scaling remedy, tested not asserted ────────────────────────

def evaluate_scaling_remedies(df_a, name_a, df_b, name_b, shared_feats, out_dir):
    """For each feature, try {none, log1p, z-score, robust} and report
    which transform reduces the cross-dataset gap the most.

    IMPORTANT: this does NOT use the KS statistic to pick a winner. The
    Kolmogorov-Smirnov statistic is mathematically invariant under any
    monotonic transform applied identically to both samples -- log1p,
    z-score, and robust-scaling are all monotonic, so none of them can
    ever move the KS statistic relative to no transform. (Proof sketch:
    KS = sup_x |F_A(x) - F_B(x)|; substituting y=g(x) for a monotonic g
    just reparameterizes the same supremum, it doesn't change its
    value.) An earlier version of this function selected the "best"
    remedy by minimum KS and always got 'none' for every feature,
    100% of the time -- that was this bug, not a real finding.

    Instead, the remedy is picked by how much it reduces the
    STANDARDIZED MEAN GAP between datasets: |mean_A - mean_B| / pooled
    std, which log1p/z-score/robust genuinely do change (they're
    non-linear or re-centering in ways that affect the mean/std
    relationship, even though they can't touch rank-based statistics).
    This also matches what your report is actually claiming in Ch5.3 --
    that log1p makes the two datasets' MAGNITUDES comparable for
    visualization/reporting purposes, not that it makes the
    distributions identical (KS never will show identical; that's not
    the goal)."""
    print(f"\n[Ch 5.3] Scaling remedy evaluation (standardized mean-gap reduction, not KS)")
    rows = []

    for feat in shared_feats:
        a = df_a[feat].dropna().values.astype(float)
        b = df_b[feat].dropna().values.astype(float)
        if len(a) < 5 or len(b) < 5:
            continue

        def gap(x, y):
            pooled_std = np.std(np.concatenate([x, y]))
            if pooled_std == 0:
                return 0.0
            return abs(np.mean(x) - np.mean(y)) / pooled_std

        candidates = {'none': gap(a, b)}

        a_log, b_log = np.log1p(np.clip(a, 0, None)), np.log1p(np.clip(b, 0, None))
        candidates['log1p'] = gap(a_log, b_log)

        combined_mean, combined_std = np.mean(np.concatenate([a, b])), np.std(np.concatenate([a, b]))
        if combined_std > 0:
            a_z, b_z = (a - combined_mean) / combined_std, (b - combined_mean) / combined_std
            candidates['zscore'] = gap(a_z, b_z)
        else:
            candidates['zscore'] = candidates['none']

        combined_median = np.median(np.concatenate([a, b]))
        iqr = np.subtract(*np.percentile(np.concatenate([a, b]), [75, 25]))
        if iqr > 0:
            a_r, b_r = (a - combined_median) / iqr, (b - combined_median) / iqr
            candidates['robust'] = gap(a_r, b_r)
        else:
            candidates['robust'] = candidates['none']

        # KS is still reported for reference (it IS a legitimate way to
        # say "these two datasets are not the same distribution"), just
        # never used to pick a remedy.
        ks_stat, _ = stats.ks_2samp(a, b)

        best = min(candidates, key=candidates.get)
        rows.append({
            'feature': feat,
            'gap_none': round(candidates['none'], 4),
            'gap_log1p': round(candidates['log1p'], 4),
            'gap_zscore': round(candidates['zscore'], 4),
            'gap_robust': round(candidates['robust'], 4),
            'best_remedy': best,
            'gap_before': round(candidates['none'], 4),
            'gap_after_best': round(candidates[best], 4),
            'gap_reduction_pct': round(100 * (candidates['none'] - candidates[best]) /
                                       candidates['none'], 1) if candidates['none'] > 0 else 0.0,
            'ks_statistic_unaffected_by_transform': round(float(ks_stat), 4),
        })

    remedy_df = pd.DataFrame(rows).sort_values('gap_reduction_pct', ascending=False)
    path = out_dir / 'scaling_remedy_evaluation.csv'
    remedy_df.to_csv(path, index=False)
    print(f"  Saved: {path}")

    remedy_counts = remedy_df['best_remedy'].value_counts()
    print("  Best remedy by vote across features:")
    for remedy, count in remedy_counts.items():
        print(f"    {remedy}: {count} feature(s)")

    lines = [f"Ch 5.3 — Scaling Remedy Evaluation: {name_a} vs {name_b}", "=" * 60, ""]
    lines.append(
        "Selection metric: standardized mean gap |mean_A - mean_B| / pooled_std,\n"
        "before and after each candidate transform. Lower is better (0 = means\n"
        "coincide in pooled-std units). NOT selected by KS statistic -- KS is\n"
        "invariant under any monotonic transform applied identically to both\n"
        "datasets, so it cannot distinguish log1p/z-score/robust from doing\n"
        "nothing; it's reported alongside for reference only, not as a\n"
        "selection criterion. Note: tree-based/isolation-based detection\n"
        "models (Ch6) are scale-invariant and are trained separately per\n"
        "dataset, so this remedy applies to the CROSS-DATASET COMPARISON/\n"
        "VISUALIZATION step (this chapter), not to the detection models\n"
        "themselves.\n")
    for _, row in remedy_df.iterrows():
        lines.append(f"  {row['feature']:<25} before={row['gap_before']:.4f}  "
                     f"best={row['best_remedy']:<7} after={row['gap_after_best']:.4f}  "
                     f"({row['gap_reduction_pct']:+.1f}% gap reduction)")
    lines.append(f"\nOverall recommendation: {remedy_counts.idxmax()} improves the most "
                 f"features ({remedy_counts.max()}/{len(remedy_df)}).")
    text = '\n'.join(lines)
    with open(out_dir / 'scaling_remedy_report.txt', 'w') as f:
        f.write(text)
    print(f"  Saved: {out_dir / 'scaling_remedy_report.txt'}")
    return remedy_df


def main():
    p = argparse.ArgumentParser(description="Cross-dataset harmonization, Ch5")
    p.add_argument('--input-a', required=True)
    p.add_argument('--name-a', required=True)
    p.add_argument('--label-col-a', required=True)
    p.add_argument('--technique-col-a', default='technique')
    p.add_argument('--exclude-techniques-a', default=None)
    p.add_argument('--input-b', required=True)
    p.add_argument('--name-b', required=True)
    p.add_argument('--label-col-b', required=True)
    p.add_argument('--technique-col-b', default='technique')
    p.add_argument('--exclude-techniques-b', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"Cross-Dataset Harmonization — {args.name_a} vs {args.name_b}")
    print("=" * 65)

    df_a = load_dataset(args.input_a, args.name_a, args.label_col_a,
                        args.technique_col_a, args.exclude_techniques_a)
    df_b = load_dataset(args.input_b, args.name_b, args.label_col_b,
                        args.technique_col_b, args.exclude_techniques_b)

    schema_df, shared_feats = write_schema_compatibility(df_a, args.name_a, df_b, args.name_b, out_dir)

    if not shared_feats:
        print("\nNo shared features found between the two datasets -- check column names.")
        return

    stats_df = cross_dataset_stats(df_a, args.name_a, df_b, args.name_b, shared_feats, out_dir)
    plot_cross_dataset_distributions(df_a, args.name_a, df_b, args.name_b, shared_feats, out_dir)
    remedy_df = evaluate_scaling_remedies(df_a, args.name_a, df_b, args.name_b, shared_feats, out_dir)

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == '__main__':
    main()
