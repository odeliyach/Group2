"""
EDA Script — Chapter 3: Exploratory Data Analysis
===================================================
Produces all required outputs for Milestone 2 Ch 3:

  Ch 3.1 — Class distribution plots
  Ch 3.2 — Feature distributions (benign vs attack)
  Ch 3.3 — Statistical tests: Mann-Whitney U + effect size per feature
  Ch 4.1 — Random Forest feature importance bar chart
  Ch 4.2 — Discrepancy analysis (domain ranking vs RF ranking)

Output files in eda_output/:
  class_distribution.png
  feature_distributions.png
  correlation_matrix.png
  mannwhitney_results.csv
  rf_feature_importance.png
  rf_feature_importance.csv
  eda_summary.txt

Usage:
    python run_eda.py \
        --features combined/dataset1_features.csv \
        --fixed    combined/dataset1_fixed.csv \
        --out      eda_output/

SLURM:
    sbatch run_eda.sh
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix)

warnings.filterwarnings('ignore')

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   11,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'figure.dpi':       150,
    'savefig.dpi':      150,
    'savefig.bbox':     'tight',
    'savefig.facecolor':'white',
})

BENIGN_COLOR = '#2196F3'   # blue
ATTACK_COLOR = '#E53935'   # red
BOTH_COLORS  = [BENIGN_COLOR, ATTACK_COLOR]

MODEL_FEATURES = [
    'lifetime_seconds', 'events_per_second', 'seq_length',
    'unique_syscalls', 'uid_is_root', 'uid_changed', 'euid_root',
    'max_uid_euid_delta', 'is_suid_exec', 'auid_euid_mismatch',
    'ephemeral_privileged', 'failed_call_rate', 'failed_call_count',
    'priv_op_count', 'exec_count', 'file_access_count',
    'network_count', 'shell_from_service', 'sensitive_path_access',
    'parent_child_rarity',
]

# Domain-expert ranking (from Ch 1 feature table rationale)
DOMAIN_RANK = {
    'is_suid_exec':          1,
    'auid_euid_mismatch':    2,
    'max_uid_euid_delta':    3,
    'priv_op_count':         4,
    'exec_count':            5,
    'sensitive_path_access': 6,
    'shell_from_service':    7,
    'parent_child_rarity':   8,
    'ephemeral_privileged':  9,
    'events_per_second':     10,
    'failed_call_rate':      11,
    'euid_root':             12,
    'uid_is_root':           13,
    'uid_changed':           14,
    'failed_call_count':     15,
    'file_access_count':     16,
    'network_count':         17,
    'unique_syscalls':       18,
    'seq_length':            19,
    'lifetime_seconds':      20,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def cohens_d(a, b):
    """Cohen's d effect size between two groups."""
    na, nb  = len(a), len(b)
    pooled  = np.sqrt(((na-1)*np.var(a,ddof=1) + (nb-1)*np.var(b,ddof=1)) / (na+nb-2))
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled

def effect_size_label(d):
    d = abs(d)
    if d >= 0.8: return 'large'
    if d >= 0.5: return 'medium'
    if d >= 0.2: return 'small'
    return 'negligible'


# ── Ch 3.1 — Class distribution ───────────────────────────────────────────────

def plot_class_distribution(df, out_dir):
    print("  Plotting class distribution...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Overall class balance
    counts = df['label'].value_counts().sort_index()
    axes[0].bar(['Benign (0)', 'Attack (1)'], counts.values,
                color=BOTH_COLORS, edgecolor='white', linewidth=0.5)
    axes[0].set_title('Overall class balance')
    axes[0].set_ylabel('Process count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + max(counts)*0.01, f'{v:,}\n({100*v/len(df):.1f}%)',
                     ha='center', va='bottom', fontsize=10)

    # By source
    source_counts = df.groupby(['source','label']).size().unstack(fill_value=0)
    source_counts.plot(kind='bar', ax=axes[1], color=BOTH_COLORS,
                       edgecolor='white', linewidth=0.5, legend=False)
    axes[1].set_title('Class balance by source')
    axes[1].set_ylabel('Process count')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha='right')
    axes[1].legend(['Benign', 'Attack'], fontsize=9)

    # By technique (top 10 attack techniques)
    atk = df[df['label']==1]
    tech_counts = atk['technique'].value_counts().head(10)
    axes[2].barh(range(len(tech_counts)), tech_counts.values,
                 color=ATTACK_COLOR, alpha=0.8)
    axes[2].set_yticks(range(len(tech_counts)))
    axes[2].set_yticklabels([t[:30] for t in tech_counts.index], fontsize=8)
    axes[2].set_title('Top 10 attack techniques')
    axes[2].set_xlabel('Process count')
    axes[2].invert_yaxis()

    plt.suptitle('Ch 3.1 — Class Distribution Analysis', fontsize=14,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    path = out_dir / 'class_distribution.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    return counts


# ── Ch 3.2 — Feature distributions ───────────────────────────────────────────

def plot_feature_distributions(df, out_dir):
    print("  Plotting feature distributions...")
    benign = df[df['label'] == 0]
    attack = df[df['label'] == 1]

    n_feat = len(MODEL_FEATURES)
    ncols  = 4
    nrows  = (n_feat + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    axes = axes.flatten()

    for i, feat in enumerate(MODEL_FEATURES):
        ax = axes[i]
        if feat not in df.columns:
            ax.set_visible(False)
            continue

        b_vals = benign[feat].dropna().values
        a_vals = attack[feat].dropna().values

        # For binary features use bar chart
        if set(df[feat].dropna().unique()).issubset({0, 1}):
            b_rate = b_vals.mean() if len(b_vals) else 0
            a_rate = a_vals.mean() if len(a_vals) else 0
            ax.bar(['Benign', 'Attack'], [b_rate, a_rate],
                   color=BOTH_COLORS, edgecolor='white', alpha=0.85)
            ax.set_ylabel('Rate')
            ax.set_ylim(0, max(b_rate, a_rate) * 1.4 + 0.01)
        else:
            # Clip extreme outliers for display
            clip_val = np.percentile(np.concatenate([b_vals, a_vals]), 99)
            b_clip   = np.clip(b_vals, 0, clip_val)
            a_clip   = np.clip(a_vals, 0, clip_val)
            bins     = min(40, max(10, int(clip_val)))
            ax.hist(b_clip, bins=bins, alpha=0.6, color=BENIGN_COLOR,
                    label='Benign', density=True)
            ax.hist(a_clip, bins=bins, alpha=0.6, color=ATTACK_COLOR,
                    label='Attack', density=True)
            if i == 0:
                ax.legend(fontsize=8)

        ax.set_title(feat, fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Ch 3.2 — Feature Distributions: Benign vs Attack',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = out_dir / 'feature_distributions.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ── Ch 3.2 — Correlation matrix ───────────────────────────────────────────────

def plot_correlation_matrix(df, out_dir):
    print("  Plotting correlation matrix...")
    feat_cols = [f for f in MODEL_FEATURES if f in df.columns]
    corr = df[feat_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')

    ax.set_xticks(range(len(feat_cols)))
    ax.set_yticks(range(len(feat_cols)))
    ax.set_xticklabels(feat_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(feat_cols, fontsize=8)

    # Annotate cells with r values
    for i in range(len(feat_cols)):
        for j in range(len(feat_cols)):
            val = corr.values[i, j]
            if abs(val) > 0.3 and i != j:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color='black' if abs(val) < 0.7 else 'white')

    ax.set_title('Ch 3.2 — Feature Correlation Matrix', fontsize=13,
                 fontweight='bold', pad=15)
    plt.tight_layout()
    path = out_dir / 'correlation_matrix.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")
    return corr


# ── Ch 3.3 — Mann-Whitney U tests ─────────────────────────────────────────────

def run_statistical_tests(df, out_dir):
    print("  Running Mann-Whitney U tests...")
    benign = df[df['label'] == 0]
    attack = df[df['label'] == 1]
    results = []

    for feat in MODEL_FEATURES:
        if feat not in df.columns:
            continue
        b = benign[feat].dropna().values
        a = attack[feat].dropna().values
        if len(b) < 5 or len(a) < 5:
            continue

        u_stat, p_val = stats.mannwhitneyu(a, b, alternative='two-sided')
        d = cohens_d(a, b)
        n_min = min(len(a), len(b))
        r_effect = u_stat / (len(a) * len(b))  # rank-biserial correlation

        results.append({
            'feature':       feat,
            'benign_mean':   round(float(b.mean()), 4),
            'attack_mean':   round(float(a.mean()), 4),
            'benign_median': round(float(np.median(b)), 4),
            'attack_median': round(float(np.median(a)), 4),
            'ratio':         round(float(a.mean()/b.mean()), 3) if b.mean()>0 else float('inf'),
            'U_statistic':   round(float(u_stat), 1),
            'p_value':       float(p_val),
            'p_value_str':   '<0.0001' if p_val < 0.0001 else f'{p_val:.4f}',
            'cohens_d':      round(float(d), 4),
            'effect_size':   effect_size_label(d),
            'significant':   p_val < 0.05,
            'domain_rank':   DOMAIN_RANK.get(feat, 99),
        })

    results_df = pd.DataFrame(results).sort_values('cohens_d', ascending=False,
                                                    key=abs)
    path = out_dir / 'mannwhitney_results.csv'
    results_df.to_csv(path, index=False)

    # Print summary
    print(f"\n  {'Feature':<25} {'B.mean':>8} {'A.mean':>8} "
          f"{'p-value':>10} {'Cohen d':>8} {'Effect':>12} {'Sig?':>5}")
    print(f"  {'─'*80}")
    for _, row in results_df.iterrows():
        sig = '✅' if row['significant'] else '❌'
        print(f"  {row['feature']:<25} {row['benign_mean']:>8.4f} "
              f"{row['attack_mean']:>8.4f} {row['p_value_str']:>10} "
              f"{row['cohens_d']:>8.4f} {row['effect_size']:>12} {sig:>5}")

    print(f"\n  Saved: {path}")
    return results_df


# ── Ch 4.1 — Random Forest feature importance ─────────────────────────────────

def run_random_forest(df, out_dir):
    print("\n  Training Random Forest for feature importance...")
    feat_cols = [f for f in MODEL_FEATURES if f in df.columns]
    X = df[feat_cols].fillna(0).values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)

    print(f"  RF AUROC:    {auc:.4f}")
    print(f"  RF F1 (atk): {report['1']['f1-score']:.4f}")
    print(f"  RF Precision:{report['1']['precision']:.4f}")
    print(f"  RF Recall:   {report['1']['recall']:.4f}")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature':     feat_cols,
        'importance':  rf.feature_importances_,
        'domain_rank': [DOMAIN_RANK.get(f, 99) for f in feat_cols],
    }).sort_values('importance', ascending=False)

    path_csv = out_dir / 'rf_feature_importance.csv'
    importance_df.to_csv(path_csv, index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: horizontal bar chart sorted by RF importance
    colors = ['#E53935' if imp > importance_df['importance'].median()
              else '#90CAF9' for imp in importance_df['importance']]
    axes[0].barh(range(len(importance_df)), importance_df['importance'].values,
                 color=colors, edgecolor='white', linewidth=0.3)
    axes[0].set_yticks(range(len(importance_df)))
    axes[0].set_yticklabels(importance_df['feature'].values, fontsize=9)
    axes[0].set_xlabel('Gini importance')
    axes[0].set_title('Ch 4.1 — RF Feature Importance (Gini)', fontweight='bold')
    axes[0].invert_yaxis()
    for i, v in enumerate(importance_df['importance'].values):
        axes[0].text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)

    # Right: domain rank vs RF rank scatter
    importance_df['rf_rank'] = range(1, len(importance_df) + 1)
    axes[1].scatter(importance_df['domain_rank'],
                    importance_df['rf_rank'],
                    s=80, color='#1565C0', alpha=0.7, zorder=3)
    for _, row in importance_df.iterrows():
        axes[1].annotate(row['feature'],
                         (row['domain_rank'], row['rf_rank']),
                         fontsize=7, textcoords='offset points',
                         xytext=(4, 0), alpha=0.8)
    # Diagonal — perfect agreement line
    max_rank = len(feat_cols)
    axes[1].plot([1, max_rank], [1, max_rank], 'k--', alpha=0.3, linewidth=1,
                 label='Perfect agreement')
    axes[1].set_xlabel('Domain expert rank (Ch 1)')
    axes[1].set_ylabel('RF rank (Gini importance)')
    axes[1].set_title('Ch 4.2 — Domain vs RF Ranking', fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, max_rank + 1)
    axes[1].set_ylim(0, max_rank + 1)
    # Spearman correlation
    spearman = stats.spearmanr(importance_df['domain_rank'],
                                importance_df['rf_rank'])
    axes[1].text(0.05, 0.92,
                 f'Spearman ρ = {spearman.correlation:.3f}  '
                 f'(p={spearman.pvalue:.3f})',
                 transform=axes[1].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow',
                           edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    path_png = out_dir / 'rf_feature_importance.png'
    plt.savefig(path_png)
    plt.close()

    print(f"  Saved: {path_csv}")
    print(f"  Saved: {path_png}")
    return rf, importance_df, auc, report, spearman


# ── Ch 4.2 — Discrepancy analysis ────────────────────────────────────────────

def discrepancy_analysis(importance_df, mw_df, spearman, out):
    """Write the Ch 4.2 discrepancy analysis to a text file."""
    lines = []
    lines.append("Ch 4.2 — Feature Importance Discrepancy Analysis")
    lines.append("="*60)
    lines.append(f"\nSpearman rank correlation (domain vs RF): "
                 f"ρ={spearman.correlation:.3f}, p={spearman.pvalue:.4f}")

    if spearman.correlation > 0.6:
        lines.append("→ Strong alignment between domain knowledge and RF ranking.")
        lines.append("  The RF confirms the security-motivated feature design.")
    elif spearman.correlation > 0.3:
        lines.append("→ Moderate alignment. Some features rank differently than expected.")
    else:
        lines.append("→ Low alignment. RF identifies different patterns than domain intuition.")

    lines.append("\nTop 5 by RF importance:")
    for _, row in importance_df.head(5).iterrows():
        lines.append(f"  RF#{int(row['rf_rank'])} (Domain#{int(row['domain_rank'])}): "
                     f"{row['feature']}  importance={row['importance']:.4f}")

    lines.append("\nLargest rank discrepancies (|RF rank - domain rank| > 5):")
    importance_df['rank_diff'] = abs(
        importance_df['rf_rank'] - importance_df['domain_rank'])
    discrepancies = importance_df[importance_df['rank_diff'] > 5].sort_values(
        'rank_diff', ascending=False)
    if len(discrepancies):
        for _, row in discrepancies.iterrows():
            direction = ("ranked higher by RF than domain"
                         if row['rf_rank'] < row['domain_rank']
                         else "ranked lower by RF than domain")
            lines.append(f"  {row['feature']}: {direction} "
                         f"(RF#{int(row['rf_rank'])} vs "
                         f"Domain#{int(row['domain_rank'])})")
    else:
        lines.append("  No major discrepancies — domain and RF rankings are consistent.")

    lines.append("\nFeatures significant in Mann-Whitney but low RF importance:")
    if mw_df is not None:
        sig_feats = set(mw_df[mw_df['significant']]['feature'].values)
        low_rf = importance_df[
            (importance_df['feature'].isin(sig_feats)) &
            (importance_df['importance'] < 0.02)
        ]
        for _, row in low_rf.iterrows():
            lines.append(f"  {row['feature']}: statistically significant "
                         f"but low RF importance={row['importance']:.4f}")
            lines.append("  → Possible explanation: feature is correlated with "
                         "a stronger feature and its variance is captured there.")

    text = '\n'.join(lines)
    path = out / 'ch4_discrepancy_analysis.txt'
    with open(path, 'w') as f:
        f.write(text)
    print(f"\n  Saved: {path}")
    print(f"\n{text}")
    return text


# ── Summary report ────────────────────────────────────────────────────────────

def write_summary(df, mw_df, auc, report, out_dir):
    lines = []
    lines.append("EDA Summary Report — Milestone 2")
    lines.append("="*60)
    lines.append(f"\nDataset: dataset1_features.csv")
    lines.append(f"Total processes: {len(df):,}")
    lines.append(f"Benign (0):  {(df['label']==0).sum():,} "
                 f"({100*(df['label']==0).mean():.1f}%)")
    lines.append(f"Attack (1):  {(df['label']==1).sum():,} "
                 f"({100*(df['label']==1).mean():.1f}%)")

    lines.append(f"\nRandom Forest results:")
    lines.append(f"  AUROC:     {auc:.4f}")
    lines.append(f"  F1:        {report['1']['f1-score']:.4f}")
    lines.append(f"  Precision: {report['1']['precision']:.4f}")
    lines.append(f"  Recall:    {report['1']['recall']:.4f}")

    if mw_df is not None:
        sig = mw_df[mw_df['significant']]
        large = mw_df[mw_df['effect_size'] == 'large']
        lines.append(f"\nMann-Whitney U results:")
        lines.append(f"  Features tested:    {len(mw_df)}")
        lines.append(f"  Significant (p<.05):{len(sig)}")
        lines.append(f"  Large effect size:  {len(large)}")
        lines.append(f"\n  Top features by Cohen's |d|:")
        for _, row in mw_df.head(5).iterrows():
            lines.append(f"    {row['feature']:<25} d={row['cohens_d']:+.4f} "
                         f"({row['effect_size']}) p={row['p_value_str']}")

    text = '\n'.join(lines)
    path = out_dir / 'eda_summary.txt'
    with open(path, 'w') as f:
        f.write(text)
    print(f"\n  Saved: {path}")
    print(f"\n{text}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EDA + Feature Importance for Milestone 2 Ch 3-4"
    )
    parser.add_argument('--features', required=True,
        help='Path to dataset1_features.csv')
    parser.add_argument('--fixed', default=None,
        help='Path to dataset1_fixed.csv (optional, for raw event EDA)')
    parser.add_argument('--out', required=True,
        help='Output directory for plots and CSVs')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*65)
    print("EDA Pipeline — Milestone 2 Ch 3 + Ch 4")
    print("="*65)

    # Load feature matrix
    print(f"\n[1] Loading feature matrix: {args.features}")
    df = pd.read_csv(args.features, low_memory=False)
    print(f"  Rows: {len(df):,}  Cols: {len(df.columns)}")
    print(f"  Benign: {(df['label']==0).sum():,}  "
          f"Attack: {(df['label']==1).sum():,}")

    available = [f for f in MODEL_FEATURES if f in df.columns]
    missing   = [f for f in MODEL_FEATURES if f not in df.columns]
    print(f"  Model features available: {len(available)}/{len(MODEL_FEATURES)}")
    if missing:
        print(f"  Missing features: {missing}")

    # Ch 3.1
    print(f"\n{'─'*65}")
    print("[2] Ch 3.1 — Class distribution")
    counts = plot_class_distribution(df, out_dir)

    # Ch 3.2
    print(f"\n{'─'*65}")
    print("[3] Ch 3.2 — Feature distributions")
    plot_feature_distributions(df, out_dir)
    corr = plot_correlation_matrix(df, out_dir)

    # Highly correlated pairs
    print("\n  Highly correlated feature pairs (|r| > 0.7):")
    feat_cols = [f for f in MODEL_FEATURES if f in df.columns]
    found = False
    for i in range(len(feat_cols)):
        for j in range(i+1, len(feat_cols)):
            r = corr.loc[feat_cols[i], feat_cols[j]]
            if abs(r) > 0.7:
                print(f"    {feat_cols[i]} ↔ {feat_cols[j]}: r={r:.3f}")
                found = True
    if not found:
        print("    None above 0.7 — no redundant feature pairs detected")

    # Ch 3.3
    print(f"\n{'─'*65}")
    print("[4] Ch 3.3 — Mann-Whitney U statistical tests")
    mw_df = run_statistical_tests(df, out_dir)

    # Ch 4.1 + 4.2
    print(f"\n{'─'*65}")
    print("[5] Ch 4.1 — Random Forest feature importance")
    rf, importance_df, auc, report, spearman = run_random_forest(df, out_dir)

    print(f"\n{'─'*65}")
    print("[6] Ch 4.2 — Discrepancy analysis")
    discrepancy_analysis(importance_df, mw_df, spearman, out_dir)

    # Summary
    print(f"\n{'─'*65}")
    print("[7] Writing EDA summary report")
    write_summary(df, mw_df, auc, report, out_dir)

    print(f"\n{'='*65}")
    print("EDA COMPLETE")
    print(f"{'='*65}")
    print(f"\nAll outputs in {out_dir}/:")
    for f in sorted(out_dir.iterdir()):
        mb = f.stat().st_size / 1e6
        print(f"  {f.name:<40} {mb:>6.2f} MB")


if __name__ == '__main__':
    main()
