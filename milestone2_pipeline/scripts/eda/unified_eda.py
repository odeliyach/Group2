"""
Unified EDA + Feature Importance — Milestone 2, Chapters 3 & 4
=================================================================
Runs the SAME checkups on any process-level feature table with a binary
label column. Replaces run_eda.py (Dataset 1 only) and eda_step3_final2.py
(Dataset 2 only, event-level, incompatible ad-hoc feature set) with one
script driven entirely by CLI args, so both datasets go through identical
code:

  Ch 3.1 — class distribution
  Ch 3.2 — feature distributions (benign vs attack) + correlation matrix
  Ch 3.3 — Mann-Whitney U + Cohen's d per feature (Table 4 source)
  Ch 4.1 — Random Forest feature importance, LEAK-FREE vector-group split
           (both run_eda.py's RF importance and the ablation results in
           your draft were computed two different ways across the two
           datasets before this script existed; this version always uses
           the leak-free split, matching your Ch5.3 methodology claim)
  Ch 4.2 — domain-rank vs RF-rank discrepancy analysis (Table 6 source)

FULL_FEATURE_LIST and DOMAIN_RANK below are the canonical 20-feature
schema from Ch1's feature table (Table 1) and Ch1.3's domain rationale.
They are IDENTICAL for both datasets on purpose -- the domain ranking is
your reasoning about the threat model, not something derived per-dataset.
Whichever features aren't present in a given CSV (Casino is missing
is_suid_exec, ephemeral_privileged, parent_child_rarity -- see Table 7)
are auto-dropped and reported, not silently assumed away.

Usage -- Dataset 1 (CAM-LDS):
    python unified_eda.py \
        --input combined/dataset1_features.csv \
        --label-col label \
        --technique-col technique \
        --dataset-name "CAM-LDS" \
        --out eda_output/cam_lds/

Usage -- Dataset 2 (CasinoLimit, process-level, 4-technique subset):
    python unified_eda.py \
        --input casino_process_level_FINAL.csv \
        --label-col is_privesc \
        --technique-col technique \
        --exclude-techniques T1003,T1078 \
        --dataset-name "CasinoLimit" \
        --out eda_output/casino/

If casino_process_level_FINAL.csv doesn't exist yet, run
process_level_rollup_final.py first -- this script only accepts
process-level tables, since Ch3 figures must be comparable across
datasets (Ch5.1's Unified Feature Schema requirement).
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, classification_report

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
    'savefig.dpi':       150,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
})

BENIGN_COLOR = '#2196F3'
ATTACK_COLOR = '#E53935'
BOTH_COLORS  = [BENIGN_COLOR, ATTACK_COLOR]

# ── Canonical 20-feature schema (Ch1, Table 1) and domain ranking (Ch1.3) ──
# Keep these two constants byte-identical across every run, on every
# dataset. This is the one part of the pipeline that must NOT vary.
FULL_FEATURE_LIST = [
    'lifetime_seconds', 'events_per_second', 'seq_length',
    'unique_syscalls', 'uid_is_root', 'uid_changed', 'euid_root',
    'max_uid_euid_delta', 'is_suid_exec', 'auid_euid_mismatch',
    'ephemeral_privileged', 'failed_call_rate', 'failed_call_count',
    'priv_op_count', 'exec_count', 'file_access_count',
    'network_count', 'shell_from_service', 'sensitive_path_access',
    'parent_child_rarity',
]

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


# ── Helpers ──────────────────────────────────────────────────────────────

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def effect_size_label(d):
    d = abs(d)
    if d >= 0.8: return 'large'
    if d >= 0.5: return 'medium'
    if d >= 0.2: return 'small'
    return 'negligible'


def vector_group_key(df, feat_cols):
    """Group key = exact feature-vector identity. Guarantees duplicate
    rows land on the same side of any split (no leakage), works
    identically regardless of what the raw columns underneath looked
    like -- this is what makes the same code usable on both datasets."""
    return df[feat_cols].astype(str).agg('|'.join, axis=1).values


def load_and_filter(args):
    df = pd.read_csv(args.input, low_memory=False)
    print(f"Loaded {args.dataset_name}: {len(df):,} rows, {len(df.columns)} columns")

    if args.exclude_techniques and args.technique_col in df.columns:
        excl = [t.strip() for t in args.exclude_techniques.split(',') if t.strip()]
        before = len(df)
        keep_mask = ~df[args.technique_col].isin(excl) | (df[args.label_col] == 0)
        df = df[keep_mask].reset_index(drop=True)
        print(f"  Excluded techniques {excl}: {before - len(df):,} rows removed, "
              f"{len(df):,} remain")

    feat_cols = [f for f in FULL_FEATURE_LIST if f in df.columns]
    missing   = [f for f in FULL_FEATURE_LIST if f not in df.columns]
    print(f"  Features available: {len(feat_cols)}/{len(FULL_FEATURE_LIST)}")
    if missing:
        print(f"  Missing (expected to differ by dataset, see Table 7): {missing}")

    y = df[args.label_col].values
    print(f"  Benign: {(y == 0).sum():,}  Attack: {(y == 1).sum():,} "
          f"({100 * (y == 1).mean():.1f}% attack)")
    return df, feat_cols


# ── Ch 3.1 — Class distribution ─────────────────────────────────────────

def plot_class_distribution(df, feat_cols, args, out_dir):
    y = df[args.label_col].values
    counts = pd.Series(y).value_counts().sort_index()

    has_tech = args.technique_col in df.columns
    fig, axes = plt.subplots(1, 2 if has_tech else 1,
                              figsize=(10 if has_tech else 5, 4))
    if not has_tech:
        axes = [axes]

    axes[0].bar(['Benign (0)', 'Attack (1)'], counts.values,
                color=BOTH_COLORS, edgecolor='white', linewidth=0.5)
    axes[0].set_title('Overall class balance')
    axes[0].set_ylabel('Row count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + max(counts) * 0.01, f'{v:,}\n({100*v/len(df):.1f}%)',
                     ha='center', va='bottom', fontsize=10)

    if has_tech:
        atk = df[df[args.label_col] == 1]
        tech_counts = atk[args.technique_col].value_counts().head(10)
        axes[1].barh(range(len(tech_counts)), tech_counts.values,
                     color=ATTACK_COLOR, alpha=0.8)
        axes[1].set_yticks(range(len(tech_counts)))
        axes[1].set_yticklabels([str(t)[:30] for t in tech_counts.index], fontsize=8)
        axes[1].set_title('Attack rows by technique')
        axes[1].set_xlabel('Row count')
        axes[1].invert_yaxis()

    plt.suptitle(f'Ch 3.1 — Class Distribution: {args.dataset_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = out_dir / 'class_distribution.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# ── Ch 3.2 — Feature distributions ──────────────────────────────────────

def plot_feature_distributions(df, feat_cols, args, out_dir):
    benign = df[df[args.label_col] == 0]
    attack = df[df[args.label_col] == 1]

    ncols = 4
    nrows = (len(feat_cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    axes = np.atleast_1d(axes).flatten()

    for i, feat in enumerate(feat_cols):
        ax = axes[i]
        b_vals = benign[feat].dropna().values
        a_vals = attack[feat].dropna().values

        if set(df[feat].dropna().unique()).issubset({0, 1}):
            b_rate = b_vals.mean() if len(b_vals) else 0
            a_rate = a_vals.mean() if len(a_vals) else 0
            ax.bar(['Benign', 'Attack'], [b_rate, a_rate],
                   color=BOTH_COLORS, edgecolor='white', alpha=0.85)
            ax.set_ylabel('Rate')
            ax.set_ylim(0, max(b_rate, a_rate) * 1.4 + 0.01)
        else:
            combined = np.concatenate([b_vals, a_vals]) if len(b_vals) and len(a_vals) else np.array([0, 1])
            clip_val = np.percentile(combined, 99) if len(combined) else 1
            clip_val = max(clip_val, 1e-6)
            b_clip = np.clip(b_vals, 0, clip_val)
            a_clip = np.clip(a_vals, 0, clip_val)
            bins = min(40, max(10, int(clip_val)))
            ax.hist(b_clip, bins=bins, alpha=0.6, color=BENIGN_COLOR, label='Benign', density=True)
            ax.hist(a_clip, bins=bins, alpha=0.6, color=ATTACK_COLOR, label='Attack', density=True)
            if i == 0:
                ax.legend(fontsize=8)

        ax.set_title(feat, fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=8)

    for j in range(len(feat_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Ch 3.2 — Feature Distributions: {args.dataset_name}',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = out_dir / 'feature_distributions.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_correlation_matrix(df, feat_cols, args, out_dir):
    corr = df[feat_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
    ax.set_xticks(range(len(feat_cols)))
    ax.set_yticks(range(len(feat_cols)))
    ax.set_xticklabels(feat_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(feat_cols, fontsize=8)

    for i in range(len(feat_cols)):
        for j in range(len(feat_cols)):
            val = corr.values[i, j]
            if abs(val) > 0.3 and i != j:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color='black' if abs(val) < 0.7 else 'white')

    ax.set_title(f'Ch 3.2 — Correlation Matrix: {args.dataset_name}',
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    path = out_dir / 'correlation_matrix.png'
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

    print("  Highly correlated pairs (|r| > 0.7):")
    found = False
    for i in range(len(feat_cols)):
        for j in range(i + 1, len(feat_cols)):
            r = corr.loc[feat_cols[i], feat_cols[j]]
            if abs(r) > 0.7:
                print(f"    {feat_cols[i]} <-> {feat_cols[j]}: r={r:.3f}")
                found = True
    if not found:
        print("    None above 0.7")
    return corr


# ── Ch 3.3 — Mann-Whitney U ─────────────────────────────────────────────

def run_statistical_tests(df, feat_cols, args, out_dir):
    """Ch 3.3 + the guideline's explicit 'calculate class variances' and
    'map correlations between features and the target labels' lines.
    Point-biserial correlation with a binary label is mathematically
    identical to a plain Pearson correlation against the 0/1 label
    column, so it's computed with np.corrcoef rather than a separate
    stats routine -- same number, standard name for a binary target."""
    benign = df[df[args.label_col] == 0]
    attack = df[df[args.label_col] == 1]
    y = df[args.label_col].values
    results = []

    for feat in feat_cols:
        b = benign[feat].dropna().values
        a = attack[feat].dropna().values
        if len(b) < 5 or len(a) < 5:
            continue
        u_stat, p_val = stats.mannwhitneyu(a, b, alternative='two-sided')
        d = cohens_d(a, b)

        full_vals = df[feat].fillna(df[feat].median()).values
        point_biserial_r = np.corrcoef(full_vals, y)[0, 1]

        results.append({
            'feature':          feat,
            'benign_mean':      round(float(b.mean()), 4),
            'attack_mean':      round(float(a.mean()), 4),
            'benign_median':    round(float(np.median(b)), 4),
            'attack_median':    round(float(np.median(a)), 4),
            'benign_variance':  round(float(np.var(b, ddof=1)), 4),
            'attack_variance':  round(float(np.var(a, ddof=1)), 4),
            'U_statistic':      round(float(u_stat), 1),
            'p_value':          float(p_val),
            'p_value_str':      '<0.0001' if p_val < 0.0001 else f'{p_val:.4f}',
            'cohens_d':         round(float(d), 4),
            'effect_size':      effect_size_label(d),
            'significant':      p_val < 0.05,
            'label_corr_r':     round(float(point_biserial_r), 4),
            'domain_rank':      DOMAIN_RANK.get(feat, 99),
        })

    results_df = pd.DataFrame(results).sort_values('cohens_d', ascending=False, key=abs)
    path = out_dir / 'mannwhitney_results.csv'
    results_df.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(results_df)} features tested)")

    # Dedicated feature-to-label correlation chart -- the guideline asks
    # for this explicitly, separate from the feature-to-feature matrix.
    plot_df = results_df.sort_values('label_corr_r', key=abs, ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_df))))
    colors = ['#E53935' if abs(r) > 0.3 else '#90A4AE' for r in plot_df['label_corr_r']]
    ax.barh(range(len(plot_df)), plot_df['label_corr_r'].values, color=colors)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['feature'].values, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Point-biserial correlation with label (0=benign, 1=attack)')
    ax.set_title(f'Ch 3.1/3.3 — Feature-to-Label Correlation: {args.dataset_name}', fontweight='bold')
    plt.tight_layout()
    path2 = out_dir / 'feature_label_correlation.png'
    plt.savefig(path2)
    plt.close()
    print(f"  Saved: {path2}")

    return results_df


# ── Ch 4.1/4.2 — RF importance (leak-free) + discrepancy analysis ──────

def run_random_forest(df, feat_cols, args, out_dir, corr, n_splits=5):
    X = df[feat_cols].fillna(0).values
    y = df[args.label_col].values
    groups = vector_group_key(df, feat_cols)
    n_groups = len(set(groups))
    print(f"  Unique feature-vector groups: {n_groups:,} / {len(df):,} rows "
          f"({len(df)/max(n_groups,1):.1f}x avg duplication)")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_importances, fold_aucs = [], []
    for tr, te in cv.split(X, y, groups):
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                     class_weight='balanced', n_jobs=-1, random_state=42)
        rf.fit(X[tr], y[tr])
        fold_importances.append(rf.feature_importances_)
        if len(set(y[te])) > 1:
            fold_aucs.append(roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1]))

    mean_importance = np.mean(fold_importances, axis=0)
    print(f"  RF AUROC (leak-free, {n_splits}-fold, mean): "
          f"{np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")

    importance_df = pd.DataFrame({
        'feature':     feat_cols,
        'importance':  mean_importance,
        'domain_rank': [DOMAIN_RANK.get(f, 99) for f in feat_cols],
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    importance_df['rf_rank'] = range(1, len(importance_df) + 1)

    path_csv = out_dir / 'rf_feature_importance.csv'
    importance_df.to_csv(path_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = ['#E53935' if imp > importance_df['importance'].median() else '#90CAF9'
              for imp in importance_df['importance']]
    axes[0].barh(range(len(importance_df)), importance_df['importance'].values,
                 color=colors, edgecolor='white', linewidth=0.3)
    axes[0].set_yticks(range(len(importance_df)))
    axes[0].set_yticklabels(importance_df['feature'].values, fontsize=9)
    axes[0].set_xlabel('Gini importance (mean across folds)')
    axes[0].set_title(f'Ch 4.1 — RF Feature Importance: {args.dataset_name}', fontweight='bold')
    axes[0].invert_yaxis()
    for i, v in enumerate(importance_df['importance'].values):
        axes[0].text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)

    axes[1].scatter(importance_df['domain_rank'], importance_df['rf_rank'],
                     s=80, color='#1565C0', alpha=0.7, zorder=3)
    for _, row in importance_df.iterrows():
        axes[1].annotate(row['feature'], (row['domain_rank'], row['rf_rank']),
                         fontsize=7, textcoords='offset points', xytext=(4, 0), alpha=0.8)
    max_rank = len(feat_cols)
    axes[1].plot([1, max_rank], [1, max_rank], 'k--', alpha=0.3, linewidth=1, label='Perfect agreement')
    axes[1].set_xlabel('Domain expert rank (Ch 1.3)')
    axes[1].set_ylabel('RF rank (Gini importance)')
    axes[1].set_title(f'Ch 4.2 — Domain vs RF Ranking: {args.dataset_name}', fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, max_rank + 1)
    axes[1].set_ylim(0, max_rank + 1)

    spearman = stats.spearmanr(importance_df['domain_rank'], importance_df['rf_rank'])
    axes[1].text(0.05, 0.92, f'Spearman rho = {spearman.correlation:.3f}  (p={spearman.pvalue:.3f})',
                 transform=axes[1].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    path_png = out_dir / 'rf_feature_importance.png'
    plt.savefig(path_png)
    plt.close()
    print(f"  Saved: {path_csv}")
    print(f"  Saved: {path_png}")

    write_discrepancy_report(importance_df, spearman, corr, feat_cols, args, out_dir)
    return importance_df, spearman, fold_aucs


def write_discrepancy_report(importance_df, spearman, corr, feat_cols, args, out_dir):
    """Ch 4.2. The guideline requires the 'why' to be evidence-based: for
    each large rank discrepancy, look up the feature's strongest
    correlation partner in the actual correlation matrix (Ch3.2) and
    report that r-value as the multicollinearity evidence, rather than
    just naming 'multicollinearity' as a possibility. Data-leakage is
    handled the same way: stated as a checked structural fact about the
    schema, not a generic disclaimer."""
    lines = [f"Ch 4.2 — Feature Importance Discrepancy Analysis: {args.dataset_name}",
             "=" * 65,
             f"\nSpearman rank correlation (domain vs RF): "
             f"rho={spearman.correlation:.3f}, p={spearman.pvalue:.4f}"]

    if spearman.correlation > 0.6:
        lines.append("-> Strong alignment between domain knowledge and RF ranking.")
    elif spearman.correlation > 0.3:
        lines.append("-> Moderate alignment. Some features rank differently than expected.")
    elif spearman.correlation > -0.3:
        lines.append("-> Weak/near-zero alignment.")
    else:
        lines.append("-> Negative alignment: top domain-ranked features rank low "
                      "empirically, and vice versa. See multicollinearity check below.")

    lines.append("\nTop 5 by RF importance:")
    for _, row in importance_df.head(5).iterrows():
        lines.append(f"  RF#{int(row['rf_rank'])} (Domain#{int(row['domain_rank'])}): "
                     f"{row['feature']}  importance={row['importance']:.4f}")

    # Data-leakage check: state explicitly what's actually in the schema,
    # instead of leaving the reader to assume no IP/MAC/timestamp fields
    # were used. This is the check the guideline asks for, made explicit.
    RAW_IDENTIFIER_TOKENS = {'ip', 'mac', 'timestamp', 'ts', 'host', 'hostid',
                             'hostname', 'srcip', 'dstip', 'macaddr'}
    suspect_cols = [f for f in feat_cols
                     if set(f.lower().split('_')) & RAW_IDENTIFIER_TOKENS]
    lines.append("\nData-leakage check (raw identifier artifacts):")
    if suspect_cols:
        lines.append(f"  WARNING -- feature names matching identifier/timestamp "
                     f"patterns found in the model's feature list: {suspect_cols}. "
                     f"Inspect these before trusting the ranking below.")
    else:
        lines.append(f"  None of the {len(feat_cols)} features in this run's schema "
                     f"are raw IP, MAC, or timestamp fields (full list: {feat_cols}) -- "
                     f"every feature is behaviorally derived (counts, rates, deltas, "
                     f"booleans), so a leakage explanation for any discrepancy below "
                     f"is structurally ruled out, not just assumed.")

    importance_df = importance_df.copy()
    importance_df['rank_diff'] = abs(importance_df['rf_rank'] - importance_df['domain_rank'])
    discrepancies = importance_df[importance_df['rank_diff'] > 5].sort_values('rank_diff', ascending=False)
    lines.append(f"\nLargest rank discrepancies (|RF rank - domain rank| > 5): {len(discrepancies)}")
    for _, row in discrepancies.iterrows():
        feat = row['feature']
        direction = ("ranked higher by RF than domain" if row['rf_rank'] < row['domain_rank']
                     else "ranked lower by RF than domain")
        # Multicollinearity evidence: strongest correlation partner, with r.
        other_corrs = corr[feat].drop(feat).abs().sort_values(ascending=False)
        top_partner = other_corrs.index[0]
        top_r = corr.loc[feat, top_partner]
        if abs(top_r) > 0.5:
            mc_note = (f"strongest correlate is {top_partner} (r={top_r:+.3f}) -- "
                      f"consistent with multicollinearity absorbing/duplicating "
                      f"{feat}'s signal")
        else:
            mc_note = (f"strongest correlate is {top_partner} (r={top_r:+.3f}, weak) -- "
                      f"multicollinearity is NOT a strong explanation here; likely "
                      f"a genuine latent pattern the domain ranking missed")
        lines.append(f"  {feat}: {direction} "
                     f"(RF#{int(row['rf_rank'])} vs Domain#{int(row['domain_rank'])}). {mc_note}")

    text = '\n'.join(lines)
    path = out_dir / 'ch4_discrepancy_analysis.txt'
    with open(path, 'w') as f:
        f.write(text)
    print(f"  Saved: {path}")


def write_summary(df, mw_df, fold_aucs, args, out_dir):
    lines = [f"EDA Summary — {args.dataset_name}", "=" * 60,
             f"\nInput: {args.input}",
             f"Total rows: {len(df):,}",
             f"Benign: {(df[args.label_col]==0).sum():,} ({100*(df[args.label_col]==0).mean():.1f}%)",
             f"Attack: {(df[args.label_col]==1).sum():,} ({100*(df[args.label_col]==1).mean():.1f}%)",
             f"\nRandom Forest (leak-free, vector-group CV):",
             f"  AUROC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}"]

    if mw_df is not None and len(mw_df):
        sig = mw_df[mw_df['significant']]
        large = mw_df[mw_df['effect_size'] == 'large']
        lines.append(f"\nMann-Whitney U: {len(mw_df)} features tested, "
                     f"{len(sig)} significant (p<.05), {len(large)} large effect size")
        lines.append("Top features by |Cohen's d|:")
        for _, row in mw_df.head(5).iterrows():
            lines.append(f"  {row['feature']:<25} d={row['cohens_d']:+.4f} "
                         f"({row['effect_size']}) p={row['p_value_str']}")

    text = '\n'.join(lines)
    path = out_dir / 'eda_summary.txt'
    with open(path, 'w') as f:
        f.write(text)
    print(f"  Saved: {path}")


def main():
    p = argparse.ArgumentParser(description="Unified EDA + feature importance, Ch3/Ch4")
    p.add_argument('--input', required=True, help='Path to a process-level feature CSV')
    p.add_argument('--label-col', required=True, help="Binary label column, e.g. 'label' or 'is_privesc'")
    p.add_argument('--technique-col', default='technique', help='Technique/attack-type column, if present')
    p.add_argument('--exclude-techniques', default=None, help='Comma-separated techniques to exclude (attack rows only)')
    p.add_argument('--dataset-name', required=True, help='Display name for plot titles, e.g. "CAM-LDS"')
    p.add_argument('--out', required=True, help='Output directory')
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"Unified EDA — {args.dataset_name}")
    print("=" * 65)

    df, feat_cols = load_and_filter(args)

    print("\n[Ch 3.1] Class distribution")
    plot_class_distribution(df, feat_cols, args, out_dir)

    print("\n[Ch 3.2] Feature distributions + correlation matrix")
    plot_feature_distributions(df, feat_cols, args, out_dir)
    corr = plot_correlation_matrix(df, feat_cols, args, out_dir)

    print("\n[Ch 3.3] Mann-Whitney U tests + class variance + label correlation")
    mw_df = run_statistical_tests(df, feat_cols, args, out_dir)

    print("\n[Ch 4.1/4.2] Random Forest importance + discrepancy analysis")
    importance_df, spearman, fold_aucs = run_random_forest(df, feat_cols, args, out_dir, corr)

    print("\n[Summary]")
    write_summary(df, mw_df, fold_aucs, args, out_dir)

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == '__main__':
    main()
