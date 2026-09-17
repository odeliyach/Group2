"""
Milestone 3 -- M2 reviewer point 1 follow-up: does an availability/missingness
indicator carry usable signal WHEN it actually has variance to learn from?

verify_unified_schema_missingness.py's CAM-LDS->Casino test found no effect
(delta AUROC ~=0.0000), but that test's `camlds_only_features_available`
indicator is a DATASET-LEVEL CONSTANT: 1 for every single CAM-LDS training
row, 0 for every single Casino test row. A tree-based model trained on
CAM-LDS alone has ZERO within-training-fold variance on that column, so it
is structurally unable to learn a split on it -- regardless of whether the
underlying signal would be useful. That experiment could only ever report
"no effect," by construction; it tested the cross-dataset transfer problem,
not the missingness-indicator idea itself.

This script isolates the question. Stay within CAM-LDS alone (where all 20
features are real for every row), and SYNTHETICALLY mask a random subset of
rows' is_suid_exec / ephemeral_privileged / parent_child_rarity (0-fill, this
project's existing convention), with an availability indicator that now
varies WITHIN every training and test fold. Same leak-free repeated_cv
engine as every other headline number in this pipeline (train_eval.py) --
no pipeline code touched.

Three arms, identical folds (same StratifiedGroupKFold split reused across
arms since y/groups are unaffected by masking, isolating the encoding as the
only variable):
  A. full_info         -- original 20 real CAM-LDS features, nothing masked.
  B. masked_no_flag     -- the 3 features 0-filled for a random row subset,
                            NO availability indicator (naive zero-fill).
  C. masked_with_flag   -- same masking as B, PLUS the availability
                            indicator (0/1, now row-varying) as a 21st
                            feature.

Read the result as:
  C close to A, and clearly above B  -> the model genuinely exploits the
      indicator to recover from missing information when it has real
      variance to learn from. The original cross-dataset null result would
      then look like a variance artifact of that experiment's design, and
      re-extracting the 3 features for Casino (or a per-row missingness
      scheme, if either dataset ever has one) would be worth prioritizing.
  C close to B                       -> the indicator carries no usable
      signal even WITH row-level variance -- a materially stronger negative
      result than the cross-dataset test could produce, since it rules out
      "no variance to learn from" as the explanation.

Usage:
    python synthetic_missingness_ablation.py --data-dir data \
        --model rf --mask-frac 0.5 --repeats 15 \
        --out results/current/synthetic_missingness_ablation/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion import ingest
from feature_selection import select_features
from preprocessing import vector_group_key, log1p_continuous
from train_eval import repeated_cv
from config import LOG1P_CONTINUOUS_FEATURES, BINARY_FEATURES

CAMLDS_ONLY_FEATURES = ['is_suid_exec', 'ephemeral_privileged', 'parent_child_rarity']
AVAILABILITY_COL = 'synthetic_features_available'


def build_arms(df, feat_cols, mask_frac, seed):
    """Returns (X_full_df, X_masked_df, X_flagged_df, masked) -- the three
    arms' feature frames plus the row mask itself, all sharing ONE random
    draw so a given row is masked (or not) identically across arms B and C,
    keeping the comparison fold-for-fold apples-to-apples."""
    rng = np.random.RandomState(seed)
    n = len(df)
    masked = rng.random(n) < mask_frac  # True = this row's 3 features are hidden

    X_full = df[feat_cols].copy()

    X_masked = X_full.copy()
    X_masked.loc[masked, CAMLDS_ONLY_FEATURES] = 0.0  # same 0-fill convention as
                                                        # verify_unified_schema_missingness.py

    X_flagged = X_masked.copy()
    X_flagged[AVAILABILITY_COL] = (~masked).astype(float)  # 1 = real values, 0 = masked

    return X_full, X_masked, X_flagged, masked


def _prep(X_df, cols, extra_binary=frozenset()):
    X = X_df[cols].values.astype(float)
    if LOG1P_CONTINUOUS_FEATURES:
        X = log1p_continuous(X, cols, BINARY_FEATURES | set(extra_binary))
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["rf", "xgb"], default="rf",
                    help="Tree-based only -- matches verify_unified_schema_missingness.py's "
                         "choice of models tolerant of a 0-filled/flag encoding")
    ap.add_argument("--mask-frac", type=float, default=0.5,
                    help="Fraction of CAM-LDS rows whose 3 features get synthetically hidden")
    ap.add_argument("--repeats", type=int, default=15,
                    help="15 matches this project's headline protocol; lower for a quick check")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ingest("camlds", data_dir=args.data_dir)
    feat_cols = select_features(ds.feat_cols, "camlds", policy="full")
    for f in CAMLDS_ONLY_FEATURES:
        assert f in feat_cols, f"{f} missing from CAM-LDS's own feature list"

    X_full_df, X_masked_df, X_flagged_df, masked = build_arms(
        ds.df, feat_cols, args.mask_frac, args.seed)
    # Grouping key computed on the REAL feature values (unaffected by the
    # synthetic mask) so all three arms use the identical leak-free split.
    groups = vector_group_key(ds.df, feat_cols)
    y = ds.y.values
    print(f"[synthetic_missingness] CAM-LDS: {len(ds.df)} rows, "
          f"{int(masked.sum())} ({masked.mean():.1%}) synthetically masked "
          f"(mask_frac={args.mask_frac})", flush=True)

    arms = {
        "A_full_info": _prep(X_full_df, feat_cols),
        "B_masked_no_flag": _prep(X_masked_df, feat_cols),
        "C_masked_with_flag": _prep(
            X_flagged_df, feat_cols + [AVAILABILITY_COL], extra_binary={AVAILABILITY_COL}),
    }

    rows = []
    for arm_name, X in arms.items():
        print(f"\n[synthetic_missingness] arm={arm_name} model={args.model}", flush=True)
        cv_df = repeated_cv(X, y, groups, args.model, repeats=args.repeats, seed=args.seed)
        rows.append({
            "arm": arm_name, "model": args.model, "mask_frac": args.mask_frac,
            "f1_mean": cv_df["f1"].mean(), "f1_std": cv_df["f1"].std(),
            "auroc_mean": cv_df["auroc"].mean(), "auroc_std": cv_df["auroc"].std(),
            "fpr_mean": cv_df["fpr"].mean(),
        })

    summary = pd.DataFrame(rows)
    csv_path = out_dir / f"camlds_{args.model}_synthetic_missingness_maskfrac{args.mask_frac}.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\n[synthetic_missingness] Saved: {csv_path}")
    print(summary.to_string(index=False))

    a_f1 = summary.loc[summary.arm == "A_full_info", "f1_mean"].iloc[0]
    b_f1 = summary.loc[summary.arm == "B_masked_no_flag", "f1_mean"].iloc[0]
    c_f1 = summary.loc[summary.arm == "C_masked_with_flag", "f1_mean"].iloc[0]
    gap = a_f1 - b_f1
    recovered = (c_f1 - b_f1) / gap if abs(gap) > 1e-9 else float("nan")

    if c_f1 - b_f1 > 0.01 and (np.isnan(recovered) or recovered > 0.3):
        verdict = ("INDICATOR HELPS -- C recovers a meaningful share of the A-vs-B gap; "
                   "the model exploits the availability flag when it has real row-level "
                   "variance to learn from. This makes re-extracting the 3 features for "
                   "Casino (or an analogous per-row missingness scheme) worth prioritizing.")
    elif abs(c_f1 - b_f1) < 0.01:
        verdict = ("INDICATOR DOES NOT HELP -- C ~= B even with genuine row-level variance. "
                   "The earlier cross-dataset null result was NOT merely a variance artifact "
                   "of that experiment's design; the availability signal itself appears to "
                   "carry no usable information for this model/feature set.")
    else:
        verdict = "AMBIGUOUS -- see full numbers; C falls between B and A without clearly tracking either."

    print(f"\nA(full)={a_f1:.4f}  B(masked, no flag)={b_f1:.4f}  C(masked, with flag)={c_f1:.4f}")
    if not np.isnan(recovered):
        print(f"Gap recovered by flag: {recovered:.1%}")
    print(f"VERDICT: {verdict}")

    verdict_lines = [
        f"A(full)={a_f1:.4f}  B(masked,no flag)={b_f1:.4f}  C(masked,with flag)={c_f1:.4f}",
    ]
    if not np.isnan(recovered):
        verdict_lines.append(f"Gap recovered by flag: {recovered:.1%}")
    verdict_lines.append(f"VERDICT: {verdict}")
    (out_dir / f"{args.model}_maskfrac{args.mask_frac}_verdict.txt").write_text(
        "\n".join(verdict_lines) + "\n")
    print(f"\nSaved: {out_dir}/")


if __name__ == "__main__":
    main()
