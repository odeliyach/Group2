"""
Milestone 3 -- MAX_FPR validation follow-up.

config.MAX_FPR=0.05 was derived from a Random Forest-only FPR-cap sweep
(operating_curve.py --model rf) and then applied pipeline-wide to XGBoost,
MLP, CNN, and Isolation Forest without checking whether 0.05 is also
near-optimal for each of those models' own F1-vs-FPR curves. This script
does not run anything itself -- operating_curve.py already supports
--model {rf,xgb,cnn,mlp,iforest}, so the sweep itself just needed to be run
for the other four models too (see run_operating_curve_all_models.sh).

This script collects the resulting
  <out_dir>/<dataset>_<model>_operating_curve.csv
files (one per dataset x model, written by operating_curve.py) and reports,
per model x dataset, the FPR cap that actually maximizes F1 -- so you can
state in the report whether 0.05 holds up for every model or only for RF.

Usage:
    python summarize_operating_curves.py --dir results/current/operating_curve_results \
        --out results/current/operating_curve_results/max_fpr_validation_summary.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

_FNAME_RE = re.compile(r"^(?P<dataset>camlds|casino)_(?P<model>rf|xgb|cnn|mlp|iforest)_operating_curve\.csv$")


def load_all_curves(curves_dir):
    rows = []
    for path in sorted(Path(curves_dir).glob("*_operating_curve.csv")):
        m = _FNAME_RE.match(path.name)
        if not m:
            continue
        df = pd.read_csv(path)
        df["dataset"] = m.group("dataset")
        df["model"] = m.group("model")
        rows.append(df)
    if not rows:
        raise FileNotFoundError(
            f"No *_operating_curve.csv files matched in {curves_dir} -- "
            f"run operating_curve.py for each model/dataset first "
            f"(see run_operating_curve_all_models.sh).")
    return pd.concat(rows, ignore_index=True)


def summarize(curves_df, reference_cap=0.05):
    out_rows = []
    for (dataset, model), g in curves_df.groupby(["dataset", "model"]):
        best = g.loc[g["f1_mean"].idxmax()]
        at_ref = g.loc[(g["max_fpr_cap"] - reference_cap).abs().idxmin()]
        out_rows.append({
            "dataset": dataset,
            "model": model,
            "best_cap": best["max_fpr_cap"],
            "best_f1_mean": best["f1_mean"],
            "best_actual_fpr_mean": best["actual_fpr_mean"],
            f"f1_at_cap_{reference_cap}": at_ref["f1_mean"],
            "gap_vs_best_f1": best["f1_mean"] - at_ref["f1_mean"],
            "auroc_mean": best["auroc_mean"],  # threshold-independent, constant across caps
        })
    summary = pd.DataFrame(out_rows).sort_values(["dataset", "model"]).reset_index(drop=True)

    def _verdict(row):
        if row["auroc_mean"] < 0.6:
            return "NOT DISCRIMINATIVE -- no cap fixes this; ignore F1 peak location"
        if row["gap_vs_best_f1"] < 0.01:
            return f"MAX_FPR={reference_cap} HOLDS -- within 0.01 F1 of this model's own best cap"
        return (f"MAX_FPR={reference_cap} SUBOPTIMAL for this model -- best cap is "
               f"{row['best_cap']} (F1 {row['gap_vs_best_f1']:+.3f} better)")
    summary["verdict"] = summary.apply(_verdict, axis=1)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing *_operating_curve.csv files")
    ap.add_argument("--reference-cap", type=float, default=0.05, help="config.MAX_FPR's current value")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    curves_df = load_all_curves(args.dir)
    summary = summarize(curves_df, reference_cap=args.reference_cap)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
