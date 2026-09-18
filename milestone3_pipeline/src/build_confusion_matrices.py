"""
Milestone 3 -- closes the confusion-matrix TODO in Part 2, Step 8.

confusion_matrix() is already called internally in train_eval.py and
cross_dataset_eval.py, but its output is never saved anywhere -- the
report currently satisfies the "deconstruct confusion matrices" rubric
line only implicitly, via scattered FN-rate charts and FP-feature CSVs.

This script does NOT run a new CV pass. It reuses error_analysis.py's
get_oof_predictions() -- the exact same pooled, leak-free out-of-fold
predictions already backing every other §2.1 number (technique FN rates,
FP/FN feature comparisons, pairwise disagreement) -- and simply computes
and SAVES sklearn's confusion_matrix() on those same predictions, once
per model, per dataset. This is deliberately the "single consistent run"
the TODO asks for: it's the same run already trusted for everything else
in this section, not a fresh, potentially-inconsistent one.

Covers the four headline paradigm representatives (RF, XGBoost, MLP,
Isolation Forest) -- CNN is intentionally excluded, matching how the rest
of §2.1 already treats CNN as the cascade's Stage-2 model rather than a
headline representative post-swap.

Usage:
    python build_confusion_matrices.py --dataset camlds --data-dir ../data \
        --out results/current/confusion_matrices/
    python build_confusion_matrices.py --dataset casino --data-dir ../data \
        --out results/current/confusion_matrices/
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

from error_analysis import get_oof_predictions

HEADLINE_MODELS = ["rf", "xgb", "mlp", "iforest"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--feature-policy", default="full")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[confusion_matrices] dataset={args.dataset} models={HEADLINE_MODELS}", flush=True)
    ds, feat_cols, X, y, results = get_oof_predictions(
        args.dataset, args.data_dir, feature_policy=args.feature_policy,
        models=HEADLINE_MODELS)

    rows = []
    for model_key in HEADLINE_MODELS:
        if model_key not in results:
            print(f"  SKIPPED {model_key} -- not present in get_oof_predictions() output "
                  f"(check the exact key name/shape returned by error_analysis.py "
                  f"before trusting this table is complete)")
            continue

        model_result = results[model_key]
        # Confirmed via debug_shapes.py: get_oof_predictions() returns, per
        # model, a (y_pred, y_proba) tuple -- index 0 is hard labels (int64,
        # 0/1), index 1 is predicted probabilities (float64). Verified for
        # all four headline models on casino; same extraction is used for
        # camlds since get_oof_predictions() returns the same shape per the
        # function contract, not a per-dataset one.
        if hasattr(model_result, "pred"):
            y_pred = model_result.pred
        elif isinstance(model_result, dict) and "pred" in model_result:
            y_pred = model_result["pred"]
        elif isinstance(model_result, (tuple, list)) and len(model_result) == 2:
            y_pred, _y_proba = model_result
        else:
            y_pred = model_result  # assume it's already the prediction array

        tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "dataset": args.dataset, "model": model_key,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "n_total": int(len(y)),
            "precision": tp / (tp + fp) if (tp + fp) > 0 else float("nan"),
            "recall": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else float("nan"),
        })
        print(f"  {model_key:<8} TN={tn:<7} FP={fp:<7} FN={fn:<7} TP={tp:<7}")

    table = pd.DataFrame(rows)
    csv_path = out_dir / f"{args.dataset}_confusion_matrices.csv"
    table.to_csv(csv_path, index=False)
    print(f"\n[confusion_matrices] Saved: {csv_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()