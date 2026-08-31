"""Milestone 3 -- Step 8 capstone: evaluate LLM arbitration (Deliverable 4).

Reads one <ds>_<model>_llm_arbitration.csv (Task 6) plus <ds>_cascade_oof.csv
(Task 4) and reports:
  A. arbitration accuracy on the contradiction set vs 5 baselines
  B. whole-population pipeline effect (cascade blend@0.5 vs cascade+LLM)
  D. confidence calibration (ECE) and key_features overlap with error-analysis
     discriminators
  E. parse-failure rate
Charts + a paste-ready report_section markdown.

Note: baseline B uses a fixed 0.5 blend cutoff (labelled 'blend@0.5'); the live
cascade picks its cutoff per fold from XGBoost train probabilities, so this is a
close, honest approximation, not the exact cascade decision.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)

# Aggregate discriminators from error_analysis.py's FP-vs-TN / FN-vs-TP tables
# (results/current/error_analysis_results/<ds>/). Population-level facts, not
# row-level -- safe to show the model as "known failure patterns".
DISCRIMINATOR_FEATURES = {
    "camlds": ["auid_euid_mismatch", "max_uid_euid_delta", "priv_op_count", "network_count",
               "shell_from_service", "parent_child_rarity", "uid_changed", "ephemeral_privileged",
               "failed_call_rate", "failed_call_count", "exec_count", "lifetime_seconds"],
    "casino": ["exec_count", "file_access_count", "euid_root", "uid_is_root", "auid_euid_mismatch",
               "priv_op_count", "failed_call_rate", "shell_from_service", "failed_call_count"],
}


def _fpr(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return fp / (fp + tn) if (fp + tn) else 0.0


def arbitration_accuracy(arb_df):
    d = arb_df[arb_df["parse_status"] != "parse_error"].copy()
    y = d["true_label"].astype(int).values
    px, pc = d["proba_xgb"].values, d["proba_cnn"].values
    more_conf = np.where(np.abs(px - 0.5) >= np.abs(pc - 0.5),
                         d["pred_xgb"].values, d["pred_cnn"].values).astype(int)
    strategies = {
        "llm": d["llm_pred"].astype(int).values,
        "always_xgb": d["pred_xgb"].astype(int).values,
        "always_cnn": d["pred_cnn"].astype(int).values,
        "trust_more_confident": more_conf,
        "blend_at_0.5": ((0.5 * px + 0.5 * pc) >= 0.5).astype(int),
        "majority_class": np.full(len(d), int(round(y.mean())) if len(d) else 0),
    }
    out = []
    for name, pred in strategies.items():
        out.append({
            "strategy": name,
            "accuracy": accuracy_score(y, pred) if len(d) else float("nan"),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "n": int(len(d)),
        })
    return pd.DataFrame(out)


def pipeline_effect(arb_df, oof_df):
    o = oof_df.copy()
    y = o["true_label"].astype(int).values
    base = ((0.5 * o["proba_xgb"] + 0.5 * o["proba_cnn"]) >= 0.5).astype(int).values
    sub = {int(r.row_id): int(r.llm_pred) for r in arb_df.itertuples()
           if int(r.llm_pred) in (0, 1)}
    idx = {rid: i for i, rid in enumerate(o["row_id"].astype(int).values)}
    withllm = base.copy()
    for rid, p in sub.items():
        if rid in idx:
            withllm[idx[rid]] = p
    rows = []
    for name, pred in [("cascade_blend@0.5", base), ("cascade+llm", withllm)]:
        rows.append({"pipeline": name,
                     "f1": f1_score(y, pred, zero_division=0),
                     "recall": recall_score(y, pred, zero_division=0),
                     "fpr": _fpr(y, pred)})
    eff = pd.DataFrame(rows)
    delta = (eff.set_index("pipeline").loc["cascade+llm"]
             - eff.set_index("pipeline").loc["cascade_blend@0.5"])
    return eff, delta


def expected_calibration_error(confidence, correct, n_bins=10):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(confidence) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        if not m.any():
            continue
        ece += (m.sum() / len(confidence)) * abs(correct[m].mean() - confidence[m].mean())
    return float(ece)


def key_feature_overlap(arb_df, discriminator_features):
    disc = set(discriminator_features)
    n = hits = 0
    for s in arb_df["key_features"]:
        kf = set(json.loads(s)) if isinstance(s, str) else set(s or [])
        if not kf:
            continue
        n += 1
        if kf & disc:
            hits += 1
    return hits / n if n else float("nan")


# -- charts ---------------------------------------------------------------
def plot_accuracy_bars(acc_df, title, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(acc_df["strategy"], acc_df["accuracy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy on contradiction set")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_reliability(confidence, correct, path):
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(confidence):
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for i in range(10):
            m = (confidence > edges[i]) & (confidence <= edges[i + 1])
            if m.any():
                xs.append(confidence[m].mean())
                ys.append(correct[m].mean())
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.plot(xs, ys, "o-")
    ax.set_xlabel("mean reported confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("LLM confidence calibration")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report_section(dataset_key, model_tag, acc_df, eff_df, eff_delta, ece,
                         kf_overlap, parse_rate, out_path):
    lines = [
        f"### LLM arbitration -- {dataset_key} / {model_tag}", "",
        f"- Contradiction rows scored: {int(acc_df['n'].iloc[0])} "
        f"(parse-failure rate {parse_rate:.1%})", "",
        "**Accuracy on the contradiction set**", "",
        acc_df.to_markdown(index=False), "",
        "**Whole-population pipeline effect (blend@0.5 baseline)**", "",
        eff_df.to_markdown(index=False), "",
        f"- Delta vs baseline: F1 {eff_delta['f1']:+.4f}, "
        f"recall {eff_delta['recall']:+.4f}, FPR {eff_delta['fpr']:+.4f}", "",
        "**Reasoning quality**", "",
        f"- Expected Calibration Error: {ece:.4f}",
        f"- key_features intersect the dataset's error-analysis discriminators "
        f"in {kf_overlap:.1%} of scored rows", "",
    ]
    Path(out_path).write_text("\n".join(str(x) for x in lines))


def main():
    ap = argparse.ArgumentParser(description="Evaluate LLM arbitration output")
    ap.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    ap.add_argument("--arbitration", required=True, help="<ds>_<model>_llm_arbitration.csv")
    ap.add_argument("--oof", required=True, help="<ds>_cascade_oof.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    arb = pd.read_csv(args.arbitration)
    oof = pd.read_csv(args.oof)
    model_tag = Path(args.arbitration).stem.replace(f"{args.dataset}_", "").replace("_llm_arbitration", "")

    acc = arbitration_accuracy(arb)
    eff, delta = pipeline_effect(arb, oof)
    usable = arb[arb["parse_status"] != "parse_error"].copy()
    correct = (usable["llm_pred"].astype(int) == usable["true_label"].astype(int)).astype(float).values
    ece = expected_calibration_error(usable["llm_confidence"].values, correct)
    kf = key_feature_overlap(arb, DISCRIMINATOR_FEATURES[args.dataset])
    parse_rate = (arb["parse_status"] == "parse_error").mean() if len(arb) else float("nan")

    acc.to_csv(out_dir / f"{args.dataset}_{model_tag}_arbitration_metrics.csv", index=False)
    eff.to_csv(out_dir / f"{args.dataset}_{model_tag}_pipeline_effect.csv", index=False)
    pd.DataFrame([{"ece": ece, "key_feature_overlap": kf, "parse_error_rate": parse_rate}]).to_csv(
        out_dir / f"{args.dataset}_{model_tag}_reasoning_quality.csv", index=False)
    plot_accuracy_bars(acc, f"{args.dataset} / {model_tag}",
                       out_dir / f"{args.dataset}_{model_tag}_accuracy.png")
    plot_reliability(usable["llm_confidence"].values, correct,
                     out_dir / f"{args.dataset}_{model_tag}_calibration.png")
    write_report_section(args.dataset, model_tag, acc, eff, delta, ece, kf, parse_rate,
                         out_dir / f"{args.dataset}_{model_tag}_report_section.md")
    print(f"[llm_triage_evaluate] wrote metrics + charts + report section to {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
