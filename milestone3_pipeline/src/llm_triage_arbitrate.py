"""Milestone 3 -- Step 8 capstone: orchestrate LLM arbitration over the
cascade-contradiction set for one dataset and one model.

    dump contradictions -> build percentile tables -> render context per row
    -> LLMClient.arbitrate -> flat arbitration CSV

Usage:
    python llm_triage_arbitrate.py --dataset camlds --data-dir data/v3 \
        --out results/current/llm_triage/camlds \
        --model fdtn-ai/Foundation-Sec-8B-Instruct --backend vllm
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from config import LLM_TRIAGE, DATASETS


def _modeltag(model_id):
    tail = model_id.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", tail).strip("-")


def assemble_row(contra_row, feat_cols, llm_result):
    verdict = llm_result.get("verdict")
    llm_pred = 1 if verdict == "MALICIOUS" else 0 if verdict == "BENIGN" else -1
    return {
        "row_id": int(contra_row["row_id"]),
        "fold": int(contra_row["fold"]),
        "true_label": int(contra_row["true_label"]),
        "technique": contra_row["technique"],
        "pred_xgb": int(contra_row["pred_xgb"]),
        "pred_cnn": int(contra_row["pred_cnn"]),
        "proba_xgb": float(contra_row["proba_xgb"]),
        "proba_cnn": float(contra_row["proba_cnn"]),
        "disagreement_direction": contra_row["disagreement_direction"],
        "llm_verdict": verdict,
        "llm_pred": llm_pred,
        "llm_confidence": llm_result.get("confidence"),
        "llm_agrees_with": llm_result.get("agrees_with"),
        "parse_status": llm_result.get("parse_status"),
        "rationale_steps": json.dumps(llm_result.get("rationale_steps", [])),
        "key_features": json.dumps(llm_result.get("key_features", [])),
    }


def arbitrate_dataset(dataset_key, data_dir, out_dir, model_id=None, backend=None,
                      max_cases=None, seed=42, client=None):
    from llm_triage_dump import find_contradictions, stratified_subsample
    from llm_triage_stats import compute_percentile_tables
    from llm_triage_context import build_context
    from llm_triage_client import LLMClient
    from ingestion import ingest

    model_id = model_id or LLM_TRIAGE["model_id"]
    max_cases = max_cases or LLM_TRIAGE["max_cases_per_dataset"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contra, _oof, feat_cols = find_contradictions(dataset_key, data_dir, seed=seed)
    contra = stratified_subsample(contra, max_cases, seed)

    ds = ingest(dataset_key, data_dir=data_dir)
    label_col = DATASETS[dataset_key]["label_col"]
    pct = compute_percentile_tables(ds.df, feat_cols, label_col)
    (out_dir / f"{dataset_key}_percentiles.json").write_text(json.dumps(pct))

    client = client or LLMClient(model_id=model_id, backend=backend)
    rows = []
    for i, (_, r) in enumerate(contra.iterrows()):
        ctx = build_context(r, feat_cols, pct, dataset_key)
        res = client.arbitrate(ctx)
        rows.append(assemble_row(r, feat_cols, res))
        if (i + 1) % 25 == 0:
            print(f"[llm_triage_arbitrate] {dataset_key}/{_modeltag(model_id)}: "
                  f"{i + 1}/{len(contra)}", flush=True)

    df = pd.DataFrame(rows)
    path = out_dir / f"{dataset_key}_{_modeltag(model_id)}_llm_arbitration.csv"
    df.to_csv(path, index=False)
    n_err = (df["parse_status"] == "parse_error").sum() if len(df) else 0
    print(f"[llm_triage_arbitrate] wrote {path}  ({len(df)} rows, {n_err} parse_error)",
          flush=True)
    return df


def main():
    ap = argparse.ArgumentParser(description="LLM arbitration over cascade contradictions")
    ap.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=LLM_TRIAGE["model_id"])
    ap.add_argument("--backend", default=LLM_TRIAGE["backend"], choices=["vllm", "ollama"])
    ap.add_argument("--max-cases", type=int, default=LLM_TRIAGE["max_cases_per_dataset"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    arbitrate_dataset(args.dataset, args.data_dir, args.out, model_id=args.model,
                      backend=args.backend, max_cases=args.max_cases, seed=args.seed)


if __name__ == "__main__":
    main()
