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

# Shape of the client's parse_error return; reused to build a call_error record.
_EMPTY_RESULT = {"verdict": None, "confidence": None, "rationale_steps": [],
                 "key_features": [], "agrees_with": None}


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
        "raw": llm_result.get("raw"),
    }


def arbitrate_dataset(dataset_key, data_dir, out_dir, model_id=None, backend=None,
                      max_cases=None, seed=42, client=None):
    from llm_triage_dump import find_contradictions, stratified_subsample
    from llm_triage_stats import compute_percentile_tables
    from llm_triage_context import build_context
    from llm_triage_client import LLMClient
    from feature_selection import select_features
    from ingestion import ingest

    model_id = model_id or LLM_TRIAGE["model_id"]
    max_cases = max_cases or LLM_TRIAGE["max_cases_per_dataset"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One ingest for the whole call -- reused by both branches below and by the
    # percentile tables.
    ds = ingest(dataset_key, data_dir=data_dir)

    dump_csv = out_dir / f"{dataset_key}_contradictions.csv"
    oof_csv = out_dir / f"{dataset_key}_cascade_oof.csv"
    if dump_csv.exists() and oof_csv.exists():
        contra = pd.read_csv(dump_csv)
        feat_cols = select_features(ds.feat_cols, dataset_key, policy="full")
        contra = stratified_subsample(contra, max_cases, seed)  # no-op when under cap
        print(f"[llm_triage_arbitrate] reusing {dump_csv} + {oof_csv} "
              f"({len(contra)} contradiction rows)", flush=True)
    else:
        contra, _oof, feat_cols = find_contradictions(dataset_key, data_dir, seed=seed)
        contra = stratified_subsample(contra, max_cases, seed)

    label_col = DATASETS[dataset_key]["label_col"]
    pct = compute_percentile_tables(ds.df, feat_cols, label_col)
    (out_dir / f"{dataset_key}_percentiles.json").write_text(json.dumps(pct))

    path = out_dir / f"{dataset_key}_{_modeltag(model_id)}_llm_arbitration.csv"

    # Resume: seed `rows` from an existing partial CSV and skip its row_ids.
    rows, done_ids = [], set()
    if path.exists():
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done_ids = set(prev["row_id"].astype(int)) if "row_id" in prev.columns else set()
        print(f"[llm_triage_arbitrate] resuming: {len(done_ids)} rows already in {path}",
              flush=True)

    client = client or LLMClient(model_id=model_id, backend=backend)
    error_logged = False
    for i, (_, r) in enumerate(contra.iterrows()):
        if int(r["row_id"]) in done_ids:
            continue
        ctx = build_context(r, feat_cols, pct, dataset_key)
        try:
            res = client.arbitrate(ctx)
        except Exception as e:  # a single bad call must not abort the batch
            res = dict(_EMPTY_RESULT)
            res.update({"parse_status": "call_error", "raw": repr(e)})
            if not error_logged:
                print(f"[llm_triage_arbitrate] first call_error (row_id={int(r['row_id'])}): "
                      f"{repr(e)}", flush=True)
                error_logged = True
        rows.append(assemble_row(r, feat_cols, res))
        if (i + 1) % 25 == 0:
            pd.DataFrame(rows).to_csv(path, index=False)  # incremental checkpoint
            print(f"[llm_triage_arbitrate] {dataset_key}/{_modeltag(model_id)}: "
                  f"{i + 1}/{len(contra)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    n_err = int((df["parse_status"] == "parse_error").sum()) if len(df) else 0
    n_call_err = int((df["parse_status"] == "call_error").sum()) if len(df) else 0
    print(f"[llm_triage_arbitrate] wrote {path}  ({len(df)} rows, "
          f"{n_err} parse_error, {n_call_err} call_error)", flush=True)
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
