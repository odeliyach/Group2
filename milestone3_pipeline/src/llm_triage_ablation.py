import os
import sys
import json
import re
import urllib.request
import pandas as pd
from tqdm import tqdm

# Point directly to the verified scripts directory
SRC_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/scripts/src"
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import llm_triage_prompts as prompts
import llm_triage_context as ctx_builder

ABLATION_DATA = "/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/llm_triage_results/camlds/camlds_contradictions_ablation_100.csv"
PERCENTILES_PATH = "/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/llm_triage_results/camlds/camlds_percentiles.json"
OUT_DIR = "/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/llm_triage_results/camlds"
OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
OLLAMA_URL = f"http://127.0.0.1:{OLLAMA_PORT}/api/generate"
MODEL_NAME = "foundation-sec-8b:latest"
SEED = 42

with open(PERCENTILES_PATH, "r") as f:
    percentiles = json.load(f)

# The evaluated feature list is exactly the keys in the percentiles dictionary
FEAT_COLS = list(percentiles.keys())

FEWSHOT_EXAMPLE_CNN_FIRST = prompts.FEWSHOT_EXAMPLE.replace(
    "Model votes:  XGBoost -> BENIGN (p_malicious=0.41)   |   CNN -> MALICIOUS (p_malicious=0.66)",
    "Model votes:  CNN -> MALICIOUS (p_malicious=0.66)   |   XGBoost -> BENIGN (p_malicious=0.41)"
)

def build_ablation_context(row, feat_cols, pct_tables, variant):
    xgb_call = "MALICIOUS" if int(row["pred_xgb"]) == 1 else "BENIGN"
    cnn_call = "MALICIOUS" if int(row["pred_cnn"]) == 1 else "BENIGN"
    xgb_p = float(row["proba_xgb"])
    cnn_p = float(row["proba_cnn"])

    if variant == "cnn_first":
        votes_str = f"Model votes:  CNN -> {cnn_call} (p_malicious={cnn_p:.2f})   |   XGBoost -> {xgb_call} (p_malicious={xgb_p:.2f})"
    else:
        votes_str = f"Model votes:  XGBoost -> {xgb_call} (p_malicious={xgb_p:.2f})   |   CNN -> {cnn_call} (p_malicious={cnn_p:.2f})"

    lines = [
        "=== RECORD UNDER REVIEW ===",
        votes_str,
        "",
        "Behavioural features  (value | pctile vs benign pop | pctile vs attack pop | meaning)",
    ]
    for f in feat_cols:
        v = float(row[f])
        pb = ctx_builder.percentile_label(v, pct_tables[f]["benign"])
        pa = ctx_builder.percentile_label(v, pct_tables[f]["attack"])
        lines.append(f"  {f:<22} {ctx_builder._fmt_value(v):<8} {pb:<6} {pa:<6} {ctx_builder.FEATURE_GLOSS[f]}")

    lines += [
        "",
        "Notable deviations (most unusual vs benign, first = most extreme):",
        "  " + ", ".join(ctx_builder.notable_deviations(row, feat_cols, pct_tables)),
    ]

    if variant != "no_failure_patterns":
        lines += [
            "",
            prompts.KNOWN_FAILURE_BLOCKS["camlds"]
        ]

    return "\n".join(lines)

def build_ablation_prompt(row, variant):
    context_str = build_ablation_context(row, FEAT_COLS, percentiles, variant)
    parts = [
        prompts.SYSTEM_PROMPT,
        prompts.COT_RUBRIC,
        prompts.SCHEMA_INSTRUCTION
    ]

    if variant != "no_fewshot":
        parts.append(FEWSHOT_EXAMPLE_CNN_FIRST if variant == "cnn_first" else prompts.FEWSHOT_EXAMPLE)

    parts.append(context_str)
    parts.append("Return ONLY the single JSON object for the RECORD UNDER REVIEW:")
    return "\n\n".join(parts)

def call_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "format": prompts.ARBITRATION_SCHEMA_V1,
        "options": {"temperature": 0.0, "seed": SEED, "num_predict": 800},
        "stream": False
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        raw_text = res.get("response", "{}")
        try:
            return json.loads(raw_text), "ok"
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0)), "ok"
                except json.JSONDecodeError:
                    return {}, "parse_error"
            return {}, "parse_error"

def run_variant(variant_name):
    print(f"\n=======================================================", flush=True)
    print(f">>> Running Ablation Variant: {variant_name} (N=100)", flush=True)
    print(f"=======================================================", flush=True)
    df = pd.read_csv(ABLATION_DATA)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=variant_name):
        prompt = build_ablation_prompt(row, variant_name)
        try:
            parsed, status = call_ollama(prompt)
        except Exception as e:
            results.append({
                "row_id": row["row_id"], "true_label": row["true_label"],
                "pred_xgb": row["pred_xgb"], "pred_cnn": row["pred_cnn"],
                "llm_verdict": None, "llm_pred": None, "llm_confidence": None,
                "key_features": None, "rationale": None,
                "parse_status": "call_error", "error": str(e)
            })
            continue

        verdict = parsed.get("verdict")
        pred_bin = 1 if verdict == "MALICIOUS" else (0 if verdict == "BENIGN" else None)
        results.append({
            "row_id": row["row_id"],
            "true_label": row["true_label"],
            "pred_xgb": row["pred_xgb"],
            "pred_cnn": row["pred_cnn"],
            "llm_verdict": verdict,
            "llm_pred": pred_bin,
            "llm_confidence": parsed.get("confidence"),
            "llm_agrees_with": parsed.get("agrees_with"),
            "key_features": json.dumps(parsed.get("key_features", [])),
            "rationale": " | ".join(parsed.get("rationale_steps", [])),
            "parse_status": status,
            "error": None
        })

    out_file = f"{OUT_DIR}/camlds_foundation-sec-8b_ablation_{variant_name}.csv"
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}", flush=True)

    ok = res_df[res_df["parse_status"] == "ok"]
    if len(ok):
        n = len(ok)
        agree_cnn = (ok["llm_pred"] == ok["pred_cnn"]).sum()
        agree_xgb = (ok["llm_pred"] == ok["pred_xgb"]).sum()
        pred_dist = ok["llm_pred"].value_counts().to_dict()
        print(f"[{variant_name}] Success: {n}/{len(df)} | Pred dist: {pred_dist}", flush=True)
        print(f"[{variant_name}] Agreement -- CNN: {agree_cnn}/{n} ({agree_cnn/n:.1%}) | XGBoost: {agree_xgb}/{n} ({agree_xgb/n:.1%})", flush=True)

if __name__ == "__main__":
    for v in ["cnn_first"]:
        run_variant(v)
