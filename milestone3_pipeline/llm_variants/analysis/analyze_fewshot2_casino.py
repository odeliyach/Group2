import csv
import sys
import os

out_path = "/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/llm_triage_results_fewshot2/casino/casino_fewshot2_summary.txt"
files = [
    ("Foundation-Sec (Few-Shot 2)", "/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/llm_triage_results_fewshot2/casino/casino_foundation-sec-8b-latest_llm_arbitration.csv"),
    ("Llama 3.1 (Few-Shot 2)", "/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac/llm_triage_results_fewshot2/casino/casino_llama3-1-8b_llm_arbitration.csv")
]

with open(out_path, "w", encoding="utf-8") as out_f:
    def log_print(msg):
        print(msg)
        out_f.write(msg + "\n")

    log_print("CASINO Few-Shot 2 Arbitration Results\n")
    
    for model, fname in files:
        log_print("=" * 55)
        log_print(f"Model: {model}")
        try:
            if not os.path.exists(fname):
                log_print(f"File not found: {fname}")
                continue
                
            with open(fname, "r", encoding="utf-8", errors="ignore") as f:
                rows = list(csv.DictReader(f))
            
            valid = [r for r in rows if r.get("llm_verdict") in ["MALICIOUS", "BENIGN"]]
            n = len(valid)
            if not n:
                log_print("No valid verdicts found")
                continue

            cnn_mal = sum(1 for r in valid if str(r.get("pred_cnn")) == "1")
            cnn_ben = sum(1 for r in valid if str(r.get("pred_cnn")) == "0")
            llm_mal = sum(1 for r in valid if r.get("llm_verdict") == "MALICIOUS")
            llm_ben = sum(1 for r in valid if r.get("llm_verdict") == "BENIGN")
            agrees_cnn = sum(1 for r in valid if r.get("llm_agrees_with") == "cnn" or str(r.get("llm_pred")) == str(r.get("pred_cnn")))
            agrees_xgb = sum(1 for r in valid if r.get("llm_agrees_with") == "xgb" or str(r.get("llm_pred")) == str(r.get("pred_xgb")))

            log_print(f"Total valid rows: {n}")
            log_print(f"CNN actual votes: MALICIOUS={cnn_mal} ({cnn_mal/n:.1%}), BENIGN={cnn_ben} ({cnn_ben/n:.1%})")
            log_print(f"LLM verdicts:     MALICIOUS={llm_mal} ({llm_mal/n:.1%}), BENIGN={llm_ben} ({llm_ben/n:.1%})")
            log_print(f"Agrees with CNN:  {agrees_cnn}/{n} ({agrees_cnn/n:.1%})")
            log_print(f"Agrees with XGB:  {agrees_xgb}/{n} ({agrees_xgb/n:.1%})")
        except Exception as e:
            log_print(f"Error reading {model}: {e}")

print(f"\nResults successfully saved to: {out_path}")
