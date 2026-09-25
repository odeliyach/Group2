import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base_dir = "/vol/joberant_nobck/data/NLP_368307701_2526a/odeliyac"
camlds_dir = os.path.join(base_dir, "llm_triage_results_fixed_percentiles/camlds")
abl_dir = os.path.join(camlds_dir, "ablation")
control_csv = os.path.join(camlds_dir, "camlds_foundation-sec-8b-latest_llm_arbitration.csv")

# טעינת קובץ המקור להשגת ה-Ground Truth
oof_csv = os.path.join(camlds_dir, "camlds_cascade_oof.csv")
contra_csv = os.path.join(camlds_dir, "camlds_contradictions.csv")

gt_map = {}
for src in [oof_csv, contra_csv]:
    if os.path.exists(src):
        tmp = pd.read_csv(src)
        col_id = "row_id" if "row_id" in tmp.columns else tmp.columns[0]
        gt_col = next((c for c in ["true_label", "ground_truth", "y_true", "label", "target", "actual"] if c in tmp.columns), None)
        if gt_col:
            for _, r in tmp.iterrows():
                k = str(r[col_id]).strip()
                gt_map[k] = r[gt_col]
            break

def to_int_safe(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if s in ["1", "1.0", "ATTACK", "TRUE", "MALICIOUS", "POS"]:
        return 1
    if s in ["0", "0.0", "BENIGN", "FALSE", "NORMAL", "NEG"]:
        return 0
    try:
        return int(float(s))
    except:
        return np.nan

conditions = {
    "Full Prompt (Control)": control_csv,
    "No Few-Shot": os.path.join(abl_dir, "camlds_foundation-sec-8b_ablation_no_fewshot.csv"),
    "CNN First": os.path.join(abl_dir, "camlds_foundation-sec-8b_ablation_cnn_first.csv"),
    "No Failure Patterns": os.path.join(abl_dir, "camlds_foundation-sec-8b_ablation_no_failure_patterns.csv")
}

summary_rows = []

for cond_name, file_path in conditions.items():
    if not os.path.exists(file_path):
        continue
    df = pd.read_csv(file_path)
    if "parse_status" in df.columns:
        df = df[df["parse_status"].isin(["ok", "repaired"])]
    if len(df) > 100:
        df = df.head(100)
    
    col_id = "row_id" if "row_id" in df.columns else df.columns[0]
    arb_col = next((c for c in ["arbitration", "decision", "prediction", "llm_pred"] if c in df.columns), None)
    cnn_col = next((c for c in ["cnn_prediction", "cnn_pred", "pred_cnn", "cnn"] if c in df.columns), None)
    
    arb_s = df[arb_col].apply(to_int_safe)
    cnn_s = df[cnn_col].apply(to_int_safe)
    
    # חישוב הסכמה מול CNN
    cnn_agree = (arb_s == cnn_s).mean()
    
    # חישוב Accuracy מול Ground Truth
    gt_s = df[col_id].astype(str).str.strip().map(gt_map).apply(to_int_safe)
    acc = (arb_s == gt_s).mean() if gt_s.notna().any() else np.nan
    
    summary_rows.append({
        "Condition": cond_name,
        "N": len(df),
        "Accuracy": round(float(acc), 4) if not np.isnan(acc) else "N/A",
        "CNN_Agreement_Rate": round(float(cnn_agree), 4)
    })

summary_df = pd.DataFrame(summary_rows)
out_csv = os.path.join(abl_dir, "ablation_comparison_summary.csv")
summary_df.to_csv(out_csv, index=False)

print("\n=== FINAL ABLATION SUMMARY TABLE ===")
print(summary_df.to_string(index=False))

# הפקת גרף מעודכן
plt.figure(figsize=(8, 5))
rates = [float(r) for r in summary_df["CNN_Agreement_Rate"]]
bars = plt.bar(
    summary_df["Condition"], 
    [r * 100 for r in rates], 
    color=["#2b5c8f", "#d9534f", "#e67e22", "#6f42c1"]
)
plt.ylabel("CNN Agreement Rate (%)", fontsize=11)
plt.title("CAM-LDS: Arbitration Agreement with CNN across Prompt Variants (N=100)", fontsize=12, pad=12)
plt.ylim(0, 115)
plt.grid(axis='y', linestyle='--', alpha=0.4)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
out_png = os.path.join(abl_dir, "ablation_agreement_comparison.png")
plt.savefig(out_png, dpi=300)

print(f"\nCSV written to:  {out_csv}")
print(f"Chart written to: {out_png}")
