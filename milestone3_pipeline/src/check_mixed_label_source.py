"""
Follow-up: WHY do mixed-label groups exist in CAM-LDS? Two very different
possible explanations with very different implications:

  A) Feature coarseness: genuinely different real events (different
     technique/source/proc_name) happen to collide into identical values
     across the 20 engineered features. Not a bug -- a real limitation of
     feature resolution, worth documenting as a limitation.
  B) Actual data bug: the exact same underlying event was extracted with
     contradictory labels (same source/proc_name appearing as both benign
     and attack). This WOULD need fixing at the extraction/labeling step.

This checks metadata columns beyond the 20 numeric features (technique,
source, proc_name) for each mixed-label group to distinguish the two.

Usage:
    python check_mixed_label_source.py --data-dir data --out results/current/bad_repeat_diagnosis/
"""

import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd

from ingestion import ingest
from feature_selection import select_features
from preprocessing import vector_group_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--show-n", type=int, default=15, help="How many mixed groups to print in detail")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ingest("camlds", data_dir=args.data_dir)
    feat_cols = select_features(ds.feat_cols, "camlds", policy="full")
    df = ds.df.copy()
    df['group_id'] = vector_group_key(df, feat_cols)
    df['label'] = ds.y.values

    meta_cols = [c for c in ['technique', 'source', 'proc_name'] if c in df.columns]
    if not meta_cols:
        print("None of technique/source/proc_name found in this dataframe's columns -- "
              f"available columns: {list(df.columns)}")
        print("Adjust meta_cols above to whatever identifying columns actually exist.")
        return
    print(f"Using metadata columns: {meta_cols}")

    # Find mixed-label groups
    group_labels = df.groupby('group_id')['label'].nunique()
    mixed_group_ids = group_labels[group_labels > 1].index.tolist()
    print(f"\n{len(mixed_group_ids)} mixed-label groups found.\n")

    results = []
    for gid in mixed_group_ids:
        sub = df[df['group_id'] == gid]
        row = {'group_id': gid[:60] + '...', 'n_rows': len(sub), 'n_benign': (sub['label']==0).sum(),
               'n_attack': (sub['label']==1).sum()}
        for col in meta_cols:
            row[f'{col}_nunique'] = sub[col].nunique()
            row[f'{col}_values'] = sub[col].unique()[:5].tolist()  # sample
        results.append(row)

    res_df = pd.DataFrame(results).sort_values('n_rows', ascending=False)
    csv_path = out_dir / "mixed_label_group_metadata.csv"
    res_df.to_csv(csv_path, index=False)

    print(res_df.head(args.show_n).to_string(index=False))

    # Verdict
    same_source_groups = 0
    if 'proc_name_nunique' in res_df.columns:
        same_source_groups = (res_df['proc_name_nunique'] == 1).sum()
    print(f"\n>>> {same_source_groups}/{len(mixed_group_ids)} mixed-label groups share "
          f"the SAME proc_name across both labels.")
    print(">>> High count here -> Hypothesis B (likely a real labeling/dedup bug).")
    print(">>> Low/zero count -> Hypothesis A (genuinely different events colliding")
    print(">>> in a coarse 20-feature space -- a real limitation, not a bug).")
    print(f"\nSaved full detail: {csv_path}")


if __name__ == "__main__":
    main()
