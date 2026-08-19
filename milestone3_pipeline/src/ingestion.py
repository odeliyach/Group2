"""
Milestone 3 — Step 7, Module 1/5: Data Ingestion
=================================================
Per the Dependency Rule: this is the ONLY module allowed to contain
dataset-specific parsing logic (different raw schemas, label column
names, technique-exclusion rules). Everything it returns downstream is a
plain, uniform (X_df, y, groups, feat_cols, meta) tuple -- Preprocessing,
Feature Selection, Training, and Evaluation never see a dataset name or
branch on one again after this point.
"""

from dataclasses import dataclass
import pandas as pd
from config import FULL_FEATURE_LIST, DATASETS


@dataclass
class IngestedDataset:
    df: pd.DataFrame        # raw rows, filtered
    feat_cols: list          # intersection of FULL_FEATURE_LIST with this CSV
    label_col: str           # always accessed as ds.y going forward
    display_name: str
    y: "pd.Series"


def ingest(dataset_key: str, data_dir: str = ".") -> IngestedDataset:
    """Load one dataset by its config key ('camlds' or 'casino') and return
    a dataset-agnostic container. All dataset-specific quirks (label column
    name, technique exclusions, legacy MITRE labeling like T1166==T1548)
    are resolved here and nowhere else."""
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_key}'. Options: {list(DATASETS)}")
    cfg = DATASETS[dataset_key]

    path = f"{data_dir.rstrip('/')}/{cfg['path']}"
    df = pd.read_csv(path, low_memory=False)
    print(f"[ingestion] Loaded {cfg['display_name']}: {len(df):,} rows from {path}")

    # Dataset-specific technique exclusion (e.g. Casino's T1003/T1078 drop,
    # kept only for benign rows so the label distribution isn't distorted)
    if cfg["exclude_techniques"] and cfg["technique_col"] in df.columns:
        excl = cfg["exclude_techniques"]
        before = len(df)
        keep = ~df[cfg["technique_col"]].isin(excl) | (df[cfg["label_col"]] == 0)
        df = df[keep].reset_index(drop=True)
        print(f"[ingestion]   Excluded techniques {excl}: "
              f"{before - len(df):,} rows removed, {len(df):,} remain")

    feat_cols = [f for f in FULL_FEATURE_LIST if f in df.columns]
    missing = [f for f in FULL_FEATURE_LIST if f not in df.columns]
    print(f"[ingestion]   Features available: {len(feat_cols)}/{len(FULL_FEATURE_LIST)}"
          + (f"  (missing: {missing})" if missing else ""))

    y = df[cfg["label_col"]]
    print(f"[ingestion]   Benign: {(y == 0).sum():,}  Attack: {(y == 1).sum():,} "
          f"({100 * (y == 1).mean():.1f}% attack)")

    return IngestedDataset(df=df, feat_cols=feat_cols, label_col=cfg["label_col"],
                            display_name=cfg["display_name"], y=y)
