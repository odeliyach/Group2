"""
Milestone 3 — Step 7, Module 3/5: Feature Selection & Engineering
==================================================================
Dataset-agnostic. This module does NOT re-derive features from raw logs
(that stays in ingestion.py's dataset-specific upstream extraction,
outside this pipeline's scope) -- it selects WHICH of the already-engineered
20 M2 features to feed the models with, based on M2 Ch3/Ch4's empirical
findings, not a fresh per-run decision.

Two selectable policies:
  "full"    -- all features available in this dataset's CSV (intersection
               with FULL_FEATURE_LIST, per ingestion.py).
  "reduced" -- M2's ablation-optimized subset (unified_validation.py's
               feature_list_optimization.txt result: drop features tagged
               'neutral' or 'hurts' by per-feature ablation).

Since Ch3.2 of the M2 report found several highly-correlated pairs (e.g.
is_suid_exec / max_uid_euid_delta) that ablation showed carry INDEPENDENT
marginal signal, this module does NOT do correlation-based pruning on its
own -- only the empirically-tested ablation tags are used to decide what's
droppable, consistent with that finding.
"""

# Populate after running unified_validation.py's ablation on your actual
# data -- placeholder policy: keep everything until ablation says
# otherwise for a *given* dataset (ablation results differ CAM-LDS vs
# CasinoLimit, see M2 Table 5/6, so this map is dataset-specific evidence
# rather than a hardcoded universal list).
ABLATION_DROPPABLE = {
    "camlds": [],   # fill in from validation_output/cam_lds/feature_ablation.csv
    "casino": [],   # fill in from validation_output/casino/feature_ablation.csv
}


def select_features(feat_cols: list, dataset_key: str, policy: str = "full") -> list:
    if policy == "full":
        return list(feat_cols)
    if policy == "reduced":
        drop = set(ABLATION_DROPPABLE.get(dataset_key, []))
        reduced = [f for f in feat_cols if f not in drop]
        if not reduced:
            raise ValueError(f"Reduced policy dropped every feature for '{dataset_key}' -- "
                              f"check ABLATION_DROPPABLE.")
        return reduced
    raise ValueError(f"Unknown feature-selection policy '{policy}' (use 'full' or 'reduced')")
