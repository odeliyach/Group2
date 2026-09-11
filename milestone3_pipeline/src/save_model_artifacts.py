"""
Milestone 3 report -- gap-filling script 2 of 3: model artifact archive.

Fits each of the four models on the FULL dataset (not a CV fold -- a CV fold
model is a throwaway internal object, not the artifact you'd actually want to
archive/deploy) and saves it to disk. This is the "Model Artifact Archive"
link the submission template asks for -- currently nothing is saved anywhere.

Usage (run once per dataset you want an archived model for):
    python save_model_artifacts.py --dataset camlds --data-dir data/v3 --out artifacts/camlds
    python save_model_artifacts.py --dataset casino --data-dir data     --out artifacts/casino

After running, upload the resulting artifacts/ directory to wherever your
"Model Artifact Archive" link should point (a release asset, a shared drive
folder, etc.) -- this script only creates the files locally, it doesn't
handle the upload/hosting step.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from config import RF_PARAMS, XGB_PARAMS, CNN_PARAMS, IFOREST_PARAMS, MAX_FPR
from ingestion import ingest
from feature_selection import select_features
from preprocessing import vector_group_key, log1p_continuous, impute
from config import LOG1P_CONTINUOUS_FEATURES, BINARY_FEATURES


def main():
    ap = argparse.ArgumentParser(description="Save final full-fit model artifacts")
    ap.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ingest(args.dataset, data_dir=args.data_dir)
    feat_cols = select_features(ds.feat_cols, args.dataset, policy="full")
    X_raw = ds.df[feat_cols].values.astype(float)
    X = (log1p_continuous(X_raw, feat_cols, BINARY_FEATURES)
         if LOG1P_CONTINUOUS_FEATURES else X_raw.copy())
    X = impute(X)
    y = ds.y.values

    print(f"[save_artifacts] {args.dataset}: {len(y):,} rows, {len(feat_cols)} features, "
          f"fitting on the FULL dataset (not a CV fold)")

    # -- Random Forest --------------------------------------------------
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X, y)
    joblib.dump(rf, out_dir / f"{args.dataset}_rf.pkl")
    print(f"  saved {out_dir}/{args.dataset}_rf.pkl")

    # -- XGBoost ----------------------------------------------------------
    import xgboost as xgb
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    xgb_params = dict(XGB_PARAMS)
    xgb_params["scale_pos_weight"] = scale_pos_weight
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X, y)
    xgb_model.save_model(out_dir / f"{args.dataset}_xgb.json")
    print(f"  saved {out_dir}/{args.dataset}_xgb.json (scale_pos_weight={scale_pos_weight:.3f} "
          f"-- recomputed on the full dataset since it's fit-time-only, not stored in config)")

    # -- Isolation Forest --------------------------------------------------
    from sklearn.ensemble import IsolationForest
    iforest = IsolationForest(**IFOREST_PARAMS)
    iforest.fit(X)  # unsupervised -- no y
    joblib.dump(iforest, out_dir / f"{args.dataset}_iforest.pkl")
    print(f"  saved {out_dir}/{args.dataset}_iforest.pkl")

    # -- CNN ----------------------------------------------------------------
    try:
        from models import build_cnn
        cnn = build_cnn(n_features=X.shape[1])  # overrides=None -> uses production CNN_PARAMS
        cnn.fit(X, y, epochs=CNN_PARAMS["epochs"], batch_size=CNN_PARAMS["batch_size"],
                class_weight=({0: 1.0, 1: scale_pos_weight}
                              if CNN_PARAMS.get("class_weight_strategy") == "balanced" else None),
                verbose=0)
        cnn.save(out_dir / f"{args.dataset}_cnn.keras")
        print(f"  saved {out_dir}/{args.dataset}_cnn.keras")
    except Exception as e:
        print(f"  SKIPPED CNN -- {type(e).__name__}: {e}")
        print(f"  RF/XGBoost/IForest artifacts above are still saved and valid --")
        print(f"  this only affects the CNN artifact.")

    # -- manifest -----------------------------------------------------------
    manifest = {
        "dataset": args.dataset,
        "n_rows": int(len(y)),
        "n_features": len(feat_cols),
        "feature_columns": feat_cols,
        "rf_params": RF_PARAMS,
        "xgb_params": xgb_params,
        "iforest_params": IFOREST_PARAMS,
        "cnn_params": CNN_PARAMS,
        "max_fpr_operating_point": MAX_FPR,
        "note": "Fit on the FULL dataset for archival/deployment -- NOT the same "
                "object as any individual cross-validation fold's model.",
    }
    (out_dir / f"{args.dataset}_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  saved {out_dir}/{args.dataset}_manifest.json (exact config used, for reproducibility)")
    print(f"\n[save_artifacts] Done. Upload the contents of {out_dir}/ to your archive "
          f"location and put that link in the report's metadata block.")


if __name__ == "__main__":
    main()
