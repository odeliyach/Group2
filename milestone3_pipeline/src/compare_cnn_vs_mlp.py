"""
Milestone 3 -- M2 reviewer point 2: "Re-evaluate the Deep Learning
Architecture: if process execution traces in CasinoLimit lack sequential
depth (98.2% single-event processes), pivot from the 1D-CNN to a deep MLP
with Batch Normalization and Dropout. If retaining the 1D-CNN, document the
exact feature order and justify why neighboring feature combinations
possess spatial coherence."

No genuine spatial-coherence argument exists for this project's feature
order (lifetime_seconds next to events_per_second next to seq_length, etc.
-- M2's original table presentation order, not a deliberately engineered
adjacency). Rather than retrofit a justification that likely doesn't hold,
this implements the reviewer's preferred alternative directly and tests it
under the IDENTICAL protocol as every other headline number in this
project: same StratifiedGroupKFold, same inner-validation-split F1-optimal
threshold calibration (mirrors train_eval.py's _fit_predict_cnn exactly,
not a simplified reimplementation), same MAX_FPR cap, same fold-safe
StandardScaler.

Standalone by design -- does not modify models.py/train_eval.py/config.py,
since this session doesn't have fresh visibility into every constant in
those files. Everything the MLP needs is self-contained below; copy
build_mlp()/MLP_PARAMS into models.py/config.py permanently later if you
decide to adopt it.

Usage:
    python compare_cnn_vs_mlp.py --dataset camlds --data-dir data --out results/current/cnn_vs_mlp/
    python compare_cnn_vs_mlp.py --dataset casino --data-dir data --out results/current/cnn_vs_mlp/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models as km, callbacks
except ImportError:
    raise ImportError("tensorflow not installed. pip install tensorflow --break-system-packages")

from config import CV_CONFIG, MAX_FPR
from ingestion import ingest
from feature_selection import select_features
from preprocessing import vector_group_key, fit_scaler_on_train, fold_class_weights, impute
from train_eval import _best_f1_threshold, _fpr
from models import build_cnn, cnn_fit_kwargs  # reuse the REAL production CNN unmodified,
                                                 # for a true apples-to-apples comparison


# -- MLP: deep, with BatchNorm + Dropout at every hidden layer, as the
# reviewer specified. Self-contained params, not pulled from config.py. --
MLP_PARAMS = {
    "dense_units": [256, 128, 64, 32],  # deeper than the CNN's dense head,
                                         # since the MLP has no conv layers
                                         # doing any of the representation work
    "dropout": 0.3,
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 512,
    "early_stopping_patience": 5,
    "random_state": 42,
}


def build_mlp(n_features: int, overrides: dict = None):
    """Deep MLP with BatchNorm + Dropout after every hidden layer, per the
    M2 reviewer's exact spec. No convolution, no assumption that adjacent
    input columns are spatially related -- every feature is treated
    symmetrically, which is the reviewer's actual point."""
    p = {**MLP_PARAMS, **(overrides or {})}
    tf.random.set_seed(p["random_state"])

    inputs = layers.Input(shape=(n_features,))
    x = inputs
    for units in p["dense_units"]:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(p["dropout"])(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = km.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=p["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


def mlp_fit_kwargs(overrides: dict = None) -> dict:
    p = {**MLP_PARAMS, **(overrides or {})}
    es = callbacks.EarlyStopping(monitor="auc", mode="max",
                                  patience=p["early_stopping_patience"],
                                  restore_best_weights=True)
    return {"epochs": p["epochs"], "batch_size": p["batch_size"], "callbacks": [es], "verbose": 0}


def _fit_predict_mlp(X_tr, y_tr, groups_tr, X_te, max_fpr):
    """Mirrors train_eval.py's _fit_predict_cnn exactly -- same fold-safe
    scaler, same inner GroupShuffleSplit for threshold calibration, same
    class-weighting -- so the ONLY thing that differs from the CNN's
    result is the architecture itself."""
    scaler = fit_scaler_on_train(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=0)
    fit_idx, val_idx = next(gss.split(X_tr_s, y_tr, groups_tr))
    cw = fold_class_weights(y_tr[fit_idx])

    clf = build_mlp(n_features=X_tr.shape[1])
    fit_kwargs = mlp_fit_kwargs()
    clf.fit(X_tr_s[fit_idx], y_tr[fit_idx], class_weight=cw,
            validation_data=(X_tr_s[val_idx], y_tr[val_idx]), **fit_kwargs)

    val_proba = clf.predict(X_tr_s[val_idx], verbose=0).ravel()
    threshold = _best_f1_threshold(y_tr[val_idx], val_proba, max_fpr=max_fpr)

    proba = clf.predict(X_te_s, verbose=0).ravel()
    return (proba >= threshold).astype(int), proba


def _fit_predict_cnn_standalone(X_tr, y_tr, groups_tr, X_te, max_fpr):
    """Same as train_eval.py's _fit_predict_cnn, reproduced here so this
    script doesn't need to import a private underscore-prefixed function
    from another module. Uses the REAL build_cnn/cnn_fit_kwargs from
    models.py -- architecture itself is untouched, not reimplemented."""
    scaler = fit_scaler_on_train(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)[..., None]

    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=0)
    fit_idx, val_idx = next(gss.split(X_tr_s, y_tr, groups_tr))
    cw = fold_class_weights(y_tr[fit_idx])

    clf = build_cnn(n_features=X_tr.shape[1])
    fit_kwargs = cnn_fit_kwargs()
    clf.fit(X_tr_s[fit_idx][..., None], y_tr[fit_idx], class_weight=cw,
            validation_data=(X_tr_s[val_idx][..., None], y_tr[val_idx]), **fit_kwargs)

    val_proba = clf.predict(X_tr_s[val_idx][..., None], verbose=0).ravel()
    threshold = _best_f1_threshold(y_tr[val_idx], val_proba, max_fpr=max_fpr)

    proba = clf.predict(X_te_s, verbose=0).ravel()
    return (proba >= threshold).astype(int), proba


def run_variant(fit_predict_fn, df, y, groups, feat_cols, repeats, n_splits, base_seed, max_fpr):
    X = impute(df[feat_cols].values)
    rows = []
    for rep in range(repeats):
        seed = base_seed + rep
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y_true_all, y_pred_all, y_proba_all = [], [], []
        for tr, te in cv.split(X, y, groups):
            if len(set(y[te])) < 2:
                continue
            pred, proba = fit_predict_fn(X[tr], y[tr], groups[tr], X[te], max_fpr)
            y_true_all.extend(y[te]); y_pred_all.extend(pred); y_proba_all.extend(proba)
        y_true_all, y_pred_all, y_proba_all = map(np.array, (y_true_all, y_pred_all, y_proba_all))
        rows.append({
            'repeat': rep,
            'f1': f1_score(y_true_all, y_pred_all, zero_division=0),
            'auroc': roc_auc_score(y_true_all, y_proba_all),
            'fpr': _fpr(y_true_all, y_pred_all),
        })
        print(f"  repeat {rep}: F1={rows[-1]['f1']:.4f} AUROC={rows[-1]['auroc']:.4f} "
              f"FPR={rows[-1]['fpr']:.4f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["camlds", "casino"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = ingest(args.dataset, data_dir=args.data_dir)
    feat_cols = select_features(ds.feat_cols, args.dataset, policy="full")
    df = ds.df
    y = ds.y.values
    groups = vector_group_key(df, feat_cols)
    base_seed = CV_CONFIG["seed"]

    print(f"=== {args.dataset}: CNN (production, unmodified) ===")
    cnn_df = run_variant(_fit_predict_cnn_standalone, df, y, groups, feat_cols,
                          args.repeats, args.n_splits, base_seed, MAX_FPR)

    print(f"\n=== {args.dataset}: MLP (BatchNorm + Dropout, M2 reviewer spec) ===")
    mlp_df = run_variant(_fit_predict_mlp, df, y, groups, feat_cols,
                          args.repeats, args.n_splits, base_seed, MAX_FPR)

    cnn_df.to_csv(out_dir / f"{args.dataset}_cnn.csv", index=False)
    mlp_df.to_csv(out_dir / f"{args.dataset}_mlp.csv", index=False)

    print(f"\n=== {args.dataset}: SUMMARY ===")
    print(f"CNN: F1={cnn_df['f1'].mean():.4f}\u00b1{cnn_df['f1'].std():.4f}  "
          f"AUROC={cnn_df['auroc'].mean():.4f}\u00b1{cnn_df['auroc'].std():.4f}")
    print(f"MLP: F1={mlp_df['f1'].mean():.4f}\u00b1{mlp_df['f1'].std():.4f}  "
          f"AUROC={mlp_df['auroc'].mean():.4f}\u00b1{mlp_df['auroc'].std():.4f}")

    verdict = ("MLP performs comparably or better -- supports the reviewer's hypothesis that "
               "spatial-locality assumptions in the CNN aren't earning their keep here"
               if mlp_df['f1'].mean() >= cnn_df['f1'].mean() - 0.01 else
               "CNN still outperforms -- if adopting, the spatial-coherence justification the "
               "reviewer asked for still needs to be written, since this result alone doesn't "
               "explain WHY adjacency in this feature order helps")
    print(f"\nVerdict: {verdict}")

    summary_path = out_dir / f"{args.dataset}_summary.txt"
    summary_path.write_text(
        f"CNN: F1={cnn_df['f1'].mean():.4f}+/-{cnn_df['f1'].std():.4f} "
        f"AUROC={cnn_df['auroc'].mean():.4f}+/-{cnn_df['auroc'].std():.4f}\n"
        f"MLP: F1={mlp_df['f1'].mean():.4f}+/-{mlp_df['f1'].std():.4f} "
        f"AUROC={mlp_df['auroc'].mean():.4f}+/-{mlp_df['auroc'].std():.4f}\n"
        f"Verdict: {verdict}\n"
    )
    print(f"\nSaved: {out_dir}/")


if __name__ == "__main__":
    main()
