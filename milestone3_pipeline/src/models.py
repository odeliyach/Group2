"""
Milestone 3 — Step 7, Module 4/5: Model Training (builders)
============================================================
One builder per required paradigm (M2 Table 8):
  - Traditional Supervised : Random Forest, XGBoost
  - Deep Learning          : 1D-CNN over the tabular feature vector
  - Unsupervised           : Isolation Forest

Every builder consumes ONLY the explicit hyperparameter dicts in
config.py -- no bare `RandomForestClassifier()` calls anywhere. Fold-local
imbalance parameters (class_weight dict, scale_pos_weight) are passed in
at call time by train_eval.py, computed from the TRAINING fold only.
"""

from sklearn.ensemble import RandomForestClassifier, IsolationForest

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models as km, callbacks
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

from config import RF_PARAMS, XGB_PARAMS, CNN_PARAMS, IFOREST_PARAMS


def build_rf(overrides: dict = None) -> RandomForestClassifier:
    """Random Forest -- non-linear, high-dimensional boundary mapping via
    an ensemble of decorrelated trees, matched to the multicollinear
    privilege/identity feature structure identified in M2 Ch4."""
    params = {**RF_PARAMS, **(overrides or {})}
    return RandomForestClassifier(**params)


def build_xgb(scale_pos_weight: float, overrides: dict = None):
    """XGBoost -- sequential boosting targets exactly the hard,
    near-tautological residual cases a single RF misses (M2 Ch6.1).
    scale_pos_weight MUST be computed from the training fold only
    (see preprocessing.xgb_scale_pos_weight) -- never from the full
    dataset, to avoid imbalance-handling leakage."""
    if not _HAS_XGB:
        raise ImportError("xgboost not installed. pip install xgboost --break-system-packages")
    params = {**XGB_PARAMS, **(overrides or {})}
    params["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**params)


def build_cnn(n_features: int, overrides: dict = None):
    """1D-CNN convolving over the fixed-length engineered feature vector
    (reshaped to (n_features, 1)). Substituted for a sequential LSTM per
    M2 Ch6.1's justification: CasinoLimit is 98.2% single-event processes,
    which structurally precludes temporal sequence modeling -- there is no
    'before' state for an LSTM to learn from. The CNN instead learns
    non-linear spatial interactions between ADJACENT tabular features."""
    if not _HAS_TF:
        raise ImportError("tensorflow not installed. pip install tensorflow --break-system-packages")
    p = {**CNN_PARAMS, **(overrides or {})}

    # BUGFIX: CNN_PARAMS["random_state"] was defined in config.py but never
    # actually wired to TensorFlow -- unlike build_rf/build_xgb, which pass
    # their whole params dict straight into a constructor that accepts
    # random_state directly, Keras has no such single knob. Without this,
    # every fit used an uncontrolled random weight init, adding pure noise
    # variance on top of genuine model instability -- especially costly on
    # CasinoLimit's ~356 unique feature-vector groups, where any single
    # fit's luck-of-the-init matters more than on CAM-LDS's larger corpus.
    tf.random.set_seed(p["random_state"])

    inputs = layers.Input(shape=(n_features, 1))
    x = inputs
    for filters in p["conv_filters"]:
        x = layers.Conv1D(filters, p["kernel_size"], padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    for units in p["dense_units"]:
        x = layers.Dense(units, activation="relu")(x)
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


def cnn_fit_kwargs(overrides: dict = None) -> dict:
    """Fit-time kwargs shared across CV folds (epochs/batch_size/early
    stopping), kept separate from build_cnn's architecture kwargs."""
    p = {**CNN_PARAMS, **(overrides or {})}
    es = callbacks.EarlyStopping(monitor="val_auc", mode="max",
                                  patience=p["early_stopping_patience"],
                                  restore_best_weights=True)
    return {"epochs": p["epochs"], "batch_size": p["batch_size"], "callbacks": [es], "verbose": 0}


def build_iforest(overrides: dict = None) -> IsolationForest:
    """Isolation Forest -- root-access/identity-mismatch features (M2 Ch1)
    produce a low-density, separable minority pattern in feature space
    even without labels; isolation-based partitioning exploits this
    directly, no labeled attack examples required at fit time. Used
    unsupervised (fit on ALL training rows regardless of label) -- its
    predictions are then compared against the true label only at
    evaluation time, and it serves as the cascade's first-stage filter
    in Step 8."""
    params = {**IFOREST_PARAMS, **(overrides or {})}
    return IsolationForest(**params)


MODEL_REGISTRY = {
    "rf": build_rf,
    "xgb": build_xgb,
    "cnn": build_cnn,
    "iforest": build_iforest,
}
