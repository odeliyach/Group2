"""
Milestone 3 — Step 7: Central configuration.

Every hyperparameter for all 4 models is defined HERE, explicitly. No
model-building code elsewhere should silently fall back on a scikit-learn/
XGBoost/Keras default.

Reused verbatim from Milestone 2 (unified_eda.py / unified_validation.py):
FULL_FEATURE_LIST, DOMAIN_RANK, class_weight='balanced' imbalance strategy,
and the vector-group leak-free CV split methodology.
"""

FULL_FEATURE_LIST = [
    'lifetime_seconds', 'events_per_second', 'seq_length',
    'unique_syscalls', 'uid_is_root', 'uid_changed', 'euid_root',
    'max_uid_euid_delta', 'is_suid_exec', 'auid_euid_mismatch',
    'ephemeral_privileged', 'failed_call_rate', 'failed_call_count',
    'priv_op_count', 'exec_count', 'file_access_count',
    'network_count', 'shell_from_service', 'sensitive_path_access',
    'parent_child_rarity',
]

DOMAIN_RANK = {
    'is_suid_exec': 1, 'auid_euid_mismatch': 2, 'max_uid_euid_delta': 3,
    'priv_op_count': 4, 'exec_count': 5, 'sensitive_path_access': 6,
    'shell_from_service': 7, 'parent_child_rarity': 8, 'ephemeral_privileged': 9,
    'events_per_second': 10, 'failed_call_rate': 11, 'euid_root': 12,
    'uid_is_root': 13, 'uid_changed': 14, 'failed_call_count': 15,
    'file_access_count': 16, 'network_count': 17, 'unique_syscalls': 18,
    'seq_length': 19, 'lifetime_seconds': 20,
}

DATASETS = {
    "camlds": {
        "path": "dataset1_features.csv",
        "label_col": "label",
        "technique_col": "technique",
        "exclude_techniques": None,
        "display_name": "CAM-LDS",
    },
    "casino": {
        "path": "casino_process_level_ds.csv",
        "label_col": "is_privesc",
        "technique_col": "technique",
        "exclude_techniques": ["T1003", "T1078"],
        "display_name": "CasinoLimit",
    },
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "min_samples_split": 10,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": 42,
    # scale_pos_weight computed per-fold at call time, not fixed here.
}

CNN_PARAMS = {
    "conv_filters": [64, 32],
    "kernel_size": 3,
    "dense_units": [32],
    "dropout": 0.3,
    "learning_rate": 1e-2,  # was 1e-3 -- sensitivity_analysis.py's own sweep shows
                             # lr=0.01 beats lr=0.001 on BOTH datasets (Casino F1
                             # 0.642->0.751, std 0.381->0.282; CAM-LDS F1 0.808->0.821)
    "batch_size": 512,
    "epochs": 15,
    "early_stopping_patience": 3,
    "class_weight_strategy": "balanced",
    "random_state": 42,
}

IFOREST_PARAMS = {
    "n_estimators": 200,
    "max_samples": 256,
    "contamination": "auto",   # note: only affects clf.predict()'s own
                                # -1/+1 cutoff, which train_eval.py no
                                # longer uses -- the actual decision
                                # threshold is F1-optimized on train-fold
                                # labels instead (see train_eval.py docstring)
    "max_features": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

# NOVELTY-DETECTION MODE: if True, Isolation Forest is fit ONLY on rows
# where y_train==0 (confirmed benign) instead of the full mixed training
# fold. Standard technique in anomaly-detection literature -- fitting on
# a population that's ~50% "anomalies" (CAM-LDS) badly dilutes what the
# model learns as "normal". Fitting on clean-only data gives it an actual
# clean baseline to measure deviation against.
#
# DISCLOSURE: this uses train-fold LABELS to select the fit() subset --
# a further step beyond the threshold-calibration disclosure in
# train_eval.py. This changes Isolation Forest from a purely unsupervised
# method to a semi-supervised "novelty detection" method. State this
# explicitly in Ch6/Ch8 rather than presenting it as unsupervised.
IFOREST_NOVELTY_MODE = True

SENSITIVITY_GRIDS = {
    "rf": {"max_depth": [4, 8, 12, 20, None],
           "n_estimators": [100, 300, 500]},
    "xgb": {"max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2]},
    "cnn": {"dropout": [0.1, 0.3, 0.5],
            "learning_rate": [1e-2, 1e-3, 1e-4]},
    "iforest": {"n_estimators": [50, 100, 200, 400],
                "max_samples": [64, 128, 256, 512]},
}

# Binary 0/1 flag features -- everything else in FULL_FEATURE_LIST is a
# continuous count/rate/delta. Used by LOG1P_FOR_ISOFOREST below.
BINARY_FEATURES = {
    'uid_is_root', 'uid_changed', 'euid_root', 'is_suid_exec',
    'auid_euid_mismatch', 'ephemeral_privileged', 'shell_from_service',
    'sensitive_path_access',
}

# PREPROCESSING FIX for Isolation Forest specifically: unlike RF/XGBoost
# (which choose splits by information gain over SORTED values, so any
# monotonic transform is a no-op), Isolation Forest picks split points
# UNIFORMLY AT RANDOM between a feature's min and max at each node. A
# heavily right-skewed feature (e.g. exec_count: mostly near 0, long tail)
# gets most of its random splits wasted in the dense low region, rarely
# probing the tail where the actual structure is. log1p-transforming the
# continuous features before fitting concentrates splits more evenly
# across the range that actually carries information.
#
# Applied globally (not just for Isolation Forest) because it's provably
# SAFE everywhere: log1p is strictly increasing for x>=0, so it's a no-op
# for RF/XGBoost's split decisions (which only depend on sort order), and
# gives CNN's gradient-based training a less skewed input distribution
# before scaling. Only Isolation Forest's behavior actually changes.
LOG1P_CONTINUOUS_FEATURES = True

CV_CONFIG = {
    "n_splits": 5,
    "repeats": 15,
    "seed": 42,
}

# Operational deployment constraint: an IDS that flags too large a fraction
# of BENIGN traffic is useless to analysts regardless of how much recall
# it buys (alert fatigue). This caps how much FPR the F1-threshold search
# in train_eval.py is allowed to spend when picking a cutoff -- it will
# maximize F1 subject to FPR staying under this bound, falling back to
# the lowest-FPR threshold available if nothing meets the cap (e.g. a
# genuinely non-discriminative model like Isolation Forest on a
# near-balanced dataset, where NO threshold can buy real precision).
#
# 0.05 chosen from operating_curve.py's real sweep on CAM-LDS/RF: F1 peaks
# at cap=0.05 (F1=0.863, actual test FPR~0.094) and DECLINES for looser
# caps (F1 drops to ~0.838 by cap=0.20+) -- loosening further trades away
# precision faster than it buys recall. See
# operating_curve_results/camlds_rf_operating_curve.csv for the full sweep.
MAX_FPR = 0.05

# =====================================================================
# Step 8 capstone -- LLM-Driven Triage and Contextual Arbitration
# =====================================================================

# One-line SOC-analyst gloss per feature, shown verbatim in the LLM context
# window (see llm_triage_context.build_context). Keys MUST match FULL_FEATURE_LIST.
FEATURE_GLOSS = {
    "lifetime_seconds": "process wall-clock lifetime",
    "events_per_second": "syscall throughput",
    "seq_length": "length of the syscall sequence",
    "unique_syscalls": "distinct syscall types used",
    "uid_is_root": "process real UID is root",
    "uid_changed": "UID changed during the process lifetime",
    "euid_root": "effective UID is root",
    "max_uid_euid_delta": "largest gap between real and effective UID; large = privilege transition",
    "is_suid_exec": "executed a setuid/setgid binary",
    "auid_euid_mismatch": "audit (login) UID differs from effective UID; identity inconsistency",
    "ephemeral_privileged": "short-lived process that held privilege",
    "failed_call_rate": "fraction of syscalls that returned an error",
    "failed_call_count": "absolute count of failed syscalls",
    "priv_op_count": "count of privileged operations (setuid, capset, ...)",
    "exec_count": "number of exec() calls",
    "file_access_count": "file-access syscalls",
    "network_count": "network-related syscalls",
    "shell_from_service": "a shell was spawned from a service/daemon context",
    "sensitive_path_access": "touched a sensitive path (/etc/shadow, /etc/sudoers, ...)",
    "parent_child_rarity": "rarity of this parent->child exec pair (1 = never seen before)",
}

LLM_TRIAGE = {
    "model_id": "fdtn-ai/Foundation-Sec-8B-Instruct",      # security-tuned primary
    "control_model_id": "NousResearch/Meta-Llama-3.1-8B-Instruct",  # ungated mirror of the gated
                                                                     # meta-llama/Llama-3.1-8B-Instruct
    "backend": "vllm",                                       # "vllm" (cluster) or "ollama" (fallback)
    "max_new_tokens": 800,
    "sampling_seed": 42,
    "temperature": 0.0,
    "max_cases_per_dataset": 400,        # stratified cap; bounds LLM cost
    # mirrors hybrid_cascade.cascade_predict low_thresh/high_thresh defaults -- keep in sync
    "cascade_low_thresh": 0.3,
    "cascade_high_thresh": 0.7,
    "schema_version": "v1",
    # GB of model weights vLLM may stream from host RAM instead of VRAM, so a
    # bf16 8B model (~16 GB) fits a smaller GPU (e.g. a 12 GB titan). 0 = off.
    "vllm_cpu_offload_gb": 0,
    # Pre-Volta GPUs (compute capability < 8.0, e.g. Titan Xp = 6.1) cannot run
    # bfloat16 at all -- vLLM raises at engine init. Use "float16" on those.
    "vllm_dtype": "bfloat16",
}
