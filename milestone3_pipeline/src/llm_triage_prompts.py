"""Milestone 3 -- Step 8 capstone: static prompt assets for LLM arbitration.

Everything here is a versioned constant so the report can quote it verbatim
(schema_version = "v1"). Nothing dataset-specific except KNOWN_FAILURE_BLOCKS,
which are aggregate facts lifted from error_analysis.py's FP-vs-TN / FN-vs-TP
tables -- population-level, not row-level, so not a label leak.
"""

SCHEMA_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a Tier-2 SOC analyst on a Linux host-intrusion team. Two ML models in "
    "an automated detection pipeline have produced CONTRADICTING classifications for "
    "one process-execution record. Your job is to arbitrate: decide MALICIOUS (a "
    "privilege-escalation attempt) or BENIGN, using structured reasoning over the "
    "process's behavioural features.\n\n"
    "The monitored environment contains normal system and administrative activity plus "
    "privilege-escalation attempts in the MITRE ATT&CK families T1548 (Abuse Elevation "
    "Control Mechanism), T1059 (Command and Scripting Interpreter), T1222 "
    "(File/Directory Permissions Modification), and T1595 (Active Scanning). Attacks "
    "are a minority of records (roughly 17-29%).\n\n"
    "Follow the reasoning rubric exactly and return ONLY a single JSON object matching "
    "the schema. Do not write anything before or after the JSON."
)

COT_RUBRIC = (
    "Reasoning rubric -- work through these five steps in order, one entry in "
    "rationale_steps per step:\n"
    "1. Privilege state: is there a genuine UID/EUID transition or root context? "
    "Consider uid_is_root, euid_root, max_uid_euid_delta, uid_changed, auid_euid_mismatch.\n"
    "2. Mechanism: setuid/setgid execution, an unusual exec chain, or a shell spawned "
    "by a service? Consider is_suid_exec, exec_count, shell_from_service, priv_op_count.\n"
    "3. Behavioural anomaly: failed-syscall rate, rare parent->child pair, ephemeral "
    "privileged process, sensitive-path access, network activity. Consider "
    "failed_call_rate, parent_child_rarity, ephemeral_privileged, sensitive_path_access, "
    "network_count.\n"
    "4. Benign-explanation test: could this plausibly be admin scripting, a cron job, "
    "package management, or a health-check? Weigh events_per_second, lifetime_seconds, "
    "seq_length, unique_syscalls.\n"
    "5. Decide: weigh steps 1-4, state which model's call the evidence supports, and "
    "give a calibrated confidence."
)

SCHEMA_INSTRUCTION = (
    "Return ONLY this JSON object and nothing else:\n"
    "{\n"
    '  "verdict": "MALICIOUS" or "BENIGN",\n'
    '  "confidence": a number from 0.0 to 1.0,\n'
    '  "rationale_steps": [3 to 6 short strings, one per rubric step],\n'
    '  "key_features": [up to 5 feature names that drove the decision],\n'
    '  "agrees_with": "xgb" or "cnn" or "neither"\n'
    "}"
)

ARBITRATION_SCHEMA_V1 = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["MALICIOUS", "BENIGN"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale_steps": {
            "type": "array", "items": {"type": "string"},
            "minItems": 3, "maxItems": 6,
        },
        "key_features": {
            "type": "array", "items": {"type": "string"}, "maxItems": 5,
        },
        "agrees_with": {"type": "string", "enum": ["xgb", "cnn", "neither"]},
    },
    "required": ["verdict", "confidence", "rationale_steps", "key_features", "agrees_with"],
    "additionalProperties": False,
}

FEWSHOT_EXAMPLE = (
    "--- EXAMPLE ---\n"
    "=== RECORD UNDER REVIEW ===\n"
    "Model votes:  XGBoost -> BENIGN (p_malicious=0.41)   |   CNN -> MALICIOUS (p_malicious=0.66)\n"
    "\n"
    "Behavioural features  (value | pctile vs benign pop | pctile vs attack pop | meaning)\n"
    "  max_uid_euid_delta     15       p99+   p90    largest gap between real and effective UID; large = privilege transition\n"
    "  auid_euid_mismatch     1        p98    p72    audit (login) UID differs from effective UID; identity inconsistency\n"
    "  is_suid_exec           1        p99+   p88    executed a setuid/setgid binary\n"
    "  shell_from_service     1        p99+   p80    a shell was spawned from a service/daemon context\n"
    "  sensitive_path_access  1        p99+   p70    touched a sensitive path (/etc/shadow, /etc/sudoers, ...)\n"
    "  lifetime_seconds       3        p10    p20    process wall-clock lifetime\n"
    "  (... remaining features omitted in this example ...)\n"
    "\n"
    "Notable deviations (most unusual vs benign, first = most extreme):\n"
    "  is_suid_exec, shell_from_service, sensitive_path_access, max_uid_euid_delta, auid_euid_mismatch\n"
    "\n"
    "Known failure patterns in this dataset (from labelled error analysis):\n"
    "  - Benign rows misflagged as attack tend to have elevated parent_child_rarity and uid_changed with short lifetime_seconds.\n"
    "  - Missed attacks tend to have auid_euid_mismatch set but low max_uid_euid_delta and priv_op_count.\n"
    "\n"
    "--- IDEAL ANSWER ---\n"
    '{"verdict": "MALICIOUS", "confidence": 0.86, "rationale_steps": ['
    '"Privilege state: root context with max_uid_euid_delta at p99+ and auid_euid_mismatch set -- a real identity inconsistency and privilege transition.", '
    '"Mechanism: is_suid_exec is set and a shell was spawned from a service context with priv_op_count elevated -- a classic setuid-shell escalation.", '
    '"Behavioural anomaly: sensitive_path_access=1 and parent_child_rarity near 1 -- an exec pair essentially never seen in benign activity.", '
    '"Benign-explanation test: very short lifetime (p10) and low throughput argue against a routine admin script, cron job, or health-check.", '
    '"Decide: the escalation mechanism and sensitive-path access outweigh XGBoost\'s benign call; this supports the CNN."], '
    '"key_features": ["is_suid_exec", "shell_from_service", "sensitive_path_access", "max_uid_euid_delta", "auid_euid_mismatch"], '
    '"agrees_with": "cnn"}\n'
    "--- END EXAMPLE ---"
)

# A second, contrasting worked example. Deliberately uses the SAME vote
# pattern as FEWSHOT_EXAMPLE (XGBoost -> BENIGN, CNN -> MALICIOUS) but a
# different feature profile that supports the opposite verdict (BENIGN,
# agrees_with xgb). A single MALICIOUS/agrees-with-cnn example risks teaching
# "when CNN and XGBoost disagree, side with CNN" as a surface pattern rather
# than teaching the reasoning process; pairing it with this example -- same
# vote shape, opposite correct answer -- forces the behavioural evidence,
# not which model said what, to be the thing that decides the verdict.
FEWSHOT_EXAMPLE_2 = (
    "--- EXAMPLE ---\n"
    "=== RECORD UNDER REVIEW ===\n"
    "Model votes:  XGBoost -> BENIGN (p_malicious=0.22)   |   CNN -> MALICIOUS (p_malicious=0.58)\n"
    "\n"
    "Behavioural features  (value | pctile vs benign pop | pctile vs attack pop | meaning)\n"
    "  shell_from_service     1        p85    p55    a shell was spawned from a service/daemon context\n"
    "  events_per_second      42       p90    p45    syscall throughput\n"
    "  parent_child_rarity    0.02     p10    p5     rarity of this parent->child exec pair (1 = never seen before)\n"
    "  max_uid_euid_delta     0        p40    p10    largest gap between real and effective UID; large = privilege transition\n"
    "  lifetime_seconds       4        p15    p20    process wall-clock lifetime\n"
    "  (... remaining features omitted in this example ...)\n"
    "\n"
    "Notable deviations (most unusual vs benign, first = most extreme):\n"
    "  shell_from_service, events_per_second, parent_child_rarity, lifetime_seconds, max_uid_euid_delta\n"
    "\n"
    "Known failure patterns in this dataset (from labelled error analysis):\n"
    "  - Benign rows misflagged as attack tend to have elevated parent_child_rarity and uid_changed with short lifetime_seconds.\n"
    "  - Missed attacks tend to have auid_euid_mismatch set but low max_uid_euid_delta and priv_op_count.\n"
    "\n"
    "--- IDEAL ANSWER ---\n"
    '{"verdict": "BENIGN", "confidence": 0.78, "rationale_steps": ['
    '"Privilege state: max_uid_euid_delta and auid_euid_mismatch are both low/unset -- no genuine UID/EUID transition.", '
    '"Mechanism: shell_from_service is set, but without is_suid_exec or elevated priv_op_count alongside it, a single flag is weak evidence of escalation on its own.", '
    '"Behavioural anomaly: parent_child_rarity is near 0 (a common, previously-seen exec pair) and no sensitive-path access -- not the profile of a novel attack chain.", '
    '"Benign-explanation test: regular high-throughput, short-lifetime execution from a service context is consistent with a routine health-check or monitoring script.", '
    '"Decide: the only elevated signal (shell_from_service) is explained by ordinary service activity and is not corroborated by any privilege-transition evidence; this supports XGBoost over CNN\'s flag on this one feature."], '
    '"key_features": ["shell_from_service", "events_per_second", "parent_child_rarity", "max_uid_euid_delta", "lifetime_seconds"], '
    '"agrees_with": "xgb"}\n'
    "--- END EXAMPLE ---"
)

# Combined constant used when building the live prompt (llm_triage_client.py).
# FEWSHOT_EXAMPLE itself is kept unchanged and importable on its own --
# llm_triage_ablation.py's "cnn_first" variant reads it directly.
FEWSHOT_EXAMPLES = FEWSHOT_EXAMPLE + "\n\n" + FEWSHOT_EXAMPLE_2

KNOWN_FAILURE_BLOCKS = {
    "camlds": (
        "Known failure patterns in this dataset (from labelled error analysis):\n"
        "  - Benign rows misflagged as attack (FP) tend to have elevated parent_child_rarity, "
        "uid_changed and ephemeral_privileged, often with unusually short lifetime_seconds "
        "(XGBoost); or elevated failed_call_rate, failed_call_count and exec_count (CNN).\n"
        "  - Missed attacks (FN) tend to have auid_euid_mismatch set but LOW max_uid_euid_delta "
        "and priv_op_count, zero network_count and no shell_from_service -- escalation attempts "
        "that never complete a visible UID transition."
    ),
    "casino": (
        "Known failure patterns in this dataset (from labelled error analysis):\n"
        "  - Benign rows misflagged as attack (FP) tend to show exec_count > 0 with "
        "euid_root/uid_is_root set but auid_euid_mismatch absent (XGBoost).\n"
        "  - Missed attacks (FN) tend to have elevated priv_op_count but LOW failed_call_rate, "
        "no shell_from_service, and uid_is_root unset (CNN)."
    ),
}
