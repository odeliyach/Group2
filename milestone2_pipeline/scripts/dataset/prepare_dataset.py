"""
Dataset Preparation Pipeline — CAM-LDS only schema
====================================================
Input schema (21 columns):
  record_type, timestamp, seq_id, pid, ppid, uid, euid, auid, gid,
  comm, exe, syscall_name, key_tag, success, failed, source,
  scenario, label, technique, name, acct

Produces:
  dataset1_fixed.csv        — cleaned raw events, NaN-free
  dataset1_features.csv     — one row per process, 20 features
  sequences_X.npy           — (N, MAX_LEN, 7) float32 for LSTM
  sequences_y.npy           — labels
  sequences_meta.csv        — process metadata
  vocab.json                — syscall name -> integer

Usage:
    python prepare_dataset.py --csv combined/raw_labeled_logs.csv --out combined/
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
MAX_SEQ_LEN = 50
MIN_EVENTS  = 2

TOKEN_FEATURES = [
    'syscall_int', 'uid_class', 'failed',
    'is_priv', 'is_exec', 'is_file', 'is_network',
]

SENSITIVE_PATHS = {
    '/etc/shadow', '/etc/passwd', '/etc/sudoers',
    '/root/', '/.ssh/authorized_keys', '/proc/mem',
}

SHELL_NAMES    = {'bash','sh','dash','zsh','fish','csh','tcsh'}
SERVICE_NAMES  = {'apache2','nginx','vsftpd','sshd','mysqld','postgres',
                  'php-fpm','httpd','www-data','apache','tomcat'}
PRIV_SYSCALLS  = {'setuid','setgid','setreuid','setregid','setresuid',
                  'setresgid','capset','ptrace','chown','fchown','lchown',
                  'fchownat','chmod','fchmod','fchmodat'}
EXEC_SYSCALLS  = {'execve','execveat'}
FILE_SYSCALLS  = {'openat','open','creat','rename','renameat','unlink',
                  'unlinkat','mkdir','mkdirat','rmdir','link','symlink'}
NET_SYSCALLS   = {'connect','bind','accept','accept4','socket'}


# ── Step 1 — Fix raw CSV ──────────────────────────────────────────────────────

def fix_raw(df):
    print(f"  Columns: {df.columns.tolist()}")

    # Integer fields — fill NaN with neutral defaults
    df['pid']  = df['pid'].fillna(0).astype(int)
    df['ppid'] = df['ppid'].fillna(0).astype(int)
    df['uid']  = df['uid'].fillna(0).astype(int)
    df['euid'] = df['euid'].fillna(df['uid']).astype(int)
    df['auid'] = df['auid'].fillna(4294967295).astype('int64')
    df['gid']  = df['gid'].fillna(df['uid']).astype(int)
    df['seq_id'] = df['seq_id'].fillna(0).astype(int)

    # String fields
    df['exe']          = df['exe'].fillna('unknown')
    df['comm']         = df['comm'].fillna(
        df['exe'].apply(lambda x: Path(x).name if isinstance(x, str) else 'unknown'))
    df['syscall_name'] = df['syscall_name'].fillna(
        df['record_type'].fillna('unknown'))

    # failed — already 0/1, just fill any NaN
    df['failed'] = df['failed'].fillna(0).astype(int)

    # Drop columns not needed for modeling
    if 'host' not in df.columns:
        df['host'] = 'unknown_host'
    df['host'] = df['host'].fillna('unknown_host')
    drop_cols = ['success', 'key_tag', 'name', 'acct']  # keep 'scenario' and 'host' -- needed for correct process grouping
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Verify core fields
    core = ['timestamp', 'pid', 'uid', 'euid', 'failed', 'label']
    nan_counts = df[core].isnull().sum()
    if nan_counts.any():
        print(f"  WARNING NaN in core: {nan_counts[nan_counts>0].to_dict()}")
        df[core] = df[core].fillna(0)
    else:
        print(f"  ✅ No NaN in core fields")

    print(f"  Final columns ({len(df.columns)}): {df.columns.tolist()}")
    return df


# ── Step 2 — Feature helpers ──────────────────────────────────────────────────

def exe_basename(exe_str):
    if not isinstance(exe_str, str) or exe_str in ('unknown', ''):
        return ''
    return Path(exe_str).name.lower()

def is_sensitive(exe_str):
    if not isinstance(exe_str, str):
        return 0
    return int(any(exe_str.startswith(p) for p in SENSITIVE_PATHS))

def uid_class(uid_val):
    if uid_val == 0:   return 0
    if uid_val < 1000: return 1
    return 2

def build_pair_rarity(df):
    pair_counts = Counter()
    for _, row in df.iterrows():
        child  = exe_basename(str(row.get('exe', '')))
        parent = exe_basename(str(row.get('exe', ''))) \
                 if row.get('ppid', 0) != 0 else '__root__'
        if child:
            pair_counts[(parent, child)] += 1
    total = sum(pair_counts.values())
    return pair_counts, total


# ── Step 3 — Process aggregation ─────────────────────────────────────────────

def aggregate_process(group, pair_counts, total_pairs):
    events   = group.sort_values('timestamp')
    n        = len(events)
    t_min    = events['timestamp'].min()
    t_max    = events['timestamp'].max()
    duration = max(t_max - t_min, 0.001)

    uid_vals  = events['uid'].values
    euid_vals = events['euid'].values
    auid_vals = events['auid'].values
    sysc_vals = events['syscall_name'].values
    exe_vals  = events['exe'].values
    fail_vals = events['failed'].values

    uid_is_root        = int((uid_vals == 0).any())
    uid_changed        = int(len(set(uid_vals.tolist())) > 1)
    euid_root          = int((euid_vals == 0).any())
    max_uid_euid_delta = float(max(abs(int(e) - int(u))
                                   for u, e in zip(uid_vals, euid_vals)))
    is_suid_exec       = int(any(u != 0 and e == 0
                                 for u, e in zip(uid_vals, euid_vals)))
    auid_mismatch      = int(any(
        int(a) < 65534 and int(e) == 0 and int(a) != int(e)
        for a, e in zip(auid_vals, euid_vals)
    ))
    ephemeral_privileged = int(duration < 2.0 and uid_is_root)
    events_per_second    = n / duration
    failed_rate          = float(fail_vals.mean())
    failed_count         = int(fail_vals.sum())

    priv_op_count = int(sum(1 for s in sysc_vals if s in PRIV_SYSCALLS))
    exec_count    = int(sum(1 for s in sysc_vals if s in EXEC_SYSCALLS))
    file_count    = int(sum(1 for s in sysc_vals if s in FILE_SYSCALLS))
    net_count     = int(sum(1 for s in sysc_vals if s in NET_SYSCALLS))

    proc_names         = [exe_basename(e) for e in exe_vals if e]
    proc_name          = proc_names[0] if proc_names else 'unknown'
    shell_from_service = int(
        any(p in SHELL_NAMES for p in proc_names) and
        any(p in SERVICE_NAMES for p in proc_names)
    )
    sensitive_path_access = int(any(is_sensitive(e) for e in exe_vals))

    ppid_val = int(events['ppid'].iloc[0]) if events['ppid'].iloc[0] else 0
    parent   = '__root__' if ppid_val == 0 else proc_name
    count    = pair_counts.get((parent, proc_name), 0)
    rarity   = round(1.0 - count / max(total_pairs, 1), 6)

    return {
        'lifetime_seconds':      round(duration, 3),
        'events_per_second':     round(events_per_second, 3),
        'seq_length':            n,
        'unique_syscalls':       int(pd.Series(sysc_vals).nunique()),
        'uid_is_root':           uid_is_root,
        'uid_changed':           uid_changed,
        'euid_root':             euid_root,
        'max_uid_euid_delta':    round(max_uid_euid_delta, 1),
        'is_suid_exec':          is_suid_exec,
        'auid_euid_mismatch':    auid_mismatch,
        'ephemeral_privileged':  ephemeral_privileged,
        'failed_call_rate':      round(failed_rate, 4),
        'failed_call_count':     failed_count,
        'priv_op_count':         priv_op_count,
        'exec_count':            exec_count,
        'file_access_count':     file_count,
        'network_count':         net_count,
        'shell_from_service':    shell_from_service,
        'sensitive_path_access': sensitive_path_access,
        'parent_child_rarity':   rarity,
        'label':                 int(group['label'].max()),
        'source':                group['source'].iloc[0],
        'technique':             (group[group['label'] == 1]['technique'].iloc[0]
                                    if (group['label'] == 1).any()
                                    else group['technique'].iloc[0]),
        'proc_name':             proc_name,
    }


# ── Step 4 — Sequence encoding ────────────────────────────────────────────────

def build_vocab(df):
    counts = Counter(df['syscall_name'].dropna().values)
    vocab  = {'<PAD>': 0, '<UNK>': 1}
    for i, (name, _) in enumerate(counts.most_common(), start=2):
        vocab[name] = i
    return vocab

def event_to_token(row, vocab):
    syscall = str(row['syscall_name'])
    uid     = int(row['uid'])
    return [
        vocab.get(syscall, 1),
        uid_class(uid),
        int(row['failed']),
        int(syscall in PRIV_SYSCALLS),
        int(syscall in EXEC_SYSCALLS),
        int(syscall in FILE_SYSCALLS),
        int(syscall in NET_SYSCALLS),
    ]

def encode_sequences(groups, vocab):
    n_feat = len(TOKEN_FEATURES)
    X_list, y_list, meta_list = [], [], []
    for group_key, group in groups:
        if len(group) < MIN_EVENTS:
            continue
        events = group.sort_values('timestamp')
        tokens = [event_to_token(row, vocab) for _, row in events.iterrows()]
        tokens = tokens[-MAX_SEQ_LEN:]
        matrix = np.zeros((MAX_SEQ_LEN, n_feat), dtype=np.float32)
        offset = MAX_SEQ_LEN - len(tokens)
        for i, tok in enumerate(tokens):
            matrix[offset + i] = tok
        X_list.append(matrix)
        y_list.append(int(group['label'].max()))
        meta_list.append({
            'group_key': group_key,
            'label':     int(group['label'].max()),
            'source':    group['source'].iloc[0],
            'technique': (group[group['label'] == 1]['technique'].iloc[0]
                         if (group['label'] == 1).any()
                         else group['technique'].iloc[0]),
            'n_events':  len(group),
        })
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, pd.DataFrame(meta_list)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max_seq_len', type=int, default=MAX_SEQ_LEN)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Dataset Preparation Pipeline")
    print("=" * 65)
    print(f"Input:  {args.csv}")
    print(f"Output: {out}/")

    # Load
    print(f"\n[1/5] Loading {args.csv}...")
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"  Rows: {len(df):,}  Cols: {len(df.columns)}")
    print(f"  Benign: {(df['label']==0).sum():,}  Attack: {(df['label']==1).sum():,}")

    # Fix
    print(f"\n[2/5] Fixing raw CSV...")
    df = fix_raw(df)
    fixed_path = out / 'dataset1_fixed.csv'
    df.to_csv(fixed_path, index=False)
    print(f"  ✅ Saved: {fixed_path}  ({len(df):,} rows, {len(df.columns)} cols)")

    # Group
    print(f"\n[3/5] Grouping by process...")
    # Group by source + pid — all events from same process instance
    # This gives the full behavioral sequence per process lifetime
    df['_group_key'] = (
        df['source'] + '_' +
        df['scenario'].fillna('unknown') + '_' +
        df['host'].fillna('unknown_host') + '_' +
        df['pid'].astype(str)
    )
    groups       = df.groupby('_group_key', sort=False)
    valid_groups = [(k, g) for k, g in groups if len(g) >= MIN_EVENTS]
    print(f"  Total groups:         {len(groups):,}")
    print(f"  Viable (>={MIN_EVENTS} events): {len(valid_groups):,}")
    sizes = [len(g) for _, g in valid_groups]
    if sizes:
        print(f"  Events/group: median={int(np.median(sizes))} max={max(sizes)}")

    # Feature matrix
    print(f"\n[4/5] Building feature matrix...")
    pair_counts, total_pairs = build_pair_rarity(df)
    feat_rows = []
    for i, (key, group) in enumerate(valid_groups):
        feat_rows.append(aggregate_process(group, pair_counts, total_pairs))
        if (i + 1) % 20000 == 0:
            print(f"  {i+1:,} / {len(valid_groups):,}...")

    features_df = pd.DataFrame(feat_rows)
    model_cols  = [c for c in features_df.columns
                   if c not in ('label','source','technique','proc_name')]
    features_df[model_cols] = features_df[model_cols].fillna(0)

    feat_path = out / 'dataset1_features.csv'
    features_df.to_csv(feat_path, index=False)
    benign_f = (features_df['label']==0).sum()
    attack_f = (features_df['label']==1).sum()
    print(f"  ✅ Saved: {feat_path}")
    print(f"     Rows: {len(features_df):,}  Benign: {benign_f:,}  Attack: {attack_f:,}")

    print(f"\n  Feature means (benign vs attack):")
    print(f"  {'Feature':<25} {'Benign':>10} {'Attack':>10} {'Ratio':>8}")
    print(f"  {'─'*56}")
    key_feats = ['events_per_second','is_suid_exec','auid_euid_mismatch',
                 'failed_call_rate','priv_op_count','exec_count',
                 'ephemeral_privileged','sensitive_path_access',
                 'parent_child_rarity','max_uid_euid_delta',
                 'uid_is_root','unique_syscalls']
    for col in key_feats:
        if col in features_df.columns:
            bm = features_df[features_df['label']==0][col].mean()
            am = features_df[features_df['label']==1][col].mean()
            ratio = am/bm if bm > 0.0001 else float('inf')
            print(f"  {col:<25} {bm:>10.4f} {am:>10.4f} {ratio:>8.2f}x")

    # Sequences
    print(f"\n[5/5] Building LSTM sequences (max_len={args.max_seq_len})...")
    vocab = build_vocab(df)
    with open(out / 'vocab.json', 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"  Vocabulary: {len(vocab)} syscall names")

    seq_groups = df.groupby('_group_key', sort=False)
    X, y, meta_df = encode_sequences(seq_groups, vocab)
    np.save(out / 'sequences_X.npy', X)
    np.save(out / 'sequences_y.npy', y)
    meta_df.to_csv(out / 'sequences_meta.csv', index=False)
    print(f"  ✅ X shape: {X.shape}  attack rate: {y.mean():.3f}")

    # Summary
    print(f"\n{'='*65}")
    print("COMPLETE")
    print(f"{'='*65}")
    for f in sorted(out.iterdir()):
        mb = f.stat().st_size / 1e6
        print(f"  {f.name:<40} {mb:>7.1f} MB")

    print(f"""
Load for training:
  RF/XGBoost:  df = pd.read_csv('{feat_path}')
  LSTM:        X = np.load('{out}/sequences_X.npy')  # {X.shape}
               y = np.load('{out}/sequences_y.npy')
""")


if __name__ == '__main__':
    main()
