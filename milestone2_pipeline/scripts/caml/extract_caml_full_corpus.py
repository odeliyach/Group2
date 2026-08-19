"""
Full-corpus CAM-LDS extraction — scenario_* + manifestations_filtered
================================================================================
extract_caml_full_v3.py only ever scanned the 28 scenario_* directories
(98 audit.log files total). manifestations_filtered/sequences/ contains
~2,252 additional audit.log files -- independent repeated TRIAL executions
of the same scripted scenarios (e.g. 19 separate trials of
"3_ssh_healthcheck" alone, confirmed via distinct MD5 hashes against the
currently-used scenario_3_ssh_healthcheck capture -- genuinely additive,
not duplicated).

Format and host-detection logic were verified compatible: the audit.log
lines use the identical type=/msg=audit()/key="<TECHNIQUE>" convention
parse_line() already expects, and manifestations_filtered's
<host>/logs/log/audit/audit.log path structure matches get_host_name()'s
existing "look for a 'logs' path segment, take the preceding component"
logic without any changes needed there.

What's NEW here vs. extract_caml_full_v3.py:
  1. Scans BOTH scenario_* (original, 98 files) AND
     manifestations_filtered/sequences/* (new, ~2,252 files).
  2. Trial-grouped directory names (e.g. "3_ssh_healthcheck-33_34",
     "1_autostart_pam-24") are normalized to a scenario FAMILY label by
     stripping the trailing "-<trial_numbers>" suffix, so per-scenario
     capping/stratification groups all trials of the same underlying
     scenario together (matching the original script's per-scenario
     benign cap and per-technique attack cap logic).
  3. Reuses extract_caml_realistic_ratio.py's --target_total_size /
     --target_attack_rate modes and stratified subsampling, since the
     corpus is now large enough that BOTH a ~90k+ total size AND a much
     lower attack rate may actually be jointly achievable (unlike with
     the scenario_*-only corpus, where the benign ceiling made that
     mathematically impossible -- see CHANGELOG.md).

NOT YET INCLUDED: manifestations_raw/ (5,753 files, different internal
organization -- sequences/steps/techniques -- not yet explored for
overlap with manifestations_filtered). Left as future work; mixing it in
without first checking for redundancy against manifestations_filtered
risks double-counting.

Usage:
    python extract_caml_full_corpus.py \\
        --target_total_size 90805 \\
        --min_events 2 \\
        --out combined_v3/raw_labeled_logs_full_corpus.csv
"""

import argparse
import re
import random
import statistics
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd

CAML_ROOT = Path('data/caml/cam_lds')
RANDOM_SEED = 42

SKIP_KEYS  = {'auditlog', '(null)', 'user_commands', 'watch_home'}
SKIP_TYPES = {'PROCTITLE'}

random.seed(RANDOM_SEED)

# Matches a trailing trial-number suffix like "-1", "-33_34", "-19_20_21_22_23_24"
TRIAL_SUFFIX_RE = re.compile(r'-[\d_]+$')


def scenario_family(name: str) -> str:
    """'3_ssh_healthcheck-33_34' -> '3_ssh_healthcheck'.
    'scenario_4' (no trial suffix) -> unchanged."""
    return TRIAL_SUFFIX_RE.sub('', name)


def find_audit_logs_scenario_dirs(root):
    """Original source: scenario_*/ directories."""
    files = []
    for scenario_dir in sorted(root.glob('scenario_*')):
        if scenario_dir.is_dir():
            for p in scenario_dir.rglob('audit.log'):
                files.append((str(p), scenario_family(scenario_dir.name)))
    return files


def find_audit_logs_manifestations(root):
    """New source: manifestations_filtered/sequences/<scenario>-<trial(s)>/."""
    seq_root = root / 'manifestations_filtered' / 'sequences'
    files = []
    if not seq_root.exists():
        print(f"  WARNING: {seq_root} not found -- skipping manifestations_filtered source")
        return files
    for trial_dir in sorted(seq_root.iterdir()):
        if trial_dir.is_dir():
            family = scenario_family(trial_dir.name)
            for p in trial_dir.rglob('audit.log'):
                files.append((str(p), family))
    return files


def parse_line(line):
    r = {}
    m = re.match(r'type=(\S+)', line)
    if not m:
        return None
    r['record_type'] = m.group(1)
    if r['record_type'] in SKIP_TYPES:
        return None
    m = re.search(r'msg=audit\(([\d.]+):(\d+)\)', line)
    if not m:
        return None
    r['timestamp'] = float(m.group(1))
    r['seq_id']    = int(m.group(2))
    for field in ['pid', 'ppid', 'uid', 'euid', 'auid', 'gid']:
        m2 = re.search(rf'\b{field}=(\d+)', line)
        if m2:
            r[field] = int(m2.group(1))
    for field in ['comm', 'exe', 'name', 'acct', 'res']:
        m2 = re.search(rf'\b{field}="([^"]*)"', line)
        if m2:
            r[field] = m2.group(1)
    m2 = re.search(r'SYSCALL=(\w+)', line)
    r['syscall_name'] = m2.group(1) if m2 else r['record_type']
    m2 = re.search(r'key="?([^"\s]+)"?', line)
    if m2:
        r['key_tag'] = m2.group(1)
    r['success'] = 'success=yes' in line or 'res=success' in line
    r['failed']  = int(not r['success'])
    return r


def get_host_name(filepath):
    """Unchanged from extract_caml_full_v3.py -- already compatible with
    manifestations_filtered's <trial>/<host>/logs/log/audit/audit.log
    structure, confirmed by direct inspection."""
    parts = Path(filepath).parts
    for i, p in enumerate(parts):
        if p == 'logs' and i > 0:
            return parts[i - 1]
    return 'unknown_host'


def stratified_subsample(items_by_technique, target_total, rng):
    total_available = sum(len(v) for v in items_by_technique.values())
    if target_total >= total_available:
        return [item for items in items_by_technique.values() for item in items]

    result = []
    remaining_target = target_total
    techniques = list(items_by_technique.items())
    for i, (technique, items) in enumerate(techniques):
        share = len(items) / total_available
        n_take = remaining_target if i == len(techniques) - 1 else round(target_total * share)
        n_take = min(n_take, len(items))
        rng.shuffle(items)
        result.extend(items[:n_take])
        remaining_target -= n_take
    return result


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--target_attack_rate', type=float,
        help='Hit this exact attack rate; total size is a consequence.')
    mode.add_argument('--target_total_size', type=int,
        help='Hit this total row count; attack rate is a consequence '
             '(though with the full corpus, both may land close to target).')
    parser.add_argument('--max_benign_per_scenario', type=int, default=100000,
        help='Per-scenario-FAMILY benign cap (raised well above the old 15000 '
             '-- the full corpus is large enough that this should rarely bind).')
    parser.add_argument('--min_events', type=int, default=2)
    parser.add_argument('--sources', default='scenario,manifestations',
        help='Comma-separated: scenario, manifestations (raw not yet supported).')
    parser.add_argument('--out', default='combined_v3/raw_labeled_logs_full_corpus.csv')
    args = parser.parse_args()

    sources = set(s.strip() for s in args.sources.split(','))
    audit_files = []
    if 'scenario' in sources:
        found = find_audit_logs_scenario_dirs(CAML_ROOT)
        print(f"Found {len(found)} audit.log files under scenario_*/")
        audit_files.extend(found)
    if 'manifestations' in sources:
        found = find_audit_logs_manifestations(CAML_ROOT)
        print(f"Found {len(found)} audit.log files under manifestations_filtered/sequences/")
        audit_files.extend(found)

    print(f"Total audit.log files to scan: {len(audit_files):,}")
    rng_shuffle = random.Random(RANDOM_SEED)
    rng_shuffle.shuffle(audit_files)

    attack_processes = []
    benign_processes  = []

    for i, (f, family) in enumerate(audit_files):
        host = get_host_name(f)

        seq_groups = defaultdict(list)
        with open(f, errors='ignore') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parsed = parse_line(line)
                if not parsed:
                    continue
                parsed['source']   = 'CAM-LDS'
                parsed['scenario'] = family
                parsed['host']     = host
                seq_groups[parsed['seq_id']].append(parsed)

        pid_groups = defaultdict(list)
        for seq_id, records in seq_groups.items():
            pid = records[0].get('pid')
            if pid is None:
                continue
            pid_groups[pid].extend(records)

        for pid, records in pid_groups.items():
            for r in records:
                r['pid'] = pid

            technique = None
            for r in records:
                kt = r.get('key_tag', '')
                if kt and kt not in SKIP_KEYS:
                    technique = kt
                    break

            if technique is None:
                benign_processes.append((records, 'benign', family))
            else:
                attack_processes.append((records, technique, family))

        if (i + 1) % 200 == 0 or i == len(audit_files) - 1:
            print(f"  [{i+1}/{len(audit_files)}] "
                  f"attack_pids={len(attack_processes):,} "
                  f"benign_pids={len(benign_processes):,}")

    print(f"\n{'='*65}\nFULL CORPUS SCAN COMPLETE\n{'='*65}")
    print(f"  Attack processes: {len(attack_processes):,}")
    print(f"  Benign processes: {len(benign_processes):,}")

    attack_viable = [(r, t, s) for r, t, s in attack_processes if len(r) >= args.min_events]
    benign_viable = [(r, t, s) for r, t, s in benign_processes if len(r) >= args.min_events]
    print(f"\n  After MIN_EVENTS>={args.min_events} filter:")
    print(f"    Attack viable: {len(attack_viable):,}")
    print(f"    Benign viable: {len(benign_viable):,}")

    rng = random.Random(RANDOM_SEED + 1)
    by_scenario = defaultdict(list)
    for records, technique, family in benign_viable:
        by_scenario[family].append((records, technique, family))
    benign_capped = []
    print(f"\n  Per-scenario-family benign process counts (cap={args.max_benign_per_scenario:,}):")
    for family, items in sorted(by_scenario.items(), key=lambda x: -len(x[1])):
        rng.shuffle(items)
        taken = items[:args.max_benign_per_scenario]
        benign_capped.extend(taken)
        flag = " <- CAPPED" if len(items) > args.max_benign_per_scenario else ""
        print(f"    {family:<40} {len(items):>8,} -> {len(taken):>8,}{flag}")

    by_technique = defaultdict(list)
    for records, technique, family in attack_viable:
        by_technique[technique].append((records, technique, family))
    print(f"\n  Attack processes by technique:")
    for technique, items in sorted(by_technique.items(), key=lambda x: -len(x[1])):
        print(f"    {technique:<50} {len(items):>8,}")

    n_benign = len(benign_capped)
    rng2 = random.Random(RANDOM_SEED + 2)

    if args.target_attack_rate is not None:
        r = args.target_attack_rate
        target_attack = int(round(n_benign * r / (1 - r)))
        if target_attack < len(attack_viable):
            attack_final = stratified_subsample(by_technique, target_attack, rng2)
            benign_final = benign_capped
        else:
            target_benign = int(round(len(attack_viable) * (1 - r) / r))
            benign_final = rng2.sample(benign_capped, min(target_benign, len(benign_capped)))
            attack_final = attack_viable
    else:
        target_total = args.target_total_size
        benign_final = benign_capped
        attack_needed = target_total - len(benign_final)
        if attack_needed <= 0:
            raise ValueError(f"target_total_size ({target_total:,}) <= benign pool alone "
                              f"({len(benign_final):,}).")
        if attack_needed > len(attack_viable):
            print(f"  WARNING: need {attack_needed:,} attack rows but only "
                  f"{len(attack_viable):,} viable -- using all attack, "
                  f"final total will be short of target.")
            attack_final = attack_viable
        else:
            attack_final = stratified_subsample(by_technique, attack_needed, rng2)

    print(f"\n{'='*65}\nFINAL SAMPLE\n{'='*65}")
    print(f"  Attack: {len(attack_final):,}")
    print(f"  Benign: {len(benign_final):,}")
    total = len(attack_final) + len(benign_final)
    print(f"  Total:  {total:,}")
    print(f"  Achieved attack rate: {len(attack_final)/total:.1%}")

    rows = []
    for groups, label in [(benign_final, 0), (attack_final, 1)]:
        for records, technique, _family in groups:
            for rec in records:
                rec['label']     = label
                rec['technique'] = technique
                rows.append(rec)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\nFinal dataset:")
    print(f"  Total rows:  {len(df):,}")
    print(f"  Benign (0):  {(df['label']==0).sum():,}")
    print(f"  Attack (1):  {(df['label']==1).sum():,}")
    print(f"\nTechnique distribution (attack, top 10):")
    print(df[df['label']==1]['technique'].value_counts().head(10).to_string())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
