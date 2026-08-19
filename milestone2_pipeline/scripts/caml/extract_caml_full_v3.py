"""
Broad-technique extraction (19 techniques), by-process sampling, host-safe
============================================================================
Combines both fixes validated on the LPE-only pipeline:
  1. PID-level sampling (not event-group/seq_id level) -- reconstructs each
     process's FULL event history before any sampling/capping happens,
     preventing the fragmentation that starved MIN_EVENTS>=2 filtering.
  2. Host tracking -- prevents PID collisions across different hosts
     within the same scenario from merging unrelated processes.

Unlike extract_caml_lpe_by_process.py, this labels ANY key-tagged event
as attack (all 19 techniques), not just the 3 strict LPE tags -- matching
the "general host anomaly, LPE emphasis" scope used in Ch1/Ch2.

Usage:
    python extract_caml_full_v3.py --benign_ratio 2.5 --min_events 2
"""

import argparse
import re
import random
import statistics
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd

CAML_ROOT   = Path('data/caml/cam_lds')
RANDOM_SEED = 42

SKIP_KEYS  = {'auditlog', '(null)', 'user_commands', 'watch_home'}
SKIP_TYPES = {'PROCTITLE'}
MAX_PER_SCENARIO_BENIGN = 15000

random.seed(RANDOM_SEED)


def find_audit_logs(root):
    files = []
    for scenario_dir in sorted(root.glob('scenario_*')):
        if scenario_dir.is_dir():
            files.extend(str(p) for p in scenario_dir.rglob('audit.log'))
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


def get_scenario_name(filepath):
    for p in Path(filepath).parts:
        if 'scenario' in p.lower():
            return p
    return 'unknown'


def get_host_name(filepath):
    parts = Path(filepath).parts
    for i, p in enumerate(parts):
        if p == 'logs' and i > 0:
            return parts[i - 1]
    return 'unknown_host'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benign_ratio', type=float, default=2.5,
        help='Sample benign PROCESSES at this multiple of available '
             'attack process count (default: 2.5x)')
    parser.add_argument('--min_events', type=int, default=2)
    parser.add_argument('--out', default='combined/raw_labeled_logs_v3.csv')
    args = parser.parse_args()

    audit_files = find_audit_logs(CAML_ROOT)
    print(f"Found {len(audit_files)} audit.log files under {CAML_ROOT}/scenario_*/")
    random.shuffle(audit_files)

    attack_processes = []  # (records, technique, scenario)
    benign_processes  = []
    single_event_attack = 0
    single_event_benign = 0

    for i, f in enumerate(audit_files):
        scenario = get_scenario_name(f)
        host     = get_host_name(f)

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
                parsed['scenario'] = scenario
                parsed['host']     = host
                seq_groups[parsed['seq_id']].append(parsed)

        # Regroup by PID -- reconstructs full process history before sampling
        pid_groups = defaultdict(list)
        for seq_id, records in seq_groups.items():
            pid = records[0].get('pid')
            if pid is None:
                continue
            pid_groups[pid].extend(records)

        for pid, records in pid_groups.items():
            # Stamp real pid onto every record (companion PATH/EXECVE
            # records don't carry their own pid= field in raw auditd)
            for r in records:
                r['pid'] = pid

            technique = None
            for r in records:
                kt = r.get('key_tag', '')
                if kt and kt not in SKIP_KEYS:
                    technique = kt
                    break

            if technique is None:
                benign_processes.append((records, 'benign', scenario))
                if len(records) == 1:
                    single_event_benign += 1
            else:
                attack_processes.append((records, technique, scenario))
                if len(records) == 1:
                    single_event_attack += 1

        if (i + 1) % 20 == 0 or i == len(audit_files) - 1:
            print(f"  [{i+1}/{len(audit_files)}] "
                  f"attack_pids={len(attack_processes):,} "
                  f"benign_pids={len(benign_processes):,}")

    print(f"\n{'='*65}")
    print("FULL CORPUS SCAN COMPLETE (PID-level, host-safe)")
    print(f"{'='*65}")
    print(f"  Attack processes: {len(attack_processes):,}  "
          f"(single-event: {single_event_attack:,}, "
          f"{100*single_event_attack/max(len(attack_processes),1):.1f}%)")
    print(f"  Benign processes: {len(benign_processes):,}  "
          f"(single-event: {single_event_benign:,}, "
          f"{100*single_event_benign/max(len(benign_processes),1):.1f}%)")

    attack_lengths = [len(r) for r, _, _ in attack_processes]
    benign_lengths = [len(r) for r, _, _ in benign_processes]
    print(f"\n  Attack process length: median={statistics.median(attack_lengths):.0f} "
          f"mean={statistics.mean(attack_lengths):.1f} max={max(attack_lengths)}")
    print(f"  Benign process length: median={statistics.median(benign_lengths):.0f} "
          f"mean={statistics.mean(benign_lengths):.1f} max={max(benign_lengths)}")

    print(f"\n  Technique breakdown (top 15):")
    for tech, count in Counter(t for _, t, _ in attack_processes).most_common(15):
        print(f"    {tech:<50} {count:>8,}")

    # MIN_EVENTS filter
    attack_viable = [(r, t, s) for r, t, s in attack_processes if len(r) >= args.min_events]
    benign_viable = [(r, t, s) for r, t, s in benign_processes if len(r) >= args.min_events]
    print(f"\n  After MIN_EVENTS>={args.min_events} filter:")
    print(f"    Attack viable: {len(attack_viable):,} / {len(attack_processes):,} "
          f"({100*len(attack_viable)/max(len(attack_processes),1):.1f}%)")
    print(f"    Benign viable: {len(benign_viable):,} / {len(benign_processes):,} "
          f"({100*len(benign_viable)/max(len(benign_processes),1):.1f}%)")

    # Cap benign per scenario
    rng = random.Random(RANDOM_SEED + 1)
    by_scenario = defaultdict(list)
    for records, technique, scenario in benign_viable:
        by_scenario[scenario].append((records, technique, scenario))
    benign_capped = []
    print(f"\n  Per-scenario benign process counts (cap={MAX_PER_SCENARIO_BENIGN:,}):")
    for scenario, items in sorted(by_scenario.items(), key=lambda x: -len(x[1])):
        rng.shuffle(items)
        taken = items[:MAX_PER_SCENARIO_BENIGN]
        benign_capped.extend(taken)
        flag = " <- CAPPED" if len(items) > MAX_PER_SCENARIO_BENIGN else ""
        print(f"    {scenario:<40} {len(items):>8,} -> {len(taken):>8,}{flag}")

    # Cap attack per TECHNIQUE (symmetric fix -- prevents T1043/T1078 from
    # drowning out rarer techniques like T1068/T1169, and reduces the
    # duplicate-vector collapse concentrated in the highest-volume classes)
    rng2 = random.Random(RANDOM_SEED + 2)
    by_technique = defaultdict(list)
    for records, technique, scenario in attack_viable:
        by_technique[technique].append((records, technique, scenario))
    attack_capped = []
    print(f"\n  Per-technique attack process counts (cap={MAX_PER_SCENARIO_BENIGN:,}):")
    for technique, items in sorted(by_technique.items(), key=lambda x: -len(x[1])):
        rng2.shuffle(items)
        taken = items[:MAX_PER_SCENARIO_BENIGN]
        attack_capped.extend(taken)
        flag = " <- CAPPED" if len(items) > MAX_PER_SCENARIO_BENIGN else ""
        print(f"    {technique:<50} {len(items):>8,} -> {len(taken):>8,}{flag}")

    # Sample benign at requested ratio relative to the CAPPED attack count
    n_attack = len(attack_capped)
    target_benign = int(n_attack * args.benign_ratio)
    benign_sample = random.sample(
        benign_capped, min(target_benign, len(benign_capped)))

    print(f"\n{'='*65}")
    print(f"FINAL SAMPLE (benign_ratio={args.benign_ratio}x, MIN_EVENTS={args.min_events})")
    print(f"{'='*65}")
    print(f"  Attack (ALL viable): {n_attack:,}")
    print(f"  Benign (target {target_benign:,}, capped pool {len(benign_capped):,}): "
          f"{len(benign_sample):,}")
    if len(benign_sample) < target_benign:
        print(f"  ⚠️  Benign pool exhausted before reaching target ratio")

    rows = []
    for groups, label in [(benign_sample, 0), (attack_capped, 1)]:
        for records, technique, _scenario in groups:
            for r in records:
                r['label']     = label
                r['technique'] = technique
                rows.append(r)

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
