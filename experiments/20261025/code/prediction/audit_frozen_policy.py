"""Read-only provenance audit. No predictor fitting or performance timing."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True, type=Path)
    ap.add_argument('--summary', required=True, type=Path)
    ap.add_argument('--checkpoint-dir', required=True, type=Path)
    ap.add_argument('--source-freeze', required=True, type=Path)
    ap.add_argument('--profiles-dir', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('do not overwrite an audit')
    policy = json.loads(args.policy.read_text())
    summary = json.loads(args.summary.read_text())
    freeze = json.loads(args.source_freeze.read_text())
    checks = {}
    checks['policy_kind'] = policy['kind'] == 'validation_frozen_active_fallback_policy'
    checks['source_summary_hash'] = policy['source_summary_sha256'] == sha(args.summary)
    checks['target_labels_not_exported'] = policy['target_latencies_or_test_errors_in_policy'] is False
    checks['source_evaluator_hash'] = policy['evaluation_code_sha256'] == summary['code_sha256'] == sha(args.source_freeze.parent / 'evaluate_profiles.py')
    checks['all_frozen_sources_unchanged'] = all(sha(args.source_freeze.parent / name) == expected for name, expected in freeze['files'].items())
    rows = [r for r in summary['results'] if all(r[k] == policy[k] for k in ('method', 'feature_mode', 'split_id'))]
    if len(rows) != 1:
        raise ValueError('policy selector does not identify exactly one summary row')
    row = rows[0]
    checks['threshold_exact_match'] = row['fallback']['threshold'] == policy['threshold'] and row['fallback']['threshold_infinite'] == policy['threshold_infinite']
    checks['threshold_validation_only'] = row['fallback']['threshold_selected_only_on_validation'] is True and policy['threshold_selected_only_on_validation'] is True
    checks['validation_measurement_budget_respected'] = row['fallback']['validation_measured_fraction'] <= policy['validation_budget_fraction'] + 1e-12
    train, val, test = (set(policy[k]) for k in ('train_model_ids', 'validation_model_ids', 'test_model_ids'))
    checks['model_partitions_disjoint'] = not (train & val or train & test or val & test)
    checkpoint_audits = []
    for item in policy['checkpoints']:
        local = args.checkpoint_dir / Path(item['path']).name
        ck = torch.load(local, map_location='cpu', weights_only=True)
        entry = {'seed': item['seed'], 'path': str(local.resolve()), 'sha256': sha(local)}
        entry['checks'] = {
            'hash_matches_policy': entry['sha256'] == item['sha256'],
            'seed_matches': ck['seed'] == item['seed'],
            'method_matches': ck['method'] == policy['method'],
            'mode_matches': ck['feature_mode'] == policy['feature_mode'],
            'train_ids_match': ck['training_model_ids'] == policy['train_model_ids'],
            'validation_ids_match': ck['validation_model_ids'] == policy['validation_model_ids'],
            'test_ids_match': ck['test_model_ids'] == policy['test_model_ids'],
            'split_matches': ck['split_id'] == policy['split_id'],
            'source_physical_hashes_match': [x['sha256'] for x in ck['source_manifest']] == policy['source_physical_hashes'],
            'source_metadata_hashes_match': [x['metadata_sidecar_sha256'] for x in ck['source_manifest']] == policy['source_metadata_hashes'],
        }
        checkpoint_audits.append(entry)
    checks['all_checkpoint_checks'] = all(all(x['checks'].values()) for x in checkpoint_audits)
    checks['five_prespecified_seeds'] = [x['seed'] for x in policy['checkpoints']] == [11, 22, 33, 44, 55]
    physical = []
    for item in summary['source_manifest']:
        local = args.profiles_dir / Path(item['path']).name
        side = args.profiles_dir / Path(item['metadata_sidecar']).name
        physical.append({'file': str(local.resolve()), 'sha256': sha(local),
                         'raw_hash_matches': sha(local) == item['sha256'],
                         'sidecar_hash_matches': sha(side) == item['metadata_sidecar_sha256'],
                         'runtime_stack_id': item['hardware']})
    checks['all_source_profiles_unchanged'] = all(x['raw_hash_matches'] and x['sidecar_hash_matches'] for x in physical)
    out = {'kind': 'read_only_frozen_policy_provenance_audit',
           'created_utc': datetime.now(timezone.utc).isoformat(),
           'auditor_code_sha256': sha(__file__),
           'policy_sha256': sha(args.policy), 'summary_sha256': sha(args.summary),
           'checks': checks, 'checkpoint_audits': checkpoint_audits,
           'source_profiles': physical, 'all_automated_checks_pass': all(checks.values()),
           'threshold': policy['threshold'], 'train_model_ids': sorted(train),
           'validation_model_ids': sorted(val), 'test_model_ids': sorted(test),
           'scope': 'Static artifact and checkpoint audit only. No model training, target inference, hardware timing, or new performance claim.',
           'manual_source_review': [
               'NPU extension import precedes torch.device creation.',
               'Unlabelled graph construction does not read operator-latency targets; y is NaN.',
               'Decisions are written and hashed before timed() is called for selected queries.',
               'Stage-two model/input hash must match stage-one hash before fresh timing.',
               'Existing checkpoint source paths are explicitly rebased only for this local audit; deployment policy paths remain unchanged.',
               'A source-freeze manifest validates dependency hashes; active_fallback does not independently re-check every imported source hash at runtime.',
               'No-trigger or failed-measurement outcomes must remain explicit; a policy is not retuned on target outcomes.'
           ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({'output': str(args.output), 'passed': out['all_automated_checks_pass'], 'checks': len(checks), 'checkpoints': len(checkpoint_audits)}))
    if not out['all_automated_checks_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
