"""Independently recompute saved physical quality-gate/rollback records.

No GPU computation; this checks persisted labels, predictions and arithmetic.
Logit allclose is attested by the execution record, not reconstructible from a
hash alone, and physical authenticity also needs the worker/job receipts.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_timing(record):
    values = record['latency_samples_ms']
    assert len(values) == 30 and all(math.isfinite(v) and v > 0 for v in values)
    assert abs(statistics.mean(values) - record['mean_ms']) < 1e-9
    assert abs(statistics.median(values) - record['median_ms']) < 1e-9
    assert abs(statistics.stdev(values) - record['sample_sd_ms']) < 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', type=Path, nargs='+', required=True)
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    manifest_path = args.assets / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    samples = {s['sample_id']: s for s in manifest['samples']}
    records = []
    for path in args.files:
        d = json.loads(path.read_text())
        assert d['kind'] == 'physical_labeled_fault_injection'
        assert d['input_assets_manifest_sha256'] == sha(manifest_path)
        assert d['source_sha256'] == sha(Path(__file__).with_name('quality_stress.py'))
        task = d['task']
        assert task['quality_definition']['mode'] == 'retention_ratio'
        assert task['quality_definition']['threshold'] == .99
        split_samples = {}
        for split, partition in [('train', 'calibration'), ('eval', 'evaluation')]:
            assert len(task[split]) == 1
            block = task[split][0]
            assert block['provenance'] == 'physical_measurement'
            assert block['quality_partition'] == partition
            expected_ids = {i for i, s in samples.items() if s['partition'] == partition}
            assert len(expected_ids) == 100
            rows = {r['mask']: r for r in block['coalitions']}
            assert set(rows) == {0, 1}
            for mask, row in rows.items():
                assert row['status'] == 'ok' and row['quality_evaluated'] is True
                assert len(row['sample_ids']) == 100 and set(row['sample_ids']) == expected_ids
                assert len(row['predictions']) == len(row['labels']) == 100
                assert row['labels'] == [samples[i]['imagenet_index'] for i in row['sample_ids']]
                assert all(isinstance(v, int) and 0 <= v < 1000 for v in row['predictions'])
                correct = sum(a == b for a, b in zip(row['predictions'], row['labels']))
                assert row['correct_count'] == correct
                assert abs(row['quality_score'] - correct / 100) < 1e-12
                if mask == 1:
                    assert row['predictions'] == [0] * 100
                check_timing(row)
            assert block['baseline_quality'] == rows[0]['quality_score'] > 0
            assert rows[1]['quality_score'] / rows[0]['quality_score'] < .99
            assert set(block['quality_sample_ids']) == expected_ids
            split_samples[split] = expected_ids
        assert not split_samples['train'] & split_samples['eval']
        eval_rows = {r['mask']: r for r in task['eval'][0]['coalitions']}
        gate = d['actual_gate_result']
        assert gate['accepted'] is False and gate['accepted_mask'] == 0
        assert gate['requested_mask'] == 1 and gate['rollback_to_baseline'] is True
        assert gate['reason'] == 'quality_below_threshold'
        fault_retention = eval_rows[1]['quality_score'] / eval_rows[0]['quality_score']
        assert abs(gate['normalized_quality'] - fault_retention) < 1e-12
        assert gate['baseline_quality'] == eval_rows[0]['quality_score']
        assert gate['raw_latency_ms'] == eval_rows[1]['mean_ms']
        assert gate['accepted_latency_ms'] == eval_rows[0]['mean_ms']
        rollback = d['executed_rollback']
        assert rollback['executed_after_rejection'] is True
        assert rollback['provenance'] == 'physical_measurement'
        assert rollback['sample_ids'] == eval_rows[0]['sample_ids']
        assert rollback['labels'] == eval_rows[0]['labels']
        assert rollback['predictions'] == eval_rows[0]['predictions']
        assert rollback['predictions_match_initial_baseline'] is True
        correct = sum(a == b for a, b in zip(rollback['predictions'], rollback['labels']))
        assert rollback['correct_count'] == correct
        assert abs(rollback['quality_score'] - correct / 100) < 1e-12
        assert rollback['retention_vs_initial_eval_baseline'] == 1.0
        assert rollback['weights_state_sha256'] == d['weights_state_sha256']
        assert rollback['logits_equivalent_to_initial_baseline'] is True
        assert rollback['logits_allclose_atol'] == rollback['logits_allclose_rtol'] == 1e-4
        assert math.isfinite(rollback['logits_max_abs_error'])
        assert math.isfinite(rollback['logits_relative_l2_error'])
        check_timing(rollback)
        assert d['stress_check_passed'] is True
        method_hash_matches = {
            name: saved == sha(Path(__file__).resolve().parents[1] / 'methods' / name)
            for name, saved in d['gate_source_sha256'].items()
        }
        records.append({
            'path': str(path.resolve()), 'sha256': sha(path),
            'device_id': task['device_id'], 'model_id': task['model_id'],
            'baseline_eval_accuracy': eval_rows[0]['quality_score'],
            'fault_eval_accuracy': eval_rows[1]['quality_score'],
            'fault_eval_retention': fault_retention,
            'actual_gate_rejected': True, 'actual_rollback_reexecuted': True,
            'rollback_prediction_accuracy_restored': True,
            'rollback_logits_allclose_recorded': True,
            'current_method_source_hash_matches': method_hash_matches,
        })
    out = {
        'kind': 'independent_physical_stress_saved_record_audit',
        'record_count': len(records), 'all_arithmetic_identity_checks_passed': True,
        'records': records, 'audit_source_sha256': sha(Path(__file__)),
        'limitations': 'Raw logits are not stored; logit allclose cannot be reconstructed from hashes. Physical authenticity also needs worker execution receipts. Fault injections are not main optimization actions or algorithm gains.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
