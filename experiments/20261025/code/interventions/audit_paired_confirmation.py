"""Lightweight independent identity, arithmetic and ordering audit of paired JSON.

All completed records, negative effects and failures are retained. This audit
does not authenticate a device from its name, or reconstruct logits from hashes.
"""
import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

MASKS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12]
ACCEPTED_SOURCE_SHAS = {
    '2f33287e7d6afffb9303724ac61186c5712cbaaa0784795196d7b1f4ec6c22bb',
    'e88bce1b159f018e5dfa5793c726d362b70bcf0ed9795bf2a881d0ac5c1408c5',
}
HELPER_SHA = '04c84a5283a3c5c1c286cf3b7dc6300c60d90fbad5275437a1ec2f15907cacdb'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-9 * max(1, abs(a), abs(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', type=Path, required=True)
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    manifest_path = args.assets / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    samples = {s['sample_id']: s for s in manifest['samples'] if s['partition'] == 'evaluation'}
    assert len(samples) == 100
    records = []
    groups = defaultdict(list)
    count = {'completed_blocks': 0, 'incomplete_or_failed_blocks': 0, 'mask_records': 0,
             'successful_pairs': 0, 'failed_pairs': 0, 'timed_calls': 0,
             'quality_predictions': 0, 'final_infeasible_masks': 0}
    for path in sorted(args.raw_dir.rglob('*.json')):
        d = json.loads(path.read_text())
        if d.get('schema_version') != 'het.fullmodel.paired-confirmation.v1':
            continue
        assert d['source_sha256'] in ACCEPTED_SOURCE_SHAS and d['frozen_helper_sha256'] == HELPER_SHA
        assert d['dataset_manifest_sha256'] == sha(manifest_path)
        assert d['quality_partition'] == 'evaluation' and set(d['quality_sample_ids']) == set(samples)
        assert d['run_index'] in range(101, 106) and d['masks'] == MASKS
        batch = d['input_shape'][0]
        assert d['input_shape'] == [batch, 3, 224, 224] and batch in [1, 4]
        assert d['input_tensor_sha256'] == manifest['timing_inputs']['evaluation_b' + str(batch)]['tensor_sha256']
        protocol = d['protocol']
        assert protocol['reselection'] is False
        assert protocol['warmup_per_model'] == 30 and protocol['paired_repeats_per_mask'] == 20
        assert protocol['quality_threshold'] == .99
        assert protocol['allclose_atol'] == protocol['allclose_rtol'] == 1e-4
        assert all(sorted(order) == MASKS for order in protocol['round_mask_orders'])
        for mask in MASKS:
            order = protocol['pair_orders'][str(mask)]
            assert len(order) == 20 and order.count('AB') == order.count('BA') == 10
        if d['device_id'].startswith('npu_'):
            assert d['runtime']['precision']['conv_hf32'] is False
            assert d['runtime']['precision']['matmul_hf32'] is False
        if d['status'] != 'completed':
            count['incomplete_or_failed_blocks'] += 1
        else:
            count['completed_blocks'] += 1
            assert len(d['coalitions']) == 11
            assert d['started_utc'] <= d['warmup_completed_utc'] <= d['timing_started_utc'] <= d['timing_completed_utc'] <= d['completed_utc']
        rows = {r['mask']: r for r in d['coalitions']}
        assert len(rows) == len(d['coalitions']) and set(rows) <= set(MASKS)
        chronological_calls = []
        summaries = []
        for mask, row in rows.items():
            count['mask_records'] += 1
            for name in ['baseline_quality', 'candidate_quality']:
                if name not in row:
                    assert row['status'] != 'ok'
                    continue
                q = row[name]
                assert q['quality_evaluated'] is True and len(q['sample_ids']) == 100
                assert set(q['sample_ids']) == set(samples)
                assert q['labels'] == [samples[i]['imagenet_index'] for i in q['sample_ids']]
                assert len(q['predictions']) == 100
                assert all(isinstance(p, int) and 0 <= p < 1000 for p in q['predictions'])
                correct = [a == b for a, b in zip(q['predictions'], q['labels'])]
                assert correct == q['correct'] and sum(correct) == q['correct_count']
                assert close(sum(correct) / 100, q['quality_score'])
                count['quality_predictions'] += 100
            if 'baseline_quality' in row and 'candidate_quality' in row:
                baseline = row['baseline_quality']['quality_score']
                retention = row['candidate_quality']['quality_score'] / baseline if baseline > 0 else None
                assert retention == row['retention_ratio']
                assert row['quality_threshold_passed'] == (retention is not None and retention >= .99)
                assert row['candidate_valid'] == (row['equivalence']['passed'] and row['quality_threshold_passed'])
                assert row['baseline_weights_sha256'] == d['weights_state_sha256']
                if mask == 0:
                    assert row['candidate_weights_sha256'] == d['weights_state_sha256']
            ok = []
            repeats = []
            for pair in row['pairs']:
                repeat = pair['pair_index']
                repeats.append(repeat)
                assert 0 <= repeat < 20
                assert pair['order'] == protocol['pair_orders'][str(mask)][repeat]
                assert protocol['round_mask_orders'][repeat][pair['round_mask_position']] == mask
                if pair['status'] != 'ok':
                    count['failed_pairs'] += 1
                    continue
                ok.append(pair)
                count['successful_pairs'] += 1
                for side in ['A', 'B']:
                    t = pair[side]
                    assert t['start_monotonic_ns'] < t['end_monotonic_ns']
                    assert t['start_utc'] <= t['end_utc']
                    assert close((t['end_monotonic_ns'] - t['start_monotonic_ns']) / 1e6, t['latency_ms'])
                    assert len(t['output_sha256']) == 64 and t['output_shape'] == [batch, 1000]
                    chronological_calls.append((t['start_monotonic_ns'], t['end_monotonic_ns']))
                    count['timed_calls'] += 1
                first, second = pair['order']
                assert pair[first]['end_monotonic_ns'] <= pair[second]['start_monotonic_ns']
                a, b = pair['A']['latency_ms'], pair['B']['latency_ms']
                assert close(pair['baseline_ms'], a) and close(pair['candidate_ms'], b)
                assert close(pair['paired_delta_ms'], a-b)
                assert close(pair['paired_gain_percent'], 100 * (a-b) / a)
            assert len(repeats) == len(set(repeats))
            if 'completed_pairs' in row:
                assert row['completed_pairs'] == len(ok)
                assert row['failed_pairs'] == len(row['pairs']) - len(ok)
                feasible = row.get('candidate_valid') is True and len(ok) == 20 and row['failed_pairs'] == 0 and all(p['output_equivalent'] for p in ok)
                assert row['final_candidate_feasible'] == feasible
                if not feasible:
                    count['final_infeasible_masks'] += 1
            if ok and 'summary' in row:
                summary = row['summary']
                a = statistics.mean(p['baseline_ms'] for p in ok)
                b = statistics.mean(p['candidate_ms'] for p in ok)
                assert close(summary['baseline_mean_ms'], a) and close(summary['candidate_mean_ms'], b)
                assert close(summary['mean_paired_gain_percent'], statistics.mean(p['paired_gain_percent'] for p in ok))
                assert close(summary['mean_paired_delta_ms'], statistics.mean(p['paired_delta_ms'] for p in ok))
                assert close(summary['ratio_of_means_gain_percent'], 100 * (a-b) / a)
                assert summary['AB_pairs'] == sum(p['order'] == 'AB' for p in ok)
                assert summary['BA_pairs'] == sum(p['order'] == 'BA' for p in ok)
                assert summary['negative_gain_pairs'] == sum(p['paired_gain_percent'] < 0 for p in ok)
                if d['status'] == 'completed':
                    assert len(ok) == 20 and summary['AB_pairs'] == summary['BA_pairs'] == 10
                summaries.append({'mask': mask, 'final_candidate_feasible': row['final_candidate_feasible'], **summary})
        chronological_calls.sort()
        assert all(a[1] <= b[0] for a, b in zip(chronological_calls, chronological_calls[1:]))
        record = {'file': str(path.resolve()), 'sha256': sha(path), 'run_id': d['run_id'],
                  'run_index': d['run_index'], 'device_id': d['device_id'], 'model_id': d['model_id'],
                  'measurement_source_sha256': d['source_sha256'],
                  'batch': batch, 'pid': d['runtime']['pid'], 'status': d['status'],
                  'mask_summaries': summaries, 'paired_temporal_order_checked': True,
                  'arithmetic_and_identity_checked': True}
        records.append(record)
        groups[(d['device_id'], d['model_id'], batch)].append(record)
    if not records:
        raise ValueError('No paired confirmation raw records found')
    contexts = []
    for (device, model, batch), rs in groups.items():
        assert len({r['run_index'] for r in rs}) == len(rs)
        assert len({r['pid'] for r in rs}) == len(rs)
        indices = sorted(r['run_index'] for r in rs)
        contexts.append({'device_id': device, 'model_id': model, 'batch': batch,
                         'run_indices': indices, 'fresh_process_pid_unique': True,
                         'complete_five_process_protocol': indices == list(range(101, 106)) and all(r['status'] == 'completed' for r in rs)})
    result = {'kind': 'independent_paired_confirmation_saved_record_audit', 'counts': count,
              'source_variants': '2f first run saved all data before terminal print failure; e88 changes only flush placement in final print. Measurement, quality and randomization unchanged; original first run is retained.',
              'contexts': contexts, 'records': records, 'audit_source_sha256': sha(Path(__file__)),
              'limitations': 'Arithmetic, IDs and recorded time ordering checked. Timing pairs remain repeated calls within a process. Hashes do not reconstruct logits or authenticate the physical device; worker receipts are also required. Original main matrix is not replaced, and no plans are reselected.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({'output': str(args.output), 'counts': count, 'contexts': contexts}, indent=2))


if __name__ == '__main__':
    main()
