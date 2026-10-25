#!/usr/bin/env python3
"""Frozen secondary statistical experiment; standard library + NumPy; local CPU only."""
from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import math
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTOCOL = HERE / 'protocol.json'
ARCHIVE = ROOT / 'v2-实证增强/prediction/formal-archive-restored/formal-five-stack'
PREDICTIONS = ARCHIVE / 'predictions.jsonl.gz'
SUMMARY = ARCHIVE / 'summary.json'
RAW = ROOT / 'v2-实证增强/evidence/physical_profiles'
METHODS = ['rgat', 'drop_contains', 'drop_has_type', 'drop_dependency',
           'drop_similarity', 'drop_supports', 'drop_performance_context']
PANELS = list(itertools.product(['strict', 'probe'],
                              ['model_holdout', 'device_holdout_seen_architectures']))
SEEDS = [11, 22, 33, 44, 55]
REPS = 50000
RNG_SEED = 2026102504


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def ci(x):
    return [float(t) for t in np.quantile(x, [.025, .975])]


def holm(values):
    p = np.asarray(values)
    order = np.argsort(p, kind='stable')
    adjusted = np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)).clip(0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out.tolist()


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    started = datetime.now(timezone.utc).isoformat()
    cfg = json.loads(PROTOCOL.read_text())
    assert cfg['bootstrap']['replicates'] == REPS
    assert cfg['seeds'] == SEEDS
    assert cfg['comparators'] == METHODS[1:]
    source_paths = [PROTOCOL, Path(__file__), PREDICTIONS, SUMMARY,
                    ROOT / 'v2-实证增强/prediction/evaluate_profiles.py',
                    ROOT / 'v4-提分条件补强/original-reviews.txt',
                    ROOT / 'v3-逐问重构/question-crosswalk-v3.json']
    summary = json.loads(SUMMARY.read_text())
    assert summary['code_sha256'] == sha(source_paths[4]), 'Evaluation-code drift'
    truth = {}
    source_verification = []
    raw_statuses = Counter()
    for item in summary['source_manifest']:
        path = RAW / Path(item['path']).name
        sidecar = RAW / Path(item['metadata_sidecar']).name
        assert sha(path) == item['sha256'], 'Raw source hash mismatch'
        assert sha(sidecar) == item['metadata_sidecar_sha256'], 'Sidecar hash mismatch'
        source_paths += [path, sidecar]
        d = json.loads(path.read_text())
        device = d['hardware']['device_id']
        assert device == item['hardware']
        for g in d['graphs']:
            nodes = {n['node_id']: n for n in g['nodes']}
            for rec in g['records']:
                raw_statuses[rec['status']] += 1
                if rec['status'] != 'measured':
                    continue
                t = rec['latency_samples_ms']
                assert len(t) >= 20 and all(math.isfinite(x) and x > 0 for x in t)
                measured_mean = math.fsum(t) / len(t)
                assert abs(measured_mean - rec['latency_mean_ms']) < 1e-9
                rid = rec['record_id']
                assert rid not in truth, 'Duplicate physical record'
                truth[rid] = {'latency': measured_mean, 'model': g['model_id'],
                              'device': device, 'node': rec['op_instance_id'],
                              'family': nodes[rec['op_instance_id']]['family'],
                              'weights': g['input_weights_sha256']}
        source_verification.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                                    'sidecar_sha256': sha(sidecar), 'device': device,
                                    'measured_rows': item['record_status_counts']['measured']})
    assert len(truth) == 13905, 'Unexpected physical support; no silent intersection'
    rids = sorted(truth)
    idx = {r: i for i, r in enumerate(rids)}
    models = sorted({x['model'] for x in truth.values()})
    devices = sorted({x['device'] for x in truth.values()})
    assert len(models) == 12 and len(devices) == 5
    errors = np.full((len(PANELS), len(METHODS), len(SEEDS), len(rids)), np.nan)
    seen = np.zeros(errors.shape, dtype=bool)
    panel_idx = {p: i for i, p in enumerate(PANELS)}
    method_idx = {m: i for i, m in enumerate(METHODS)}
    seed_idx = {s: i for i, s in enumerate(SEEDS)}
    fold_ids = {}
    lines = used = 0
    print('Raw physical support verified; streaming archived predictions.', flush=True)
    with gzip.open(PREDICTIONS, 'rt') as f:
        for line in f:
            lines += 1
            row = json.loads(line)
            if row['method'] not in method_idx:
                continue
            pi = panel_idx[row['feature_mode'], row['protocol']]
            mi = method_idx[row['method']]
            si = seed_idx[row['seed']]
            ri = idx[row['record_id']]
            t = truth[row['record_id']]
            assert row['model_id'] == t['model'] and row['device_id'] == t['device']
            assert row['node_id'] == t['node'] and row['family'] == t['family']
            assert row['input_weights_sha256'] == t['weights']
            assert abs(row['true_latency_ms'] - t['latency']) < 1e-9
            assert math.isfinite(row['prediction_ms']) and row['prediction_ms'] > 0
            key = pi, mi, si, ri
            assert not seen[key], 'Duplicate selected prediction'
            seen[key] = True
            errors[key] = 100 * abs(row['prediction_ms'] - row['true_latency_ms']) / max(row['true_latency_ms'], 1e-6)
            fold_key = (pi, row['model_id'] if row['protocol'] == 'model_holdout' else row['device_id'])
            old = fold_ids.setdefault(fold_key, row['split_id'])
            assert old == row['split_id'], 'Method/seed fold identity mismatch'
            used += 1
            if lines % 500000 == 0:
                print(f'{lines:,} prediction rows read.', flush=True)
    assert seen.all(), f'Missing predictions: {np.size(seen) - seen.sum()}'
    assert np.isfinite(errors).all()
    print(f'Exact matching passed: {used:,} selected of {lines:,} archived rows.', flush=True)
    counts = np.array([sum(truth[r]['model'] == model for r in rids) for model in models])
    per_model = np.empty((4, 7, 5, 12))
    sufficient, sufficient_device = [], []
    time = np.array([truth[r]['latency'] for r in rids])
    floor = np.maximum(time, 1e-6)
    for j, model in enumerate(models):
        mask = np.array([truth[r]['model'] == model for r in rids])
        for pi, (mode, protocol) in enumerate(PANELS):
            for mi, method in enumerate(METHODS):
                for si, seed in enumerate(SEEDS):
                    e = errors[pi, mi, si, mask]
                    per_model[pi, mi, si, j] = e.mean()
                    rec = {'feature_mode': mode, 'protocol': protocol, 'method': method,
                           'seed': seed, 'model_id': model, 'n_records': int(mask.sum()),
                           'sum_absolute_relative_error_percent': float(e.sum()),
                           'sum_squared_relative_error_percent2': float(np.square(e).sum()),
                           'mean_relative_error_percent': float(e.mean()),
                           'sum_absolute_error_ms': float((e * floor[mask] / 100).sum()),
                           'sum_true_latency_ms': float(time[mask].sum())}
                    sufficient.append(rec)
                    for device in devices:
                        dm = mask & np.array([truth[r]['device'] == device for r in rids])
                        de = errors[pi, mi, si, dm]
                        sufficient_device.append({**{k: rec[k] for k in ['feature_mode', 'protocol', 'method', 'seed', 'model_id']},
                                                  'device_id': device, 'n_records': int(dm.sum()),
                                                  'sum_absolute_relative_error_percent': float(de.sum()),
                                                  'mean_relative_error_percent': float(de.mean())})
    write_csv(HERE / 'per-model-seed-sufficient-statistics.csv', sufficient)
    write_csv(HERE / 'per-model-device-seed-sufficient-statistics.csv', sufficient_device)
    rng = np.random.default_rng(RNG_SEED)
    draw_models = rng.integers(0, 12, (REPS, 12))
    draw_seeds = rng.integers(0, 5, (REPS, 5))
    signs = np.array(list(itertools.product([-1, 1], repeat=12)))
    results, per_device = [], []
    for pi, (mode, protocol) in enumerate(PANELS):
        full = per_model[pi, 0].mean(axis=0)
        for mi, method in enumerate(METHODS[1:], 1):
            ablation = per_model[pi, mi].mean(axis=0)
            diff_seed_model = (per_model[pi, mi] - per_model[pi, 0]).T
            diff = diff_seed_model.mean(axis=1)
            boot = diff[draw_models].mean(axis=1)
            crossed = diff_seed_model[draw_models[:, :, None], draw_seeds[:, None, :]].mean(axis=(1, 2))
            micro_boot = (diff[draw_models] * counts[draw_models]).sum(axis=1) / counts[draw_models].sum(axis=1)
            observed = float(diff.mean())
            null = (signs * diff).mean(axis=1)
            p = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
            item = {'feature_mode': mode, 'protocol': protocol, 'ablation': method,
                    'sign': 'ablation minus full RGAT; positive favors full',
                    'n_model_clusters': 12, 'n_training_seeds': 5, 'n_records_per_seed': len(rids),
                    'full_model_macro_mre_percent': float(full.mean()),
                    'ablation_model_macro_mre_percent': float(ablation.mean()),
                    'paired_model_macro_difference_pp': observed,
                    'model_cluster_percentile95_pp': ci(boot),
                    'crossed_model_seed_percentile95_pp': ci(crossed),
                    'full_micro_mre_percent': float((full * counts).sum() / counts.sum()),
                    'ablation_micro_mre_percent': float((ablation * counts).sum() / counts.sum()),
                    'paired_micro_difference_pp': float((diff * counts).sum() / counts.sum()),
                    'micro_model_cluster_percentile95_pp': ci(micro_boot),
                    'models_with_positive_difference': int((diff > 0).sum()),
                    'models_with_negative_difference': int((diff < 0).sum()),
                    'per_model_differences_pp': dict(zip(models, diff.tolist())),
                    'per_seed_macro_differences_pp': dict(zip(map(str, SEEDS), diff_seed_model.mean(axis=0).tolist())),
                    'leave_one_model_out_macro_range_pp': [float(x) for x in [min((diff.sum()-diff)/11), max((diff.sum()-diff)/11)]],
                    'descriptive_signflip_two_sided_p': p,
                    'same_record_support_verified': True}
            results.append(item)
            for device in devices:
                dm = np.array([truth[r]['device'] == device for r in rids])
                vec = []
                for model in models:
                    mm = dm & np.array([truth[r]['model'] == model for r in rids])
                    vec.append(float((errors[pi, mi, :, mm] - errors[pi, 0, :, mm]).mean()))
                per_device.append({'feature_mode': mode, 'protocol': protocol, 'ablation': method,
                                   'device_id': device, 'paired_model_macro_difference_pp': float(np.mean(vec)),
                                   'per_model_differences_pp': dict(zip(models, vec))})
    for pi in range(4):
        group = results[pi * 6:(pi + 1) * 6]
        adjusted = holm([x['descriptive_signflip_two_sided_p'] for x in group])
        for x, a in zip(group, adjusted):
            x['descriptive_holm6_p'] = a
    for x, a in zip(results, holm([x['descriptive_signflip_two_sided_p'] for x in results])):
        x['descriptive_holm24_p'] = a
    # Independently reconcile point estimates with the original summary, not its intervals.
    maximum_difference = 0.0
    for pi, (mode, protocol) in enumerate(PANELS):
        for mi, method in enumerate(METHODS):
            aggregate = next(x for x in summary['aggregate_results']
                             if (x['feature_mode'], x['protocol'], x['method']) == (mode, protocol, method))
            for si, seed in enumerate(SEEDS):
                expected = next(x for x in aggregate['seed_metrics'] if x['seed'] == seed)
                discrepancy = abs(float(per_model[pi, mi, si].mean()) - expected['model_macro_mre_percent'])
                maximum_difference = max(maximum_difference, discrepancy)
                assert discrepancy < 1e-8, 'Raw point estimate disagrees with archived summary'
    output = {'kind': 'new-offline-paired-model-cluster-relation-drop-experiment',
              'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'runtime': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(), 'device': 'local CPU'},
              'protocol_sha256': sha(PROTOCOL), 'analysis_code_sha256': sha(Path(__file__)),
              'support': {'original_prediction_rows_read': lines, 'selected_rows_exactly_matched': used,
                          'physical_records_per_method_seed_panel': len(rids), 'models': models, 'devices': devices,
                          'per_model_record_counts': dict(zip(models, counts.tolist())),
                          'raw_statuses': dict(raw_statuses),
                          'record_ids_sha256': hashlib.sha256('\n'.join(rids).encode()).hexdigest(),
                          'source_verification': source_verification,
                          'maximum_macro_discrepancy_from_original_summary_pp': maximum_difference,
                          'all_checks_passed': True},
              'bootstrap_replicates': REPS, 'rng_seed': RNG_SEED,
              'results': results, 'per_device_descriptive': per_device,
              'interpretation_limits': [
                  'All 24 results are reported; no target-result selection.',
                  'New descriptive secondary analysis of existing V2 fitted predictions, not new hardware measurements or reproduction of original 14.3%.',
                  'Twelve fixed model clusters and shared cross-validation fits limit population inference. Intervals are not confirmatory or simultaneous.',
                  'Sign-flip exchangeability is unproven; Holm-adjusted values remain exploratory descriptive diagnostics.',
                  'Probe includes three measured hardware microprofiles; it is not zero-measurement transfer.',
                  'Device holdout sees model architectures in training and five runtime stacks are not five independent hardware families.',
                  'Relation deletion changes the trained predictor. It is a controlled architecture ablation on matched data, not causal identification of relation semantics in arbitrary graphs.'
              ]}
    write_json(HERE / 'relation-drop-model-cluster-results.json', output)
    flat = [{k: v for k, v in x.items() if not isinstance(v, dict)} for x in results]
    write_csv(HERE / 'relation-drop-model-cluster-results.csv', flat)
    write_json(HERE / 'source-hashes.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
               'files': [{'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in source_paths]})
    print(json.dumps({'all_checks_passed': True, 'results': len(results), 'macro_summary':
                     [{k: x[k] for k in ['feature_mode', 'protocol', 'ablation', 'paired_model_macro_difference_pp',
                                         'model_cluster_percentile95_pp', 'descriptive_holm6_p']} for x in results]}, indent=2))


if __name__ == '__main__':
    main()
