"""Post hoc evaluation of already-frozen, physically executed fallback decisions.

This program never fits a model or changes a threshold. A complete independent
physical profile is read only after the active decision and measurement files
have been finalized. Fresh selected-query timings are not their own reference.
"""
import argparse
import collections
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def samples_mean(record):
    samples = record['latency_samples_ms']
    if len(samples) < 20 or not all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in samples):
        raise ValueError('invalid or insufficient physical timing samples')
    value = statistics.mean(samples)
    if not math.isclose(value, record['latency_mean_ms'], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('physical mean does not match raw timing samples')
    return value


def metrics(rows):
    if not rows:
        return {'records': 0, 'micro_mre_percent': None, 'family_macro_mre_percent': None, 'model_macro_mre_percent': None, 'attention_mre_percent': None, 'attention_n': 0}
    groups = {k: collections.defaultdict(list) for k in ('family', 'model_id', 'device_id')}
    errs = []
    for r in rows:
        error = 100 * abs(r['prediction_ms'] - r['reference_ms']) / max(r['reference_ms'], 1e-6)
        errs.append(error)
        for key in groups:
            groups[key][r[key]].append(error)
    means = {k: {g: statistics.mean(v) for g, v in groups[k].items()} for k in groups}
    return {'records': len(rows), 'micro_mre_percent': statistics.mean(errs),
            'family_macro_mre_percent': statistics.mean(means['family'].values()),
            'model_macro_mre_percent': statistics.mean(means['model_id'].values()),
            'per_family_mre_percent': means['family'],
            'per_family_n': {k: len(v) for k, v in groups['family'].items()},
            'per_model_mre_percent': means['model_id'],
            'per_device_mre_percent': means['device_id'],
            'per_device_n': {k: len(v) for k, v in groups['device_id'].items()},
            'attention_mre_percent': means['family'].get('Attention'),
            'attention_n': len(groups['family'].get('Attention', [])),
            'record_ids_sha256': hashlib.sha256('\n'.join(sorted(r['record_id'] for r in rows)).encode()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True, type=Path)
    ap.add_argument('--decisions', required=True, type=Path)
    ap.add_argument('--active', required=True, type=Path)
    ap.add_argument('--reference-profile', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError('do not overwrite post hoc evaluation')
    policy, freeze, active, profile = [json.loads(p.read_text()) for p in (args.policy, args.decisions, args.active, args.reference_profile)]
    if policy['kind'] != 'validation_frozen_active_fallback_policy' or policy['target_latencies_or_test_errors_in_policy']:
        raise ValueError('policy was not target-label-free')
    if freeze['kind'] != 'active_fallback_decisions_frozen_before_selected_query_timing' or freeze['operator_latency_labels_read']:
        raise ValueError('decision artifact does not establish target-label-free freezing')
    if active['kind'] != 'physically_executed_active_fallback' or active['unmeasured_labels_used_or_invented']:
        raise ValueError('not a physical active execution')
    if freeze['policy_sha256'] != sha(args.policy) or active['policy_sha256'] != sha(args.policy):
        raise ValueError('policy hash differs')
    if active['frozen_decisions_sha256'] != sha(args.decisions):
        raise ValueError('active result does not bind frozen decisions')
    if profile['schema_version'] != 'het-physical-profile-v2.1' or not profile['provenance'].get('completed_utc'):
        raise ValueError('incomplete physical reference profile')
    device = profile['hardware']['device_id']
    if freeze['device_id'] != device or active['device_id'] != device:
        raise ValueError('physical runtime-stack identity differs')
    graph_hashes = {g['graph_id']: g['input_weights_sha256'] for g in freeze['graphs']}
    reference = {}
    for graph in profile['graphs']:
        if graph['model_id'] not in policy['test_model_ids'] or graph['shape_id'] not in policy['shape_ids']:
            continue
        if graph_hashes.get(graph['graph_id']) != graph['input_weights_sha256']:
            raise ValueError('active/reference graph input/weight hash differs')
        nodes = {n['node_id']: n for n in graph['nodes']}
        for record in graph['records']:
            if record['status'] != 'measured':
                continue
            key = record['record_id']
            if key in reference:
                raise ValueError('duplicate reference record')
            reference[key] = {'record_id': key, 'reference_ms': samples_mean(record),
                              'family': nodes[record['op_instance_id']]['family'],
                              'node_id': record['op_instance_id'], 'model_id': graph['model_id'],
                              'shape_id': graph['shape_id'], 'device_id': device,
                              'input_weights_sha256': graph['input_weights_sha256']}
    decisions = {r['record_id']: r for r in freeze['decisions']}
    if len(decisions) != len(freeze['decisions']):
        raise ValueError('duplicate frozen decision')
    if set(decisions) != set(reference):
        raise ValueError('decision/reference record support differs; cannot report a complete closed-set global metric')
    threshold = math.inf if policy['threshold_infinite'] else policy['threshold']
    if not policy['threshold_infinite'] and (not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0):
        raise ValueError('invalid frozen threshold')
    for key, row in decisions.items():
        ref = reference[key]
        if not isinstance(row['prediction_ms'], (int, float)) or not math.isfinite(row['prediction_ms']) or row['prediction_ms'] <= 0:
            raise ValueError('invalid frozen prediction')
        if not isinstance(row['uncertainty'], (int, float)) or not math.isfinite(row['uncertainty']) or row['uncertainty'] < 0:
            raise ValueError('invalid frozen uncertainty')
        if row['trigger_measurement'] != (row['uncertainty'] > threshold):
            raise ValueError('frozen decision differs from frozen policy threshold')
        if row['node_id'] != ref['node_id'] or row['model_id'] != ref['model_id'] or row['shape_id'] != ref['shape_id'] or row['input_weights_sha256'] != ref['input_weights_sha256']:
            raise ValueError('frozen query identity differs from physical reference')
    selected = {k for k, r in decisions.items() if r['trigger_measurement']}
    measured, failed, seen_active = {}, [], set()
    for row in active['records']:
        key = row['record_id']
        if key not in selected or key in seen_active:
            raise ValueError('active result contains unselected or duplicate measurement')
        seen_active.add(key)
        if row['policy_sha256'] != sha(args.policy) or row['frozen_decisions_sha256'] != sha(args.decisions) or row['input_weights_sha256'] != reference[key]['input_weights_sha256']:
            raise ValueError('fresh measurement is not bound to the same frozen query')
        if row['status'] == 'measured':
            measured[key] = row
            samples_mean(row)
        else:
            failed.append(key)
    if active['query_count'] != len(decisions) or active['triggered_count'] != len(selected) or active['successful_measurements'] != len(measured):
        raise ValueError('active aggregate counts differ from records')
    pure, selective, replay, fresh, selected_vs_fresh, drift = [], [], [], [], [], []
    for key, row in decisions.items():
        ref = reference[key]
        q = {**ref, 'prediction_ms': row['prediction_ms']}
        pure.append(q)
        replay.append({**q, 'prediction_ms': ref['reference_ms'] if key in selected else row['prediction_ms']})
        # Failed selected calls retain the original predictor output; they are
        # never replaced by a reference value and never silently dropped.
        fresh.append({**q, 'prediction_ms': samples_mean(measured[key]) if key in measured else row['prediction_ms']})
        if key not in selected:
            selective.append(q)
        if key in measured:
            selected_vs_fresh.append({**q, 'reference_ms': samples_mean(measured[key])})
            drift.append({**q, 'prediction_ms': samples_mean(measured[key])})
    out = {'kind': 'post_hoc_evaluation_of_frozen_physical_active_fallback',
           'created_utc': datetime.now(timezone.utc).isoformat(), 'code_sha256': sha(__file__),
           'source_hashes': {k: sha(v) for k, v in [('policy', args.policy), ('decisions', args.decisions), ('active', args.active), ('independent_physical_reference', args.reference_profile)]},
           'device_id': device, 'threshold': policy['threshold'], 'threshold_infinite': policy['threshold_infinite'],
           'query_count': len(decisions), 'selected_count': len(selected),
           'selection_fraction': len(selected) / len(decisions),
           'successful_fresh_measurements': len(measured), 'failed_measurement_record_ids': failed,
           'selected_without_successful_measurement': sorted(selected - measured.keys()),
           'pure_prediction_against_independent_reference': metrics(pure),
           'unmeasured_selective_prediction_against_independent_reference': metrics(selective),
           'oracle_label_substitution_replay_against_reference': metrics(replay),
           'fresh_measurement_hybrid_against_independent_reference': metrics(fresh),
           'selected_predictor_against_fresh_measurement': metrics(selected_vs_fresh),
           'fresh_measurement_drift_against_independent_reference': metrics(drift),
           'wall_cost_seconds': {k: active[k] for k in ['decision_phase_wall_seconds', 'measurement_phase_wall_seconds_including_graph_setup', 'individual_measurement_wall_seconds_including_warmup']},
           'fresh_timed_blocks_cost_ms_excluding_warmup_and_setup': sum(sum(r['latency_samples_ms']) * r['inner_repeats'] for r in measured.values()),
           'physical_reference_read_only_after_active_execution': True,
           'threshold_refitted': False, 'failures': active['failures'],
           'scope': [
               'Post hoc accuracy is computed against a separately measured physical target profile, joined only after frozen active execution.',
               'Oracle replay substitutes existing reference labels and therefore is not a fresh runtime measurement.',
               'Fresh hybrid substitutes only successfully remeasured selected queries. Drift against the reference can be nonzero.',
               'Failed or unmeasured selected queries retain pure predictions; no missing target value is invented.',
               'Wall cost includes predictor shape propagation and frozen inference; measurement-phase wall includes setup and warmup. This is not end-to-end application latency.',
               'No confidence guarantee is inferred from ensemble dispersion or the validation budget under target-stack shift.'
           ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({'output': str(args.output), 'queries': len(decisions), 'selected': len(selected), 'successful_fresh': len(measured), 'pure_mre': out['pure_prediction_against_independent_reference']['micro_mre_percent'], 'fresh_hybrid_mre': out['fresh_measurement_hybrid_against_independent_reference']['micro_mre_percent']}))


if __name__ == '__main__':
    main()
