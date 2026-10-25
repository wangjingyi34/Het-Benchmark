"""Synthetic arithmetic/provenance unit tests; zero physical observations."""
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import sys
import tempfile
from pathlib import Path


path = Path(__file__).with_name('summarize_active_fallback.py')
spec = importlib.util.spec_from_file_location('active_postprocess', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_case(root, failed=False, mismatch=False, invalid=None):
    policy = {'kind': 'validation_frozen_active_fallback_policy',
              'target_latencies_or_test_errors_in_policy': False,
              'test_model_ids': ['unit_model'], 'shape_ids': [0],
              'threshold': .5, 'threshold_infinite': False}
    if invalid == 'threshold':
        policy['threshold'] = float('nan')
    pp = root / 'policy.json';pp.write_text(json.dumps(policy))
    decisions = []
    for i, (p, u) in enumerate([(2., .7), (1., .1)]):
        decisions.append({'record_id': f'unit/unit_model:s0/n{i}', 'node_id': f'unit_model:s0/n{i}',
                          'model_id': 'unit_model', 'shape_id': 0,
                          'input_weights_sha256': 'unit-hash', 'prediction_ms': p,
                          'uncertainty': u, 'trigger_measurement': u > .5})
    if invalid == 'prediction':
        decisions[0]['prediction_ms'] = float('nan')
    if invalid == 'uncertainty':
        decisions[0]['uncertainty'] = float('inf')
    freeze = {'kind': 'active_fallback_decisions_frozen_before_selected_query_timing',
              'operator_latency_labels_read': False, 'policy_sha256': sha(pp),
              'device_id': 'unit', 'graphs': [{'graph_id': 'unit_model:s0', 'input_weights_sha256': 'unit-hash'}],
              'decisions': decisions}
    dp = root / 'decisions.json';dp.write_text(json.dumps(freeze))
    row = {**decisions[0], 'policy_sha256': sha(pp), 'frozen_decisions_sha256': sha(dp),
           'status': 'failed' if failed else 'measured',
           'latency_samples_ms': [.9] * 20, 'latency_mean_ms': .9, 'inner_repeats': 8}
    active = {'kind': 'physically_executed_active_fallback', 'unmeasured_labels_used_or_invented': False,
              'policy_sha256': sha(pp), 'frozen_decisions_sha256': sha(dp), 'device_id': 'unit',
              'records': [row], 'query_count': 2, 'triggered_count': 1,
              'successful_measurements': 0 if failed else 1, 'failures': [],
              'decision_phase_wall_seconds': 1., 'measurement_phase_wall_seconds_including_graph_setup': 2.,
              'individual_measurement_wall_seconds_including_warmup': .5}
    if invalid == 'duplicate_failed':
        active['records'].append(dict(row))
    ap = root / 'active.json';ap.write_text(json.dumps(active))
    records = [{'record_id': d['record_id'], 'op_instance_id': d['node_id'], 'status': 'measured',
                'latency_samples_ms': [t] * 20, 'latency_mean_ms': t} for d, t in zip(decisions, (1., 2.))]
    if mismatch:
        records.pop()
    profile = {'schema_version': 'het-physical-profile-v2.1', 'provenance': {'completed_utc': 'unit'},
               'hardware': {'device_id': 'unit'},
               'graphs': [{'graph_id': 'unit_model:s0', 'model_id': 'unit_model', 'shape_id': 0,
                           'input_weights_sha256': 'unit-hash', 'records': records,
                           'nodes': [{'node_id': d['node_id'], 'family': 'Unit'} for d in decisions]}]}
    rp = root / 'reference.json';rp.write_text(json.dumps(profile))
    out = root / 'output.json'
    args = ['unit', '--policy', str(pp), '--decisions', str(dp), '--active', str(ap),
            '--reference-profile', str(rp), '--output', str(out)]
    with contextlib.redirect_stdout(io.StringIO()):
        old = sys.argv
        try:
            sys.argv = args
            module.main()
        finally:
            sys.argv = old
    return json.loads(out.read_text())


def main():
    checks = {}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name in ('normal', 'failed', 'mismatch'):
            (root / name).mkdir()
        result = test_case(root / 'normal')
        checks['pure_mre_arithmetic'] = math.isclose(result['pure_prediction_against_independent_reference']['micro_mre_percent'], 75)
        checks['fresh_hybrid_does_not_zero_measurement_noise'] = math.isclose(result['fresh_measurement_hybrid_against_independent_reference']['micro_mre_percent'], 30)
        checks['oracle_replay_separate'] = math.isclose(result['oracle_label_substitution_replay_against_reference']['micro_mre_percent'], 25)
        checks['unmeasured_selective_mre'] = math.isclose(result['unmeasured_selective_prediction_against_independent_reference']['micro_mre_percent'], 50)
        checks['timed_blocks_cost'] = math.isclose(result['fresh_timed_blocks_cost_ms_excluding_warmup_and_setup'], 144)
        failed = test_case(root / 'failed', failed=True)
        checks['failed_measurement_retains_prediction'] = math.isclose(failed['fresh_measurement_hybrid_against_independent_reference']['micro_mre_percent'], 75)
        try:
            test_case(root / 'mismatch', mismatch=True)
        except ValueError as exc:
            checks['record_support_mismatch_rejected'] = 'support differs' in str(exc)
        else:
            checks['record_support_mismatch_rejected'] = False
        for invalid, expected in [('prediction', 'invalid frozen prediction'), ('uncertainty', 'invalid frozen uncertainty'), ('threshold', 'invalid frozen threshold'), ('duplicate_failed', 'duplicate measurement')]:
            folder = root / invalid;folder.mkdir()
            try:
                test_case(folder, failed=invalid == 'duplicate_failed', invalid=invalid)
            except ValueError as exc:
                checks[invalid + '_rejected'] = expected in str(exc)
            else:
                checks[invalid + '_rejected'] = False
    output = {'kind': 'synthetic_logic_unit_tests_only', 'physical_observations': 0,
              'checks': checks, 'passed': all(checks.values()), 'test_code_sha256': sha(Path(__file__)),
              'postprocessor_code_sha256': sha(path)}
    Path(__file__).with_name('active-postprocess-unit-test-results.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    if not output['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
