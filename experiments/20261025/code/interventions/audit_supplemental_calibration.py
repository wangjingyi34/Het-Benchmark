"""Independent JSON-only audit of family14 calibration and 60 MHA challenges.

Preserves the previous NPU closeout and creates a separate increment. No timing,
model inference or retraining is performed by this audit.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

COLLECTOR = 'cdd236c31f3c223889e68026f400aad5a6413ec6c1173603d92fb05a4d91dce2'
MANIFEST = 'ea400f2b2df47abbd712dee71322a038f7282831d369b422ee8f016bff6c3c94'
CORE = '7050d276d82bead7f6b936be35f93059be5d24b0073eb7045434174bec618f26'
ATTENTION = 'dfab91261cc4e9a83f232ca9275c20509d5360f78423c071e7a80236f88b5b1c'
FREEZE = '1ade61fff0b8e6ddca4b40f4778d9882b2d86b8ef3c467af4963b5e6713398ea'
STACKS = ['m5cpu', 'm5mps', 'a100mig', 'cloudcpu', 'npub3']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timing(row, count):
    values = row['latency_samples_ms']
    assert len(values) == count and all(math.isfinite(v) and v > 0 for v in values)
    mean, median, sd = statistics.mean(values), statistics.median(values), statistics.stdev(values)
    assert abs(mean - row['latency_mean_ms']) < 1e-9
    assert abs(median - row['latency_median_ms']) < 1e-9
    assert abs(sd - row['latency_sd_ms']) < 1e-9
    return {'mean_ms': mean, 'median_ms': median, 'sample_sd_ms': sd,
            'sample_variance_ms2': statistics.variance(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v2-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    root = a.v2_root
    manifest_path = root / 'methods/family-calibration-manifest.json'
    freeze_path = root / 'prediction/attention-freeze-main-strict.json'
    assert sha(manifest_path) == MANIFEST and sha(freeze_path) == FREEZE
    assert sha(root / 'methods/family_calibration_probes.py') == COLLECTOR
    assert sha(root / 'prediction/collect_profiles.py') == CORE
    assert sha(root / 'prediction/attention_ood_challenge.py') == ATTENTION
    manifest, frozen = json.loads(manifest_path.read_text()), json.loads(freeze_path.read_text())
    assert manifest['timing_observations_at_freeze'] == 0
    assert manifest['total_query_type_budget'] == 17
    assert frozen['challenge_labels_used_for_fitting_or_selection'] is False
    specifications = {p['probe_id']: p for p in manifest['probes']}
    assert len(specifications) == 14
    family_rows, attention_rows, source_records = [], [], []
    attention_identity, hardware_identity = {}, {}
    npu_wall = None
    for stack in STACKS:
        path = root / f'evidence/family14-{stack}.json'
        d = json.loads(path.read_text())
        assert d['schema_version'] == 'het.family-calibration-probes.v1'
        assert d['collector_sha256'] == COLLECTOR and d['manifest_sha256'] == MANIFEST
        assert d['frozen_core_sha256'] == CORE
        assert d['model_latency_or_task_quality_labels_used'] is False
        assert d['successful_probes'] == len(d['records']) == 14 and not d['failures']
        assert (d['existing_probe_count'], d['new_probe_count'], d['total_query_type_budget']) == (3, 14, 17)
        assert (d['warmup'], d['repeats'], d['inner_repeats']) == (10, 20, 8)
        assert manifest['frozen_utc'] < d['started_utc'] < d['completed_utc']
        source_name = 'xeon_cpu' if stack == 'cloudcpu' else stack
        source = root / f'evidence/physical_profiles/{source_name}.json'
        assert sha(source) == d['hardware_source_profile_sha256']
        h = d['hardware']
        original_hardware = json.loads(source.read_text())['hardware']
        for key in ['device_id', 'backend', 'hardware_name', 'threads', 'visible_device_memory_bytes', 'probes']:
            assert h[key] == original_hardware[key]
        assert len(h['probes']) == 3
        hardware_identity[stack] = {k: h[k] for k in ['device_id', 'backend', 'hardware_name', 'visible_device_memory_bytes', 'threads']}
        seen = set()
        for row in d['records']:
            probe = row['probe_id']
            assert probe not in seen and probe in specifications
            seen.add(probe)
            assert row['status'] == 'measured' and row['output_all_finite'] is True
            assert row['untimed_output_check_calls_after_timing'] == 1
            assert len(row['output_sha256']) == 64
            for key, value in specifications[probe].items():
                assert row[key] == value, (stack, probe, key)
            assert row['family'] == specifications[probe]['specification']['family']
            family_rows.append({'stack': stack, 'probe': probe, 'input_weights_sha256': row['input_weights_sha256'],
                                'input_sha256': row['input_sha256'], 'weights_sha256': row['weights_sha256'],
                                'output_all_finite_recorded': True, **timing(row, 20)})
        assert seen == set(specifications)
        if stack == 'npub3':
            assert d['observed_runtime']['hardware_name'] == 'Ascend910B3'
            assert d['observed_runtime']['hf32_flags'] == {'conv': False, 'matmul': False}
            npu_wall = d['wall_seconds_including_runtime_validation']
            assert abs(npu_wall - 11.23394765) < 1e-7
        source_records.append({'kind': 'family14', 'stack': stack, 'sha256': sha(path)})
        path = root / ('evidence/npu-final-backup/attention-npub3.json' if stack == 'npub3'
                       else f'evidence/attention-{stack}.json')
        d = json.loads(path.read_text())
        p = d['provenance']
        assert p['code_sha256'] == ATTENTION and p['frozen_collector_core_sha256'] == CORE
        assert p['prediction_freeze_sha256'] == FREEZE
        assert p['predictors_frozen_utc'] == frozen['frozen_utc'] < p['created_utc'] < p['completed_utc']
        assert p['challenge_labels_added_to_main_training_or_validation'] is False
        assert p['mha_native_fastpath_enabled'] is False
        assert (p['warmup'], p['repeats'], p['inner_repeats']) == (5, 10, 4)
        assert d['fixed_configurations'] == frozen['fixed_configurations']
        assert len(d['graphs']) == 12 and not d['failures']
        for key, value in hardware_identity[stack].items():
            assert d['hardware'][key] == value
        assert p['hardware_source_sha256'] == sha(source)
        if stack == 'npub3':
            assert p['observed_runtime']['hf32_flags'] == {'conv': False, 'matmul': False}
        configs = set()
        for graph in d['graphs']:
            case = graph['shape_id']
            assert case not in configs
            configs.add(case)
            identity = (graph['configuration'], graph['input_weights_sha256'], graph['input_shapes'])
            if case in attention_identity:
                assert attention_identity[case] == identity
            attention_identity[case] = identity
            assert len(graph['input_weights_sha256']) == 64 and len(graph['records']) == 1
            row = graph['records'][0]
            assert row['status'] == 'measured'
            attention_rows.append({'stack': stack, 'case': case,
                                   'input_weights_sha256': graph['input_weights_sha256'], **timing(row, 10)})
        assert configs == set(range(12))
        source_records.append({'kind': 'attention', 'stack': stack, 'sha256': sha(path)})
    assert len(family_rows) == 70 and len(attention_rows) == 60
    out = {'kind': 'supplemental_calibration_closeout_increment_independent_audit',
           'all_checks_passed': True, 'previous_closeout_not_overwritten': True,
           'family14': {'stacks': 5, 'successful_probe_records': 70, 'raw_timing_block_means': 1400,
                        'timed_inner_calls': 11200, 'query_types_per_stack': 17,
                        'input_weight_configuration_identical_to_frozen_manifest_across_stacks': True,
                        'all_output_finite_flags_true': True, 'NPU_wall_seconds': npu_wall},
           'attention': {'stacks': 5, 'successful_cases': 60, 'raw_timing_block_means': 600,
                         'timed_inner_calls': 2400, 'twelve_case_input_weight_configuration_identical_across_stacks': True,
                         'collector_sha256': ATTENTION, 'predictor_freeze_sha256': FREEZE},
           'hardware_identity': hardware_identity, 'source_records': source_records,
           'family_probe_statistics': family_rows, 'attention_statistics': attention_rows,
           'additional_NPU_physical_data_required': False, 'audit_source_sha256': sha(Path(__file__)),
           'limitations': 'JSON/hash/record arithmetic audit. Family finite outputs were checked during collection; raw output values are not reconstructed from hashes. Attention timing vectors are finite, but collector does not save a finite-output assertion. Query-budget and k/tau/gamma/fallback consumer audit is separate. Shutdown still requires the user-requested verified GitHub main publication.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k not in ['family_probe_statistics', 'attention_statistics', 'source_records']}, indent=2))


if __name__ == '__main__':
    main()
