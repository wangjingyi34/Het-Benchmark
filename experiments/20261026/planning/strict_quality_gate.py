"""Fail-closed gate for immutable measured baseline/candidate ledgers.

Trust boundary: expected_source_sha256 must come from an independently pinned
manifest or authenticated measurement collector, never from the candidate.
Hashes bind artifacts; they do not prove that a physical experiment occurred.
No torch, hardware, network, deployment, or mutation of source files is used.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple
import copy
import hashlib
import json
import math
import statistics


IDENTITY_FIELDS = (
    'schema_version', 'provenance', 'task_id', 'device_id', 'model_id', 'run_id',
    'run_index', 'phase', 'quality_partition', 'weights_state_sha256',
    'input_tensor_sha256', 'dataset_manifest_sha256', 'dataset_id', 'input_shape',
    'model_configuration', 'quality_metric', 'quality_sample_count', 'action_ids',
    'protocol',
)
SHA_FIELDS = ('weights_state_sha256', 'input_tensor_sha256', 'dataset_manifest_sha256')
SUPPORTED_QUALITY_MODE = 'retention_ratio'
TABLE_QUALITY_METRIC = 'ImageNet1000_top1_accuracy'
RAW_QUALITY_METRIC = 'unrestricted ImageNet-1000 top-1 accuracy, not ten-class-restricted logits'


class _Reject(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _need(condition, code):
    if not condition:
        raise _Reject(code)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _hash_valid(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _same_number(a, b):
    return _number(a) and _number(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        _need(key not in obj, 'duplicate_json_key')
        obj[key] = value
    return obj


def _bad_constant(_):
    raise _Reject('nonfinite_json_constant')


@dataclass(frozen=True)
class VerifiedRun:
    """Created by load_verified_run; make_binding rechecks in-memory integrity."""
    raw: Mapping[str, Any]
    source_sha256: str
    canonical_payload_sha256: str


@dataclass(frozen=True)
class GateBinding:
    """Frozen identity and ledger anchors, separate from candidate-owned input."""
    source_sha256: str
    identity_json: str
    baseline_record_sha256: str
    candidate_record_sha256: str
    sample_order_sha256: str
    sample_labels_sha256: str
    baseline_artifact_id: str
    candidate_artifact_id: str
    candidate_mask: int
    candidate_actions: Tuple[str, ...]
    hardware_supported: Tuple[bool, ...]
    max_actions: int
    threshold: float
    quality_mode: str
    quality_metric: str
    quality_dataset_id: str

    def to_dict(self):
        return asdict(self)


def load_verified_run(path, expected_source_sha256):
    """Verify bytes before parsing; callers must pin expected hash externally."""
    _need(_hash_valid(expected_source_sha256), 'missing_or_invalid_trusted_source_hash')
    data = Path(path).read_bytes()
    _need(hashlib.sha256(data).hexdigest() == expected_source_sha256, 'source_hash_mismatch')
    raw = json.loads(data, object_pairs_hook=_unique_object, parse_constant=_bad_constant)
    _need(isinstance(raw, dict), 'invalid_run_object')
    return VerifiedRun(raw, expected_source_sha256, _digest(raw))


def _verified(run):
    _need(isinstance(run, VerifiedRun), 'verified_run_required')
    _need(_hash_valid(run.source_sha256), 'invalid_verified_source_hash')
    _need(_digest(run.raw) == run.canonical_payload_sha256, 'verified_run_mutated')


def _quality_contract(raw):
    protocol = raw.get('protocol')
    _need(isinstance(protocol, Mapping), 'raw_quality_protocol_missing')
    _need(protocol.get('quality_mode') == SUPPORTED_QUALITY_MODE, 'raw_quality_mode_unsupported')
    threshold = protocol.get('quality_threshold')
    _need(_number(threshold) and 0 < threshold <= 1, 'raw_quality_threshold_invalid')
    _need(raw.get('quality_metric') == RAW_QUALITY_METRIC, 'raw_quality_metric_unsupported')
    dataset = raw.get('dataset_id')
    _need(isinstance(dataset, str) and bool(dataset.strip()), 'raw_quality_dataset_invalid')
    return {'mode': SUPPORTED_QUALITY_MODE, 'threshold': float(threshold),
            'metric': TABLE_QUALITY_METRIC, 'raw_metric': RAW_QUALITY_METRIC, 'dataset_id': dataset}


def quality_contract_from_run(run):
    """Explicit supported semantic mapping from the pinned raw protocol."""
    _verified(run)
    return _quality_contract(run.raw)


def _identity(raw):
    _need(isinstance(raw, Mapping), 'identity_missing')
    _need(all(k in raw for k in IDENTITY_FIELDS), 'identity_field_missing')
    identity = {k: copy.deepcopy(raw[k]) for k in IDENTITY_FIELDS}
    _need(identity['schema_version'] == 'het.fullmodel.run.v2', 'unsupported_run_schema')
    _need(identity['provenance'] == 'physical_measurement', 'measurement_provenance_invalid')
    _need(identity['phase'] == 'eval' and identity['quality_partition'] == 'evaluation', 'evaluation_partition_required')
    for key in ['task_id', 'device_id', 'model_id', 'run_id', 'dataset_id', 'quality_metric']:
        _need(isinstance(identity[key], str) and bool(identity[key].strip()), 'identity_string_invalid')
    for key in SHA_FIELDS:
        _need(_hash_valid(identity[key]), 'identity_hash_invalid')
    _need(type(identity['run_index']) is int and identity['run_index'] > 0, 'run_index_invalid')
    _need(type(identity['quality_sample_count']) is int and identity['quality_sample_count'] > 0, 'quality_sample_count_invalid')
    shape = identity['input_shape']
    _need(isinstance(shape, list) and shape and all(type(x) is int and x > 0 for x in shape), 'input_shape_invalid')
    actions = identity['action_ids']
    _need(isinstance(actions, list) and actions and all(isinstance(a, str) and a for a in actions), 'action_ids_invalid')
    _need(len(set(actions)) == len(actions), 'duplicate_action_ids')
    _need(isinstance(identity['model_configuration'], dict), 'model_configuration_invalid')
    _quality_contract(identity)
    _json(identity)  # Reject non-finite values in nested configuration metadata.
    return identity


def _records(raw):
    records = raw.get('coalitions')
    _need(isinstance(records, list) and records, 'coalitions_missing')
    rows = {}
    for rec in records:
        _need(isinstance(rec, dict), 'coalition_record_invalid')
        mask = rec.get('mask')
        _need(type(mask) is int and 0 <= mask < (1 << len(raw['action_ids'])), 'coalition_mask_invalid')
        _need(mask not in rows, 'duplicate_coalition_mask')
        rows[mask] = rec
    return rows


def make_binding(run, candidate_mask, *, threshold=.99, max_actions=2):
    """Freeze anchors from an already byte-verified collector artifact.

    This does not accept a scientific claim; strict_final_gate still validates
    the raw baseline/candidate ledgers and the quality/performance thresholds.
    """
    _verified(run)
    raw = run.raw
    identity = _identity(raw)
    actions = identity['action_ids']
    _need(type(candidate_mask) is int and 0 <= candidate_mask < (1 << len(actions)), 'candidate_mask_invalid')
    _need(type(max_actions) is int and 0 <= max_actions <= len(actions), 'action_budget_invalid')
    _need(_number(threshold) and 0 < threshold <= 1, 'threshold_invalid')
    quality_contract = _quality_contract(raw)
    _need(float(threshold) == quality_contract['threshold'], 'threshold_differs_from_pinned_quality_protocol')
    rows = _records(raw)
    _need(0 in rows and candidate_mask in rows, 'baseline_or_candidate_missing')
    baseline, candidate = rows[0], rows[candidate_mask]
    profiles = raw.get('action_profiles')
    _need(isinstance(profiles, list), 'support_evidence_missing')
    support = {}
    for p in profiles:
        _need(isinstance(p, dict) and p.get('id') in actions, 'support_identity_invalid')
        _need(p['id'] not in support and type(p.get('hardware_supported')) is bool, 'support_evidence_invalid')
        support[p['id']] = p['hardware_supported']
    _need(set(support) == set(actions), 'support_evidence_incomplete')
    _need(isinstance(baseline.get('sample_ids'), list) and isinstance(baseline.get('labels'), list), 'baseline_ledger_missing')
    _need(isinstance(baseline.get('artifact_id'), str) and isinstance(candidate.get('artifact_id'), str), 'artifact_id_missing')
    return GateBinding(
        source_sha256=run.source_sha256, identity_json=_json(identity),
        baseline_record_sha256=_digest(baseline), candidate_record_sha256=_digest(candidate),
        sample_order_sha256=_digest(baseline['sample_ids']),
        sample_labels_sha256=_digest(list(zip(baseline['sample_ids'], baseline['labels']))),
        baseline_artifact_id=baseline['artifact_id'], candidate_artifact_id=candidate['artifact_id'],
        candidate_mask=candidate_mask, candidate_actions=tuple(a for i, a in enumerate(actions) if candidate_mask & (1 << i)),
        hardware_supported=tuple(support[a] for a in actions), max_actions=max_actions, threshold=float(threshold),
        quality_mode=quality_contract['mode'], quality_metric=quality_contract['metric'],
        quality_dataset_id=quality_contract['dataset_id'],
    )


def packet_from_run(run, mask):
    """Return a detached packet; later changes are detected by the bound gate."""
    _verified(run)
    identity = _identity(run.raw)
    rows = _records(run.raw)
    _need(type(mask) is int and mask in rows, 'packet_mask_missing')
    return {'identity': identity, 'source_sha256': run.source_sha256, 'record': copy.deepcopy(rows[mask])}


def _ledger(rec, binding, identity):
    _need(rec.get('quality_evaluated') is True, 'quality_not_evaluated')
    fields = ['sample_ids', 'labels', 'predictions', 'correct', 'correct_count', 'sample_count', 'quality_score']
    _need(all(k in rec for k in fields), 'quality_field_missing')
    ids, labels, predictions = (rec[k] for k in ['sample_ids', 'labels', 'predictions'])
    _need(isinstance(ids, list) and ids and all(isinstance(i, str) and i for i in ids), 'sample_ids_invalid')
    _need(len(set(ids)) == len(ids), 'sample_ids_duplicate')
    _need(isinstance(labels, list) and isinstance(predictions, list), 'label_or_prediction_vector_invalid')
    _need(len(ids) == len(labels) == len(predictions) == identity['quality_sample_count'], 'sample_vector_length_mismatch')
    _need(type(rec['sample_count']) is int and rec['sample_count'] == len(ids), 'sample_count_mismatch')
    _need(_digest(ids) == binding.sample_order_sha256, 'sample_identity_or_order_mismatch')
    _need(all(type(x) is int and 0 <= x < 1000 for x in labels + predictions), 'class_label_or_prediction_invalid')
    _need(_digest(list(zip(ids, labels))) == binding.sample_labels_sha256, 'sample_label_binding_mismatch')
    correct = [p == y for p, y in zip(predictions, labels)]
    count = sum(correct)
    _need(isinstance(rec['correct'], list) and all(type(x) is bool for x in rec['correct']) and rec['correct'] == correct, 'correct_vector_mismatch')
    _need(type(rec['correct_count']) is int and rec['correct_count'] == count, 'correct_count_mismatch')
    score = count / len(ids)
    _need(_number(rec['quality_score']) and 0 <= rec['quality_score'] <= 1, 'quality_score_nonfinite_or_invalid')
    _need(_same_number(score, rec['quality_score']), 'quality_score_disagrees_with_raw_ledger')
    return {'correct': count, 'total': len(ids), 'score': score}


def _latency(rec):
    values = rec.get('latency_samples_ms')
    _need(isinstance(values, list) and values and all(_number(x) and x > 0 for x in values), 'latency_samples_invalid')
    mean = statistics.mean(values)
    _need(_number(mean) and mean > 0, 'latency_mean_nonfinite')
    _need(_same_number(rec.get('mean_ms'), mean), 'latency_mean_mismatch')
    return mean


def _packet(packet, binding, baseline):
    _need(isinstance(packet, Mapping), 'packet_invalid')
    _need(packet.get('source_sha256') == binding.source_sha256, 'packet_source_hash_mismatch')
    identity = _identity(packet.get('identity'))
    _need(_json(identity) == binding.identity_json, 'model_input_run_or_dataset_identity_mismatch')
    rec = packet.get('record')
    _need(isinstance(rec, Mapping), 'record_missing')
    _need(rec.get('provenance') == 'physical_measurement' and rec.get('quality_partition') == 'evaluation', 'record_provenance_or_partition_mismatch')
    expected_mask = 0 if baseline else binding.candidate_mask
    expected_artifact = binding.baseline_artifact_id if baseline else binding.candidate_artifact_id
    expected_actions = [] if baseline else list(binding.candidate_actions)
    _need(type(rec.get('mask')) is int and rec['mask'] == expected_mask, 'record_mask_binding_mismatch')
    _need(rec.get('artifact_id') == expected_artifact and rec.get('artifact_id') == f"{identity['run_id']}_mask{expected_mask}", 'artifact_id_binding_mismatch')
    _need(rec.get('action_ids') == expected_actions, 'record_action_binding_mismatch')
    _need(rec.get('status') == 'ok', 'measurement_status_invalid')
    _need(isinstance(rec.get('equivalence'), Mapping) and rec['equivalence'].get('passed') is True, 'numerical_equivalence_not_passed')
    measured_ms = _latency(rec)
    ledger = _ledger(rec, binding, identity)
    expected_digest = binding.baseline_record_sha256 if baseline else binding.candidate_record_sha256
    _need(_digest(rec) == expected_digest, 'record_changed_after_collector_freeze')
    return rec, ledger, measured_ms, identity


def _rejection(reason, *, mask=0, baseline_ms=None, baseline_quality=None, run_id=None, artifact_id=None):
    return {'accepted': False, 'reason': reason, 'run_id': run_id, 'requested_mask': mask,
            'accepted_mask': 0, 'rollback_to_baseline': True, 'fallback': 'baseline',
            'requires_verified_baseline_reload': baseline_ms is None,
            'accepted_latency_ms': baseline_ms, 'accepted_benefit_percent': 0.0,
            'baseline_quality': baseline_quality, 'artifact_id': artifact_id,
            'raw_latency_ms': None, 'raw_benefit_percent': None, 'normalized_quality': None,
            'baseline_correct': None, 'candidate_correct': None, 'quality_total': None,
            'quality_mode': 'not_evaluated', 'quality_contract_evaluated': False,
            'fallback_action_required': True, 'rollback_executed': False,
            'decision_implementation': 'strict_quality_gate.v1.1',
            'deployment_or_rollback_execution_proved': False}


def baseline_fallback(reason, requested_mask=0):
    """Public safe rejection for an integration failure before validation."""
    return _rejection(reason, mask=requested_mask if type(requested_mask) is int else 0)


def strict_final_gate(baseline_packet, candidate_packet, binding):
    """Always return a decision. Any invalid evidence falls back to baseline.

    If baseline evidence itself is invalid, accepted_latency_ms is None and
    requires_verified_baseline_reload is True: the caller must reload the
    pinned baseline, not execute the supplied untrusted packet as fallback.
    """
    result = _rejection('invalid_binding')
    try:
        _need(isinstance(binding, GateBinding), 'invalid_binding')
        _need(_hash_valid(binding.source_sha256), 'invalid_binding_hash')
        _need(type(binding.candidate_mask) is int and binding.candidate_mask >= 0, 'invalid_binding_mask')
        _need(type(binding.max_actions) is int and binding.max_actions >= 0, 'invalid_binding_budget')
        _need(_number(binding.threshold) and 0 < binding.threshold <= 1, 'invalid_binding_threshold')
        _need(binding.quality_mode == SUPPORTED_QUALITY_MODE, 'binding_quality_mode_unsupported')
        _need(binding.quality_metric == TABLE_QUALITY_METRIC, 'binding_quality_metric_unsupported')
        result['requested_mask'] = binding.candidate_mask
        base, bq, baseline_ms, identity = _packet(baseline_packet, binding, True)
        contract = _quality_contract(identity)
        _need(binding.threshold == contract['threshold'] and binding.quality_dataset_id == contract['dataset_id'],
              'binding_quality_contract_mismatch')
        result.update(run_id=identity['run_id'], baseline_quality=bq['score'],
                      baseline_correct=bq['correct'], quality_total=bq['total'],
                      accepted_latency_ms=baseline_ms, requires_verified_baseline_reload=False,
                      threshold=binding.threshold, source_sha256=binding.source_sha256,
                      quality_mode=binding.quality_mode, quality_metric=binding.quality_metric,
                      quality_dataset_id=binding.quality_dataset_id, quality_contract_evaluated=True)
        _need(bq['score'] > 0, 'baseline_quality_not_positive')
        _need(len(binding.hardware_supported) == len(identity['action_ids']) and all(type(x) is bool for x in binding.hardware_supported), 'invalid_binding_support')
        _need(binding.candidate_mask < (1 << len(identity['action_ids'])), 'invalid_binding_mask')
        _need(bin(binding.candidate_mask).count('1') <= binding.max_actions, 'action_budget_exceeded')
        _need(all(not binding.candidate_mask & (1 << i) or supported for i, supported in enumerate(binding.hardware_supported)), 'hardware_unsupported')
        candidate, cq, candidate_ms, _ = _packet(candidate_packet, binding, False)
        retention = cq['score'] / bq['score']
        raw_gain = 100 * (1 - candidate_ms / baseline_ms)
        _need(_number(raw_gain), 'nonfinite_derived_metric')
        result.update(raw_latency_ms=candidate_ms, raw_benefit_percent=raw_gain,
                      normalized_quality=retention, candidate_correct=cq['correct'],
                      artifact_id=candidate['artifact_id'], candidate_quality=cq['score'])
        _need(_same_number(candidate.get('retention_ratio'), retention), 'stored_retention_mismatch')
        _need(type(candidate.get('quality_threshold_passed')) is bool and candidate['quality_threshold_passed'] == (retention >= binding.threshold), 'stored_quality_threshold_flag_mismatch')
        _need(retention >= binding.threshold, 'quality_below_threshold')
        _need(candidate_ms <= baseline_ms, 'latency_regression')
        result.update(accepted=True, reason='ok', accepted_mask=binding.candidate_mask,
                      accepted_latency_ms=candidate_ms, accepted_benefit_percent=raw_gain,
                      rollback_to_baseline=False, fallback='none', fallback_action_required=False)
    except _Reject as exc:
        result['reason'] = exc.code
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        result['reason'] = 'malformed_or_nonfinite_evidence'
    return result


def gate_verified_raw(path, expected_source_sha256, candidate_mask, *, threshold=.99, max_actions=2):
    """Simple final_gate upgrade: pinned artifact -> strict decision, fail closed."""
    try:
        run = load_verified_run(path, expected_source_sha256)
        binding = make_binding(run, candidate_mask, threshold=threshold, max_actions=max_actions)
        return strict_final_gate(packet_from_run(run, 0), packet_from_run(run, candidate_mask), binding)
    except _Reject as exc:
        return _rejection(exc.code, mask=candidate_mask if type(candidate_mask) is int else 0)
    except (OSError, KeyError, TypeError, ValueError, IndexError, OverflowError):
        return _rejection('artifact_unreadable_or_malformed', mask=candidate_mask if type(candidate_mask) is int else 0)
