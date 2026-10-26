"""Explicit upgrade adapter for the archived MeasuredTable.final_gate call.

Replace:
    decision = table.final_gate(eval_block, selected_mask)
with:
    decision = upgraded_final_gate(table, eval_block, selected_mask,
                                  raw_path=pinned.path,
                                  expected_source_sha256=pinned.sha256)

`pinned` must be resolved using a trusted run manifest outside candidate input.
This adapter is executable but has NOT been installed in the public runtime.
It returns a baseline fallback on every mismatch; the execution controller must
apply the decision and reload the pinned baseline when required. A returned
decision by itself is not evidence that a real rollback was executed.
"""
from collections.abc import Mapping
import math

from strict_quality_gate import (
    load_verified_run, make_binding, packet_from_run, strict_final_gate,
    baseline_fallback, quality_contract_from_run, SUPPORTED_QUALITY_MODE,
    TABLE_QUALITY_METRIC,
)


def _finite_threshold(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and 0 < value <= 1 and math.isfinite(value))


def upgraded_final_gate(table, block, selected_mask, *, raw_path, expected_source_sha256):
    """Bind original table/block to the exact byte-verified measurement run."""
    # Explicit object/mapping boundary; do not broadly swallow arbitrary errors.
    if table is None or not isinstance(block, Mapping):
        return baseline_fallback('upgrade_table_or_block_invalid', selected_mask)
    try:
        payload, mode, threshold, supported = table.payload, table.mode, table.threshold, table.supported
    except AttributeError:
        return baseline_fallback('upgrade_table_attribute_missing', selected_mask)
    if not isinstance(payload, Mapping) or not isinstance(supported, (list, tuple)):
        return baseline_fallback('upgrade_table_payload_or_support_invalid', selected_mask)
    if not all(type(x) is bool for x in supported):
        return baseline_fallback('upgrade_support_invalid', selected_mask)
    if not all(k in block for k in ['run_id', 'quality_partition', 'provenance', 'quality_sample_ids', 'records']):
        return baseline_fallback('upgrade_block_field_missing', selected_mask)
    if not isinstance(block['records'], Mapping) or not isinstance(block['quality_sample_ids'], list):
        return baseline_fallback('upgrade_block_mapping_invalid', selected_mask)
    definition = payload.get('quality_definition')
    if not isinstance(definition, Mapping) or not all(k in definition for k in ['mode', 'threshold', 'metric', 'dataset_id']):
        return baseline_fallback('table_quality_definition_incomplete', selected_mask)
    if mode != SUPPORTED_QUALITY_MODE or definition['mode'] != SUPPORTED_QUALITY_MODE:
        return baseline_fallback('table_quality_mode_unsupported', selected_mask)
    if not _finite_threshold(threshold) or not _finite_threshold(definition['threshold']):
        return baseline_fallback('table_quality_threshold_invalid', selected_mask)
    if float(threshold) != float(definition['threshold']):
        return baseline_fallback('table_quality_threshold_inconsistent', selected_mask)
    if definition['metric'] != TABLE_QUALITY_METRIC:
        return baseline_fallback('table_quality_metric_unsupported', selected_mask)
    if not isinstance(definition['dataset_id'], str) or not definition['dataset_id'].strip():
        return baseline_fallback('table_quality_dataset_invalid', selected_mask)
    try:
        run = load_verified_run(raw_path, expected_source_sha256)
        raw = run.raw
        contract = quality_contract_from_run(run)
        if any(definition[key] != contract[key] for key in ['mode', 'threshold', 'metric', 'dataset_id']):
            return baseline_fallback('table_raw_quality_contract_mismatch', selected_mask)
        for key in ['task_id', 'device_id', 'model_id', 'input_shape', 'action_ids']:
            if payload.get(key) != raw.get(key):
                return baseline_fallback('table_raw_identity_mismatch', selected_mask)
        if (block.get('run_id') != raw['run_id'] or block.get('quality_partition') != raw['quality_partition']
                or block.get('provenance') != raw['provenance']):
            return baseline_fallback('block_raw_identity_mismatch', selected_mask)
        if type(selected_mask) is not int:
            return baseline_fallback('candidate_mask_invalid')
        binding = make_binding(run, selected_mask, threshold=threshold, max_actions=2)
        if tuple(supported) != binding.hardware_supported:
            return baseline_fallback('table_raw_support_mismatch', selected_mask)
        baseline = packet_from_run(run, 0)
        candidate = packet_from_run(run, selected_mask)
        if block.get('quality_sample_ids') != baseline['record']['sample_ids']:
            return baseline_fallback('block_sample_order_mismatch', selected_mask)
        # Reject mutable table entries even if the immutable raw file is valid.
        if (block['records'][0] != baseline['record']
                or block['records'][selected_mask] != candidate['record']):
            return baseline_fallback('table_record_differs_from_pinned_raw', selected_mask)
        return strict_final_gate(baseline, candidate, binding)
    except (OSError, KeyError, TypeError, ValueError, IndexError, OverflowError):
        return baseline_fallback('upgrade_input_invalid', selected_mask)
