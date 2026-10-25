"""Independent, fixed Attention shape/module OOD challenge; never training data.

The measured target is a complete torch.nn.MultiheadAttention leaf (QKV/output
projections plus attention), not the primitive SDPA target in the main catalog.
No original model name, task accuracy or production-LLM equivalence is claimed.
"""
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.fx import GraphModule

import collect_profiles as core
from structural_attributes import attributes_for_node

# Fixed before challenge labels are collected; no configuration is chosen by
# prediction error. Tuple order: batch, sequence length, embedding width, heads.
CONFIGS = [(1,128,128,4), (2,128,128,8), (1,512,128,4), (2,1024,128,8),
           (1,128,512,8), (2,512,512,8), (1,1024,512,8), (2,1024,512,8),
           (1,128,768,12), (2,512,768,12), (1,1024,768,12), (2,1024,768,12)]
MODEL_ID = 'attention_mha_ood_ref'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()


class MHAReference(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(width, heads, dropout=0., bias=True, batch_first=True)

    def forward(self, x):
        return self.mha(x, x, x, need_weights=False)


def build_case(index):
    batch, seq, width, heads = CONFIGS[index]
    if width % heads:raise ValueError('invalid fixed MHA configuration')
    model = MHAReference(width, heads).eval()
    core.portable_init(model, 681000 + index)
    rng = np.random.Generator(np.random.PCG64(781000 + index))
    x = core.exact_uniform((batch, seq, width), rng)
    meta = {'batch': batch, 'sequence_length': seq, 'embedding_width': width, 'heads': heads,
            'head_dim': width // heads, 'weight_seed': 681000 + index, 'input_seed': 781000 + index,
            'dropout': 0., 'bias': True, 'batch_first': True, 'need_weights': False,
            'causal': False, 'mask': None, 'dtype': 'float32',
            'initialization': 'frozen core PCG64 integer stream, binary-exact FP32 conversion and fan-scaled parameter initialization'}
    return model, [x], meta


def set_runtime(device_string, threads):
    if device_string == 'npu':import torch_npu  # noqa: register before torch.device
    device = torch.device(device_string)
    torch.set_num_threads(threads);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not hasattr(torch.backends, 'mha') or not hasattr(torch.backends.mha, 'set_fastpath_enabled'):
        raise RuntimeError('cannot explicitly fix MHA native fastpath across runtimes')
    torch.backends.mha.set_fastpath_enabled(False)
    if torch.backends.mha.get_fastpath_enabled():raise RuntimeError('MHA native fastpath was not disabled')
    return device


def construct_graph(index, device):
    model, inputs, config = build_case(index)
    ih = core.tensor_hash(model, inputs)
    gm = GraphModule(model, core.SafeTracer().trace(model)).eval().to(device)
    inputs = [x.to(device) for x in inputs]
    cap = core.Capture(gm);cap.run(*inputs);core.sync(device)
    gid = f'{MODEL_ID}:c{index:02d}'
    graph = {'graph_id': gid, 'model_id': MODEL_ID, 'shape_id': index, 'workload': 'AttentionChallenge',
             'architecture_source': 'torch.nn.MultiheadAttention leaf; complete MHA module, not primitive SDPA',
             'dtype': 'float32', 'configuration': config, 'input_shapes': core.shapeinfo(inputs),
             'input_weights_sha256': ih, 'nodes': [], 'edges': [], 'records': []}
    nodes = list(gm.graph.nodes)
    for i, n in enumerate(nodes):
        info = cap.info[n.name]
        family = core.family_for(gm, n) if n.name in cap.calls else 'Context'
        node = {'node_id': f'{gid}/{n.name}', 'fx_op': n.op, 'target': str(n.target),
                'module_path': str(n.target) if n.op == 'call_module' else '',
                'family': family, **info, 'dependency_ids': [f'{gid}/{p.name}' for p in n.all_input_nodes],
                'features': core.features(gm, n, info, i, len(nodes))}
        node['operator_attributes'] = attributes_for_node(gm, node, MODEL_ID)
        if n.op == 'call_module' and isinstance(gm.get_submodule(n.target), nn.MultiheadAttention):
            node['operator_attributes'].update({'attention_heads': config['heads'],
                'attention_head_dim': config['head_dim'], 'embedding_dim': config['embedding_width'],
                'in_features': config['embedding_width'], 'out_features': config['embedding_width'],
                'dropout_p': 0., 'is_causal': 0, 'bias_present': 1,
                'module_configuration': config, 'composite_leaf': 1})
        graph['nodes'].append(node)
        for pred in n.all_input_nodes:
            graph['edges'].append({'source': f'{gid}/{pred.name}', 'target': node['node_id'], 'relation': 'dependency'})
    attention = [n for n in graph['nodes'] if n['family'] == 'Attention']
    if len(attention) != 1 or len(cap.calls) != 1:
        raise ValueError('challenge must retain exactly one executable MHA Attention leaf')
    return graph, gm, cap


def write_snapshot(path, data):
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def freeze(args):
    if args.output.exists():raise ValueError('do not overwrite a predictor freeze')
    summary = json.loads(args.summary.read_text())
    rows = [r for r in summary['results'] if r['split_id'] == args.split_id and r['method'] in args.methods and r['feature_mode'] in args.feature_modes]
    if {(r['method'], r['feature_mode']) for r in rows} != {(m, f) for m in args.methods for f in args.feature_modes}:
        raise ValueError('not all requested completed method/feature rows exist yet')
    entries = []
    for row in rows:
        if MODEL_ID in row['train_model_ids'] + row['validation_model_ids']:
            raise ValueError('challenge model appeared in fitting/validation')
        for fit in row['seed_fits']:
            path = args.checkpoint_dir / Path(fit['checkpoint']).name
            if sha(path) != fit['checkpoint_sha256']:raise ValueError('checkpoint hash mismatch')
            entries.append({'method': row['method'], 'feature_mode': row['feature_mode'],
                            'seed': fit['seed'], 'path': str(path.resolve()), 'sha256': fit['checkpoint_sha256'],
                            'split_id': row['split_id'], 'training_model_ids': row['train_model_ids'],
                            'validation_model_ids': row['validation_model_ids'], 'test_model_ids': row['test_model_ids']})
    out = {'kind': 'attention_challenge_frozen_predictors', 'frozen_utc': utc(),
           'source_summary_sha256': sha(args.summary), 'evaluation_code_sha256': summary['code_sha256'],
           'challenge_code_sha256': sha(__file__), 'fixed_configurations': CONFIGS,
           'source_manifest': summary['source_manifest'], 'checkpoints': entries,
           'challenge_labels_used_for_fitting_or_selection': False}
    args.output.parent.mkdir(parents=True, exist_ok=True);args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({'freeze': str(args.output), 'checkpoints': len(entries)}))


def collect(args):
    if args.output.exists():raise ValueError('do not overwrite a challenge measurement')
    predictor_freeze = json.loads(args.predictor_freeze.read_text())
    if predictor_freeze['kind'] != 'attention_challenge_frozen_predictors' or predictor_freeze['challenge_labels_used_for_fitting_or_selection']:
        raise ValueError('invalid predictor freeze')
    if predictor_freeze['fixed_configurations'] != [list(x) for x in CONFIGS] or predictor_freeze['challenge_code_sha256'] != sha(__file__):
        raise ValueError('challenge implementation/configurations changed after freezing')
    source = json.loads(args.hardware_from.read_text())
    h = source['hardware']
    device = set_runtime(args.device, h['threads'])
    from active_fallback import validate_runtime
    observed = validate_runtime(h, device)
    out = {'schema_version': 'het-attention-shape-ood-v1', 'kind': 'independent_physical_attention_challenge',
           'provenance': {'created_utc': utc(), 'code_sha256': sha(__file__),
               'frozen_collector_core_sha256': sha(Path(core.__file__)),
               'structural_attributes_sha256': sha(Path(__file__).with_name('structural_attributes.py')),
               'prediction_freeze_sha256': sha(args.predictor_freeze),
               'predictors_frozen_utc': predictor_freeze['frozen_utc'],
               'hardware_source_sha256': sha(args.hardware_from), 'python': sys.version,
               'torch': torch.__version__, 'platform': platform.platform(),
               'observed_runtime': observed, 'mha_native_fastpath_enabled': torch.backends.mha.get_fastpath_enabled(),
               'warmup': 5, 'repeats': 10, 'inner_repeats': 4,
               'timing_protocol': 'FP32 complete MultiheadAttention leaf; ten synchronized four-call block means after five warmups',
               'challenge_labels_added_to_main_training_or_validation': False},
           'hardware': h, 'fixed_configurations': CONFIGS, 'graphs': [], 'failures': [],
           'scope': 'Independent module/shape OOD challenge. Complete MHA includes QKV/output projections; main Attention labels are functional SDPA primitives. No pretrained model or task-accuracy claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_snapshot(args.output, out)
    with torch.inference_mode():
        for index, config in enumerate(CONFIGS):
            started = time.monotonic()
            try:
                graph, gm, cap = construct_graph(index, device)
                name, (node, nargs, nkwargs) = next(iter(cap.calls.items()))
                record = {'record_id': h['device_id'] + '/' + graph['graph_id'] + '/' + name,
                          'op_instance_id': graph['graph_id'] + '/' + name, 'status': 'measured',
                          'warmup': 5, 'inner_repeats': 4, 'timing_units': 'ms per complete MHA call; synchronized four-call block mean'}
                try:
                    record.update(core.timed(core.callable_for(gm, node, nargs, nkwargs), device, 5, 10, 4))
                except Exception as exc:
                    record['status'] = 'failed';record['error'] = type(exc).__name__ + ': ' + str(exc)[:1000]
                graph['records'].append(record)
                graph['collection_wall_seconds'] = time.monotonic() - started
                out['graphs'].append(graph)
                print(json.dumps({'case': index, 'configuration': config, 'status': record['status'], 'seconds': graph['collection_wall_seconds']}), flush=True)
                del graph, gm, cap
            except Exception as exc:
                failure = {'case_index': index, 'configuration': config, 'stage': 'construction_or_shape_capture', 'error': type(exc).__name__ + ': ' + str(exc)[:1000]}
                out['failures'].append(failure);print(json.dumps(failure), flush=True)
            if device.type == 'cuda':torch.cuda.empty_cache()
            elif device.type == 'mps':torch.mps.empty_cache()
            elif device.type == 'npu':torch.npu.empty_cache()
            write_snapshot(args.output, out)
    out['provenance']['completed_utc'] = utc();write_snapshot(args.output, out)
    print(json.dumps({'complete': True, 'graphs': len(out['graphs']), 'construction_failures': len(out['failures'])}), flush=True)


def predict_challenge(args):
    from evaluate_profiles import build_graph, collate, predict
    from active_fallback import load_fit
    if args.output.exists():raise ValueError('do not overwrite challenge predictions')
    d = json.loads(args.profile.read_text());frozen = json.loads(args.predictor_freeze.read_text())
    if d['schema_version'] != 'het-attention-shape-ood-v1' or not d['provenance'].get('completed_utc'):
        raise ValueError('incomplete independent challenge profile')
    if d['provenance']['prediction_freeze_sha256'] != sha(args.predictor_freeze):raise ValueError('predictor freeze differs from collection binding')
    if sha(Path(__file__).with_name('evaluate_profiles.py')) != frozen['evaluation_code_sha256']:
        raise ValueError('frozen evaluator source changed')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    rows, failures = [], []
    for graph in d['graphs']:
        for ck in frozen['checkpoints']:
            try:
                # Reject accidentally introduced challenge supervision before
                # constructing the same frozen structural feature graph.
                inference_graph = {**graph, 'records': []}
                built = build_graph(inference_graph, d['hardware'], ck['feature_mode'], 10, unlabelled=True)
                batch = collate([built], 'cpu')
                if not torch.isnan(batch['y']).all():raise ValueError('challenge labels entered inference graph')
                path = args.checkpoint_dir / Path(ck['path']).name
                policy = {'feature_mode': ck['feature_mode'], 'method': ck['method'],
                          'train_model_ids': ck['training_model_ids'], 'validation_model_ids': ck['validation_model_ids']}
                fit = load_fit(path, ck['sha256'], batch['x'].shape[1], policy)
                values = predict(fit, batch)
                if len(values) != 1 or not np.isfinite(values).all() or not (values > 0).all():raise ValueError('invalid challenge prediction')
                rows.append({'case_index': graph['shape_id'], 'configuration': graph['configuration'],
                             'model_id': MODEL_ID, 'family': 'Attention', 'device_id': d['hardware']['device_id'],
                             'method': ck['method'], 'feature_mode': ck['feature_mode'], 'seed': ck['seed'],
                             'split_id': ck['split_id'], 'checkpoint_sha256': ck['sha256'],
                             'input_weights_sha256': graph['input_weights_sha256'],
                             'prediction_ms': float(values[0]), 'inference_latency_labels_used': False})
            except Exception as exc:
                failures.append({'case_index': graph['shape_id'], 'method': ck['method'], 'feature_mode': ck['feature_mode'], 'seed': ck['seed'], 'error': type(exc).__name__ + ': ' + str(exc)[:1000]})
    # Prediction rows are already frozen above. Only this separate post hoc
    # stage reads actual challenge measurements and calculates absolute errors.
    truth = {g['shape_id']: r for g in d['graphs'] for r in g['records'] if r['status'] == 'measured'}
    for row in rows:
        target = truth.get(row['case_index'])
        if target is not None:
            ts = target['latency_samples_ms']
            if len(ts) != 10 or not all(math.isfinite(x) and x > 0 for x in ts):raise ValueError('invalid challenge timing labels')
            value = float(np.mean(ts))
            if not math.isclose(value, target['latency_mean_ms'], rel_tol=1e-10, abs_tol=1e-12):raise ValueError('challenge mean mismatch')
            row.update({'physical_latency_ms': value, 'relative_error_percent': abs(row['prediction_ms'] - value) / max(value, 1e-6) * 100})
    out = {'kind': 'frozen_predictor_independent_attention_challenge_evaluation', 'created_utc': utc(),
           'code_sha256': sha(__file__), 'profile_sha256': sha(args.profile), 'prediction_freeze_sha256': sha(args.predictor_freeze),
           'rows': rows, 'failures': failures, 'physical_collection_failures': d['failures'],
           'source_checkpoint_count': len(frozen['checkpoints']), 'challenge_labels_used_for_training_validation_or_tuning': False,
           'scope': d['scope'], 'metrics_scope': 'Per-case/seed errors for complete MHA module OOD. Do not merge with main SDPA primitive table or claim12 independent architectures.'}
    args.output.parent.mkdir(parents=True, exist_ok=True);args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps({'output': str(args.output), 'predictions': len(rows), 'failures': len(failures)}))


def validate_interface(args):
    from evaluate_profiles import build_graph, collate
    device = set_runtime('cpu', 1)
    with torch.inference_mode():
        # Smallest fixed case, one untimed forward. No warmup or timing labels.
        graph, gm, cap = construct_graph(0, device)
    h = {'device_id': 'interface-only', 'backend': 'cpu', 'descriptor_strict':
         {k: 0 for k in ['backend_cpu','backend_cuda','backend_mps','backend_npu','host_arm64','host_logical_cpus','threads','device_memory_bytes','compute_major','compute_minor']}}
    built = build_graph(graph, h, 'strict', 10, unlabelled=True);batch = collate([built], 'cpu')
    out = {'kind': 'untimed_attention_challenge_interface_check', 'physical_timing_observations': 0,
           'configuration_count': len(CONFIGS), 'all_heads_legal': all(w % h == 0 for _, _, w, h in CONFIGS),
           'attention_queries_in_smallest_graph': len(batch['query']), 'feature_dim': batch['x'].shape[1],
           'target_labels_all_nan': bool(torch.isnan(batch['y']).all()),
           'fx_nodes': [(n.op, str(n.target)) for n in gm.graph.nodes],
           'input_weights_sha256': graph['input_weights_sha256'], 'code_sha256': sha(__file__)}
    args.output.write_text(json.dumps(out, indent=2));print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser();sub = ap.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze');f.add_argument('--summary', required=True, type=Path)
    f.add_argument('--checkpoint-dir', required=True, type=Path);f.add_argument('--output', required=True, type=Path)
    f.add_argument('--split-id', default='model-fold0');f.add_argument('--methods', nargs='+', default=['mlp','gcn','rgat'])
    f.add_argument('--feature-modes', nargs='+', default=['strict','probe'])
    c = sub.add_parser('collect');c.add_argument('--device', choices=['cpu','mps','cuda','npu'], required=True)
    c.add_argument('--hardware-from', required=True, type=Path);c.add_argument('--predictor-freeze', required=True, type=Path)
    c.add_argument('--output', required=True, type=Path)
    p = sub.add_parser('predict');p.add_argument('--profile', required=True, type=Path)
    p.add_argument('--predictor-freeze', required=True, type=Path);p.add_argument('--checkpoint-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path);p.add_argument('--threads', type=int, default=4)
    v = sub.add_parser('validate');v.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    if args.command == 'freeze':freeze(args)
    elif args.command == 'collect':collect(args)
    elif args.command == 'predict':predict_challenge(args)
    else:validate_interface(args)


if __name__ == '__main__':main()
