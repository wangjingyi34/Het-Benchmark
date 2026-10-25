"""Preregistered held-out, adjacent AB/BA confirmation of all <=2 fold actions.

New experiment: does not overwrite fullmodel.run.v2 or reselect a plan. Each
mask has separate A/B model objects and equal call frequency. Mask0 is an
independent baseline-versus-baseline negative control. All raw pairs survive.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn.utils import fuse_conv_bn_eval
from torchvision import models

from full_model_coalitions import (
    ACTIONS, assign, disable_inplace_modules, hardware, sha, state_hash, sync,
)

FROZEN_HELPER_SHA256 = '04c84a5283a3c5c1c286cf3b7dc6300c60d90fbad5275437a1ec2f15907cacdb'
MASKS = [m for m in range(16) if bin(m).count('1') <= 2]


def utc():
    return datetime.now(timezone.utc).isoformat()


def tensor_sha(value):
    return sha(value.detach().cpu().contiguous().numpy().tobytes())


def error_record(error):
    return {'type': type(error).__name__, 'message': str(error),
            'traceback': traceback.format_exc()}


def resource_snapshot(device):
    out = {'utc': utc(), 'pid': os.getpid(), **hardware(device)}
    if hasattr(os, 'sched_getaffinity'):
        out['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    backend = getattr(torch, device.type, None)
    for name in ['memory_allocated', 'memory_reserved', 'max_memory_allocated']:
        fn = getattr(backend, name, None)
        if fn:
            try:
                out[name + '_bytes'] = int(fn())
            except Exception as error:
                out[name + '_error'] = str(error)
    if device.type == 'npu':
        try:
            result = subprocess.run(['npu-smi', 'info'], text=True,
                                    capture_output=True, timeout=15)
            out['npu_smi'] = {'returncode': result.returncode,
                              'stdout': result.stdout, 'stderr': result.stderr}
        except Exception as error:
            out['npu_smi_error'] = str(error)
    return out


def quality(model, images, labels, sample_ids, device, batch=16):
    pieces = []
    for start in range(0, len(sample_ids), batch):
        pieces.append(model(images[start:start + batch].to(device)).cpu())
    logits = torch.cat(pieces)
    prediction = logits.argmax(1)
    correct = (prediction == labels)
    return logits, {
        'quality_evaluated': True, 'quality_score': int(correct.sum()) / len(sample_ids),
        'correct_count': int(correct.sum()), 'sample_count': len(sample_ids),
        'sample_ids': sample_ids, 'labels': labels.tolist(),
        'predictions': prediction.tolist(), 'correct': correct.tolist(),
        'logits_sha256': tensor_sha(logits),
    }


def one_call(model, x, device):
    # Hashes/CPU copies happen only after both adjacent timed calls.
    sync(device)
    before_utc = utc()
    start_ns = time.perf_counter_ns()
    output = model(x)
    sync(device)
    end_ns = time.perf_counter_ns()
    return {'start_monotonic_ns': start_ns, 'end_monotonic_ns': end_ns,
            'start_utc': before_utc, 'end_utc': utc(),
            'latency_ms': (end_ns - start_ns) / 1e6}, output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['resnet18', 'resnet50'], required=True)
    ap.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'npu'], required=True)
    ap.add_argument('--batch', type=int, choices=[1, 4], required=True)
    ap.add_argument('--run', type=int, choices=range(101, 106), required=True)
    ap.add_argument('--assets', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Never overwrite paired confirmation evidence')
    helper_path = Path(__file__).with_name('full_model_coalitions.py')
    if sha(helper_path.read_bytes()) != FROZEN_HELPER_SHA256:
        raise RuntimeError('Frozen 04c helper differs; refuse silent protocol drift')
    if args.device == 'npu':
        import torch_npu
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.device == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('No silent MPS CPU fallback')
    precision = {'dtype': 'float32', 'cuda_matmul_tf32': False, 'cudnn_tf32': False}
    if args.device == 'npu':
        precision.update({'conv_hf32': bool(torch.npu.conv.allow_hf32),
                          'matmul_hf32': bool(torch.npu.matmul.allow_hf32),
                          'torch_npu': torch_npu.__version__})
        if precision['conv_hf32'] or precision['matmul_hf32']:
            raise RuntimeError('Use the fixed FP32 npu-python launcher; HF32 must be disabled')
    manifest_path = args.assets / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert sha((args.assets / 'quality_inputs.npz').read_bytes()) == manifest['quality_asset_sha256']
    inp_meta = manifest['timing_inputs']['evaluation_b' + str(args.batch)]
    xnp = np.load(args.assets / inp_meta['file'], allow_pickle=False)
    assert sha(xnp.tobytes()) == inp_meta['tensor_sha256']
    q = np.load(args.assets / 'quality_inputs.npz', allow_pickle=False)
    ids = [i for i, s in enumerate(manifest['samples']) if s['partition'] == 'evaluation']
    assert len(ids) == 100
    sample_ids = [manifest['samples'][i]['sample_id'] for i in ids]
    images = torch.from_numpy(q['images'][ids].copy())
    labels = torch.from_numpy(q['labels'][ids].copy())
    x = torch.from_numpy(xnp.copy()).to(device)
    assert images.dtype == x.dtype == torch.float32
    seed_text = f'het.paired.confirmation.v1:{args.device}:{args.model}:b{args.batch}:r{args.run}'
    seed = int(sha(seed_text.encode())[:16], 16)
    rng = random.Random(seed)
    pair_orders = {}
    for mask in MASKS:
        orders = ['AB'] * 10 + ['BA'] * 10
        rng.shuffle(orders)
        pair_orders[mask] = orders
    round_masks = []
    for _ in range(20):
        order = list(MASKS)
        rng.shuffle(order)
        round_masks.append(order)
    warmup_order = [(mask, side) for mask in MASKS for side in ['A', 'B']]
    rng.shuffle(warmup_order)
    weights = (models.ResNet18_Weights if args.model == 'resnet18' else models.ResNet50_Weights).IMAGENET1K_V1
    base_cpu = getattr(models, args.model)(weights=weights, progress=True).eval()
    disabled = disable_inplace_modules(base_cpu)
    for parameter in base_cpu.parameters():
        parameter.requires_grad_(False)
    base_hash = state_hash(base_cpu)
    out = {
        'schema_version': 'het.fullmodel.paired-confirmation.v1',
        'status': 'initializing', 'started_utc': utc(), 'run_index': args.run,
        'run_id': f'{args.device}_{args.model}_b{args.batch}_paired_run{args.run}',
        'task_id': f'{args.model}_b{args.batch}_imagenet_val10class_v2',
        'model_id': args.model, 'device_id': args.device + '_' + hardware(device)['hardware'].replace(' ', '_'),
        'provenance': 'physical_measurement', 'purpose': 'heldout confirmation; no calibration plan reselection',
        'source_sha256': sha(Path(__file__).read_bytes()), 'frozen_helper_sha256': FROZEN_HELPER_SHA256,
        'input_shape': list(xnp.shape), 'input_tensor_sha256': inp_meta['tensor_sha256'],
        'dataset_manifest_sha256': sha(manifest_path.read_bytes()), 'dataset_id': manifest['dataset_id'],
        'weights': 'IMAGENET1K_V1', 'weights_url': weights.url, 'weights_state_sha256': base_hash,
        'quality_partition': 'evaluation', 'quality_sample_ids': sample_ids,
        'action_ids': [a[0] for a in ACTIONS], 'masks': MASKS,
        'runtime': {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'python': sys.version,
                    'platform': platform.platform(), 'pid': os.getpid(), 'precision': precision},
        'model_configuration': {'inplace_disabled_modules': disabled},
        'protocol': {'fresh_process_runs': [101, 102, 103, 104, 105], 'warmup_per_model': 30,
                     'paired_repeats_per_mask': 20, 'quality_images_per_model': 100,
                     'quality_mode': 'retention_ratio', 'quality_threshold': .99,
                     'allclose_atol': 1e-4, 'allclose_rtol': 1e-4,
                     'mask0': 'two separate unmodified baseline model objects',
                     'independence_unit': 'fresh process block, not individual timing pair',
                     'timing': 'adjacent per-side synchronized wall clock; hash/copy outside both timed calls',
                     'model_use_frequency': 'separate A and B for every mask; one call per side per round',
                     'reselection': False, 'seed_text': seed_text, 'seed': seed,
                     'pair_orders': pair_orders, 'round_mask_orders': round_masks,
                     'warmup_order': warmup_order},
        'resources_before': resource_snapshot(device), 'coalitions': [],
        'limitations': ['Additional held-out confirmation, not replacement of original measurements.',
                        'Twenty timing pairs are repeated calls within one process, not twenty independent models.',
                        'Fixed 100-image ten-class validation audit does not establish full ImageNet accuracy.',
                        'No method is reselected from these confirmation outcomes.'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temp = args.output.with_suffix(args.output.suffix + '.tmp')
        temp.write_text(json.dumps(out, indent=2))
        temp.replace(args.output)

    save()
    deployed = {}
    rows = {}
    try:
        with torch.inference_mode():
            for mask in MASKS:
                row = {'mask': mask, 'action_ids': [ACTIONS[i][0] for i in range(4) if mask & (1 << i)],
                       'status': 'initializing', 'artifact_id': out['run_id'] + f'_mask{mask}', 'pairs': []}
                rows[mask] = row
                out['coalitions'].append(row)
                try:
                    baseline_cpu = copy.deepcopy(base_cpu)
                    candidate_cpu = copy.deepcopy(base_cpu)
                    for i, (_, cpath, bpath) in enumerate(ACTIONS):
                        if mask & (1 << i):
                            assign(candidate_cpu, cpath, fuse_conv_bn_eval(candidate_cpu.get_submodule(cpath), candidate_cpu.get_submodule(bpath)))
                            assign(candidate_cpu, bpath, nn.Identity())
                    row['baseline_weights_sha256'] = state_hash(baseline_cpu)
                    row['candidate_weights_sha256'] = state_hash(candidate_cpu)
                    deployed[(mask, 'A')] = baseline_cpu.to(device).eval()
                    deployed[(mask, 'B')] = candidate_cpu.to(device).eval()
                    alogits, aq = quality(deployed[(mask, 'A')], images, labels, sample_ids, device)
                    blogits, bq = quality(deployed[(mask, 'B')], images, labels, sample_ids, device)
                    equivalent = bool(torch.allclose(alogits, blogits, atol=1e-4, rtol=1e-4))
                    delta = blogits - alogits
                    retention = bq['quality_score'] / aq['quality_score'] if aq['quality_score'] > 0 else None
                    row.update({'baseline_quality': aq, 'candidate_quality': bq,
                                'retention_ratio': retention, 'quality_threshold_passed': retention is not None and retention >= .99,
                                'equivalence': {'passed': equivalent, 'atol': 1e-4, 'rtol': 1e-4,
                                                'max_abs_error': float(delta.abs().max()),
                                                'relative_l2_error': float(delta.norm() / alogits.norm().clamp_min(1e-12))},
                                'status': 'ok', 'candidate_valid': equivalent and retention is not None and retention >= .99})
                    del alogits, blogits
                except Exception as error:
                    row.update({'status': 'oom' if 'out of memory' in str(error).lower() else 'failed',
                                'error': error_record(error), 'candidate_valid': False})
                save()
            # All model warmups finish before the first timed pair.
            for mask, side in warmup_order:
                if rows[mask]['status'] != 'ok':
                    continue
                try:
                    for _ in range(30):
                        deployed[(mask, side)](x)
                    sync(device)
                except Exception as error:
                    rows[mask].update({'status': 'failed', 'warmup_error': error_record(error)})
            out['warmup_completed_utc'] = utc()
            out['timing_started_utc'] = utc()
            out['status'] = 'measuring'
            save()
            for repeat, order in enumerate(round_masks):
                for position, mask in enumerate(order):
                    row = rows[mask]
                    if row['status'] != 'ok':
                        continue
                    pair = {'pair_index': repeat, 'round_mask_position': position,
                            'order': pair_orders[mask][repeat], 'status': 'running'}
                    row['pairs'].append(pair)
                    outputs = {}
                    try:
                        for side in pair['order']:
                            pair[side], outputs[side] = one_call(deployed[(mask, side)], x, device)
                        # Keep all output hashes and equivalence outside both timed windows.
                        for side in ['A', 'B']:
                            pair[side]['output_sha256'] = tensor_sha(outputs[side])
                            pair[side]['output_shape'] = list(outputs[side].shape)
                        pair['output_equivalent'] = bool(torch.allclose(outputs['A'], outputs['B'], atol=1e-4, rtol=1e-4))
                        a, b = pair['A']['latency_ms'], pair['B']['latency_ms']
                        pair.update({'baseline_ms': a, 'candidate_ms': b, 'paired_delta_ms': a - b,
                                     'paired_gain_percent': 100 * (a - b) / a, 'status': 'ok'})
                    except Exception as error:
                        pair.update({'status': 'oom' if 'out of memory' in str(error).lower() else 'failed',
                                     'error': error_record(error)})
                out['completed_pair_rounds'] = repeat + 1
                save()
            out['timing_completed_utc'] = utc()
        for row in out['coalitions']:
            ok = [p for p in row['pairs'] if p['status'] == 'ok']
            row['completed_pairs'] = len(ok)
            row['failed_pairs'] = len(row['pairs']) - len(ok)
            if ok:
                gains = [p['paired_gain_percent'] for p in ok]
                deltas = [p['paired_delta_ms'] for p in ok]
                row['summary'] = {'mean_paired_gain_percent': statistics.mean(gains),
                                  'mean_paired_delta_ms': statistics.mean(deltas),
                                  'baseline_mean_ms': statistics.mean(p['baseline_ms'] for p in ok),
                                  'candidate_mean_ms': statistics.mean(p['candidate_ms'] for p in ok),
                                  'ratio_of_means_gain_percent': 100 * (statistics.mean(p['baseline_ms'] for p in ok) - statistics.mean(p['candidate_ms'] for p in ok)) / statistics.mean(p['baseline_ms'] for p in ok),
                                  'paired_delta_sample_sd_ms': statistics.stdev(deltas) if len(deltas) > 1 else None,
                                  'negative_gain_pairs': sum(v < 0 for v in gains),
                                  'AB_pairs': sum(p['order'] == 'AB' for p in ok),
                                  'BA_pairs': sum(p['order'] == 'BA' for p in ok),
                                  'all_timed_outputs_equivalent': all(p['output_equivalent'] for p in ok)}
            row['final_candidate_feasible'] = bool(
                row.get('candidate_valid') is True and len(ok) == 20
                and row['failed_pairs'] == 0 and all(p['output_equivalent'] for p in ok))
        out['resources_after'] = resource_snapshot(device)
        out['status'] = 'completed' if all(r['status'] == 'ok' and r.get('completed_pairs') == 20 for r in out['coalitions']) else 'completed_with_failures'
    except BaseException as error:
        out['status'] = 'failed'
        out['fatal_error'] = error_record(error)
        raise
    finally:
        out['completed_utc'] = utc()
        save()
    print(json.dumps({'output': str(args.output), 'status': out['status'], 'run_id': out['run_id'],
                      'masks': len(out['coalitions']), 'pairs': sum(r.get('completed_pairs', 0) for r in out['coalitions'])}), flush=True)


if __name__ == '__main__':
    main()
