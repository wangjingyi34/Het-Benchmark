"""Offline X8/A7 reanalysis; only writes beside this script. Python 3.9+.

Default is stdlib-only. --reload-policies additionally verifies all 160 saved
policies with the original CPU implementation (torch 2.9.1 in frozen runtime).
No hardware execution, training, outcome selection, or original-file mutation.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics as st
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
METHODS = ROOT / 'v2-实证增强/methods'
RAW = ROOT / 'v2-实证增强/evidence/fullmodel_raw'
RESULTS = METHODS / 'planner-matrix-results/all-complete-v1'
T95 = 2.7764451051977987
SEEDS = [11, 22, 33, 44, 55]
CONFIGS = ['ppo_hardware_mask', 'ppo_without_hardware_mask',
           'ppo_with_heldout_embedding', 'ppo_embedding_zeroed_same_capacity']
MASKS = [m for m in range(16) if bin(m).count('1') <= 2]
CONTROLS = ['same_evidence_exhaustive_search', 'same_evidence_greedy', 'latency',
            'uniform_at_most_2', 'uniform_exact_2']


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csvout(name, rows):
    assert rows
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in r.items()})


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (a, b)


def summary(values):
    values = list(values)
    assert len(values) == 5 and all(math.isfinite(x) for x in values)
    mean = st.mean(values)
    sd = st.stdev(values)
    half = T95 * sd / math.sqrt(5)
    return {'physical_processes': 5, 'values': values, 'mean': mean, 'sample_sd': sd,
            'descriptive_t95_df4': [mean - half, mean + half]}


def quality(rec):
    if rec.get('quality_evaluated') is not True:
        raise ValueError('quality not evaluated')
    ids, labels, predictions = (rec.get(k) for k in ['sample_ids', 'labels', 'predictions'])
    if not ids or not labels or not predictions or len(set(ids)) != len(ids):
        raise ValueError('missing or duplicate sample IDs')
    if not len(ids) == len(labels) == len(predictions):
        raise ValueError('quality vector length mismatch')
    if any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < 1000
           for x in labels + predictions):
        raise ValueError('invalid prediction or label')
    correct = [a == b for a, b in zip(labels, predictions)]
    q = sum(correct) / len(ids)
    if abs(q - rec.get('quality_score', float('nan'))) > 1e-12 or not math.isfinite(rec.get('quality_score', float('nan'))):
        raise ValueError('raw prediction accuracy disagrees with supplied score')
    if rec.get('correct_count') != sum(correct):
        raise ValueError('raw correct count mismatch')
    if 'correct' in rec and rec['correct'] != correct:
        raise ValueError('raw correctness vector mismatch')
    return q, dict(zip(ids, zip(labels, predictions)))


def latency(rec):
    samples = rec['latency_samples_ms']
    assert len(samples) == 30 and all(math.isfinite(v) and v > 0 for v in samples)
    value = st.mean(samples)
    close(value, rec['mean_ms'])
    if 'sample_sd_ms' in rec:
        close(st.stdev(samples), rec['sample_sd_ms'])
    return value


def strict_gate(raw, mask, supported):
    """Audit-only hardened gate: verifies raw ledger before the historical rule.

    This is a validation implementation, not a patch to the archived planner.
    """
    rows = {r['mask']: r for r in raw['coalitions']}
    base, r = rows[0], rows[mask]
    b = latency(base)
    if any(mask & (1 << i) and not yes for i, yes in enumerate(supported)):
        return {'accepted': False, 'reason': 'hardware_unsupported', 'accepted_gain_percent': 0.0}
    if r['status'] != 'ok':
        return {'accepted': False, 'reason': r['status'], 'accepted_gain_percent': 0.0}
    q0, map0 = quality(base)
    q, qmap = quality(r)
    if q0 <= 0 or set(map0) != set(qmap) or any(map0[i][0] != qmap[i][0] for i in map0):
        raise ValueError('quality baseline/sample/label mismatch')
    value = latency(r)
    gain = 100 * (b - value) / b
    retention = q / q0
    reason = 'quality_below_threshold' if retention < .99 else ('latency_regression' if value > b else 'ok')
    return {'accepted': reason == 'ok', 'reason': reason, 'baseline_ms': b, 'candidate_ms': value,
            'raw_gain_percent': gain, 'accepted_gain_percent': gain if reason == 'ok' else 0.0,
            'baseline_top1': q0, 'candidate_top1': q, 'quality_retention': retention,
            'quality_n': len(map0), 'accepted_mask': mask if reason == 'ok' else 0}


def validate_task(path):
    task = read(path)
    assert task['schema_version'] == 'het.measured.coalitions.v2'
    assert len(task['action_ids']) == 4
    supported = [m['hardware_supported'] is True for m in task['action_metadata']]
    blocks, refs = {'train': [], 'eval': []}, []
    source_hashes = {Path(r['path']).name: r['sha256'] for r in task['source_artifacts']}
    for phase in ['train', 'eval']:
        for block in task[phase]:
            p = RAW / Path(block['artifact_path']).name
            assert sha(p) == block['artifact_sha256'] == source_hashes[p.name]
            raw = read(p)
            assert raw['schema_version'] == 'het.fullmodel.run.v2'
            assert raw['provenance'] == 'physical_measurement' and raw['phase'] == phase
            for key in ['task_id', 'device_id', 'model_id', 'input_shape']:
                assert raw[key] == task[key], key
            for key in ['run_id', 'weights_state_sha256', 'input_tensor_sha256', 'baseline_quality', 'coalitions']:
                assert raw[key] == block[key], key
            assert {r['mask'] for r in raw['coalitions']} == set(range(16))
            maps = []
            for rec in raw['coalitions']:
                assert rec['status'] == 'ok' and rec['equivalence']['passed'] is True
                latency(rec)
                q, qmap = quality(rec)
                assert len(qmap) == 100
                maps.append(qmap)
            assert all(set(m) == set(maps[0]) for m in maps)
            assert all(all(m[s][0] == maps[0][s][0] for s in m) for m in maps)
            assert set(block['quality_sample_ids']) == set(maps[0])
            blocks[phase].append(raw)
            refs.append({'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'phase': phase,
                         'run_id': raw['run_id'], 'run_index': raw['run_index'], 'pid': raw['runtime']['pid']})
    assert [r['run_index'] for r in blocks['train']] == [1, 2, 3]
    assert [r['run_index'] for r in blocks['eval']] == [4, 5, 6, 7, 8]
    allraw = blocks['train'] + blocks['eval']
    assert len({r['run_id'] for r in allraw}) == len({r['runtime']['pid'] for r in allraw}) == 8
    for k in ['weights_state_sha256', 'dataset_manifest_sha256']:
        assert len({r[k] for r in allraw}) == 1
    train_ids = set(blocks['train'][0]['coalitions'][0]['sample_ids'])
    eval_ids = set(blocks['eval'][0]['coalitions'][0]['sample_ids'])
    assert train_ids.isdisjoint(eval_ids)
    return task, blocks, refs, supported


def freeze_check():
    paths = {}
    for name in ['analysis-freeze.json', 'input-freeze-supplement.json']:
        for r in read(HERE / name)['inputs']:
            assert sha(ROOT / r['path']) == r['sha256'], r['path']
            paths[r['path']] = r['sha256']
    return paths


def independent_search(task, blocks, greedy=False):
    def obs(m):
        rows = [{r['mask']: r for r in b['coalitions']} for b in blocks['train']]
        return st.mean(latency(r[m]) for r in rows), st.mean(quality(r[m])[0] / quality(r[0])[0] for r in rows)
    safe = [m['hardware_supported'] is True for m in task['action_metadata']]
    eligible = [m for m in MASKS if all(not m & (1 << i) or safe[i] for i in range(4)) and obs(m)[1] >= .99]
    if not greedy:
        return min((obs(m)[0], m) for m in eligible)[1]
    mask = 0
    for _ in range(2):
        candidates = [(obs(mask | (1 << i))[0], mask | (1 << i)) for i in range(4)
                      if not mask & (1 << i) and mask | (1 << i) in eligible]
        if not candidates:
            break
        value, nxt = min(candidates)
        if value >= obs(mask)[0]:
            break
        mask = nxt
    return mask


def panel(reload_policies):
    contexts, seed_rows, seed_stats, flat, contrasts, fits, mask_rows = [], [], [], [], [], [], []
    if reload_policies:
        sys.path.insert(0, str(ROOT / '.experiment-deps'))
        sys.path.insert(0, str(METHODS))
        import torch
        from measured_planner import MeasuredTable, CalibrationEnv, ActorCritic, select_plan
        torch.set_num_threads(1)
    paths = sorted((METHODS / 'planner-matrix-inputs/tasks').glob('*.json'))
    assert len(paths) == 8
    for tp in paths:
        task, blocks, refs, safe = validate_task(tp)
        rp = RESULTS / 'results' / (tp.stem + '.json')
        result = read(rp)
        assert result['status'] == 'completed' and result['input_sha256'] == sha(tp)
        assert result['training_runs'] == [r['run_id'] for r in blocks['train']]
        assert result['evaluation_runs'] == [r['run_id'] for r in blocks['eval']]
        assert result['budget'] == 2 and len(result['ppo']) == 20
        assert result['quality_sample_disjointness_checked'] is True
        cid = tp.stem
        outcomes = [{m: strict_gate(b, m, safe) for m in range(16)} for b in blocks['eval']]
        methods, variant_masks = {}, {}

        def add(name, masks, seed=None):
            raws = [st.mean(o[m]['raw_gain_percent'] for m in masks) for o in outcomes]
            gated = [st.mean(o[m]['accepted_gain_percent'] for m in masks) for o in outcomes]
            p = {'masks': masks, 'raw_gain_percent': summary(raws), 'accepted_gain_percent': summary(gated),
                 'acceptance_fraction': st.mean(o[m]['accepted'] for o in outcomes for m in masks)}
            if seed is None:
                methods[name] = p
                flat.append({'context': cid, 'device': task['device_id'], 'method': name,
                             'masks': masks, 'seed_count': 5 if name in CONFIGS else '',
                             'raw_mean_pct': p['raw_gain_percent']['mean'],
                             'raw_ci95_lo': p['raw_gain_percent']['descriptive_t95_df4'][0],
                             'raw_ci95_hi': p['raw_gain_percent']['descriptive_t95_df4'][1],
                             'accepted_mean_pct': p['accepted_gain_percent']['mean'],
                             'accepted_ci95_lo': p['accepted_gain_percent']['descriptive_t95_df4'][0],
                             'accepted_ci95_hi': p['accepted_gain_percent']['descriptive_t95_df4'][1],
                             'raw_by_process': raws, 'accepted_by_process': gated,
                             'physical_n': 5, 'acceptance_fraction': p['acceptance_fraction'],
                             'task_sha256': sha(tp), 'result_sha256': sha(rp)})
            return p

        for name, r in result['fixed_strategies'].items():
            if r['status'] != 'evaluated':
                methods[name] = {'status': r['status']}
                continue
            mask = r['mask']
            assert mask in MASKS
            if name == 'same_evidence_exhaustive_search':
                assert independent_search(task, blocks) == mask
            if name == 'same_evidence_greedy':
                assert independent_search(task, blocks, True) == mask
            if name == 'latency':
                order = sorted(range(4), key=lambda i: (-task['action_metadata'][i]['operator_latency_ms'], i))
                assert mask == sum(1 << i for i in order[:2])
            add(name, [mask])
            for o, stored in zip(outcomes, r['evaluation']):
                close(o[mask]['raw_gain_percent'], stored['raw_benefit_percent'])
                close(o[mask]['accepted_gain_percent'], stored['accepted_benefit_percent'])
                assert o[mask]['accepted'] == stored['accepted']
        for config in CONFIGS:
            group = sorted([r for r in result['ppo'] if r['configuration'] == config], key=lambda r: r['seed'])
            assert [r['seed'] for r in group] == SEEDS
            masks, seed_summaries = [], []
            for r in group:
                m = r['plan']['mask']
                assert m in MASKS and r['plan']['selection_source'] == 'calibration_only'
                assert r['eval_access_during_training'] is False and r['unit_fixture_data'] is False
                assert r['training_run_ids'] == result['training_runs'] and 4096 <= r['steps'] <= 4097
                ck = RESULTS / Path(r['checkpoint']).relative_to('/work/planner-matrix-v2')
                assert sha(ck) == r['checkpoint_sha256']
                replayed = None
                state_dim = 19 + (sum(len(x['graph_embedding']) for x in task['action_metadata']) if r['with_embedding'] else 0)
                parameter_count = (state_dim * 64 + 64) + (64 * 64 + 64) + (64 * 5 + 5) + (64 + 1)
                if reload_policies:
                    table = MeasuredTable(task)
                    env = CalibrationEnv(table, 2, r['hardware_mask'], r['with_embedding'], r['drop_embedding'])
                    env.reset(table.splits['train'][0])
                    model = ActorCritic(len(env.state()), 5)
                    model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True))
                    model.eval()
                    assert len(env.state()) == state_dim
                    assert sum(v.numel() for v in model.parameters()) == parameter_count
                    replayed = select_plan(model, table, 2, r['hardware_mask'], r['with_embedding'], r['drop_embedding'])['mask']
                    assert replayed == m
                fit = {'context': cid, 'configuration': config, 'seed': r['seed'], 'mask': m,
                       'steps': r['steps'], 'invalid_attempts': r['training_invalid_attempts'],
                       'state_dim': state_dim, 'policy_parameter_count': parameter_count,
                       'checkpoint_path': str(ck.relative_to(ROOT)), 'checkpoint_sha256': sha(ck),
                       'reloaded_plan_mask': replayed, 'result_path': str(rp.relative_to(ROOT)),
                       'result_sha256': sha(rp), 'task_sha256': sha(tp)}
                fits.append(fit)
                p = add(config, [m], r['seed'])
                seed_summaries.append({'seed': r['seed'], 'mask': m, **p, 'checkpoint_sha256': sha(ck)})
                seed_stats.append({'context': cid, 'configuration': config, 'seed': r['seed'], 'mask': m,
                                   'raw_mean_pct': p['raw_gain_percent']['mean'],
                                   'raw_ci95_lo': p['raw_gain_percent']['descriptive_t95_df4'][0],
                                   'raw_ci95_hi': p['raw_gain_percent']['descriptive_t95_df4'][1],
                                   'accepted_mean_pct': p['accepted_gain_percent']['mean'],
                                   'accepted_ci95_lo': p['accepted_gain_percent']['descriptive_t95_df4'][0],
                                   'accepted_ci95_hi': p['accepted_gain_percent']['descriptive_t95_df4'][1],
                                   'raw_by_process': p['raw_gain_percent']['values'],
                                   'accepted_by_process': p['accepted_gain_percent']['values'],
                                   'checkpoint_sha256': sha(ck), 'physical_n': 5})
                masks.append(m)
                for j, (raw, stored) in enumerate(zip(blocks['eval'], r['evaluation'])):
                    o = outcomes[j][m]
                    assert raw['run_id'] == stored['run_id'] and o['accepted'] == stored['accepted']
                    close(o['raw_gain_percent'], stored['raw_benefit_percent'])
                    close(o['accepted_gain_percent'], stored['accepted_benefit_percent'])
                    seed_rows.append({'context': cid, 'configuration': config, 'seed': r['seed'],
                                      'mask': m, 'run_id': raw['run_id'], 'run_index': raw['run_index'],
                                      **o, 'raw_sha256': refs[3 + j]['sha256'],
                                      'checkpoint_sha256': sha(ck), 'result_sha256': sha(rp)})
            add(config, masks)
            methods[config]['seedwise'] = seed_summaries
            variant_masks[config] = masks
        for name, subset, archived in [
            ('uniform_at_most_2', MASKS, 'uniform_random_at_most_k_expectation'),
            ('uniform_exact_2', [m for m in MASKS if bin(m).count('1') == 2], 'uniform_random_exact_k_expectation')]:
            add(name, subset)
            for j, stored in enumerate(result[archived]):
                close(methods[name]['raw_gain_percent']['values'][j], stored['raw_benefit_percent'])
                close(methods[name]['accepted_gain_percent']['values'][j], stored['accepted_benefit_percent'])
        oracle_masks = [min(((-o[m]['raw_gain_percent'], m) for m in MASKS if o[m]['accepted']))[1] for o in outcomes]
        oracle_raw = [outcomes[j][m]['raw_gain_percent'] for j, m in enumerate(oracle_masks)]
        for value, stored in zip(oracle_raw, result['heldout_feasible_oracle']):
            close(value, stored['accepted_benefit_percent'])
        methods['heldout_feasible_oracle'] = {'posthoc': True, 'not_a_deployable_comparator': True,
                                             'masks_per_process': oracle_masks,
                                             'raw_gain_percent': summary(oracle_raw),
                                             'accepted_gain_percent': summary(oracle_raw)}
        flat.append({'context': cid, 'device': task['device_id'], 'method': 'heldout_feasible_oracle_POSTHOC',
                     'masks': oracle_masks, 'raw_mean_pct': st.mean(oracle_raw),
                     'raw_ci95_lo': summary(oracle_raw)['descriptive_t95_df4'][0],
                     'raw_ci95_hi': summary(oracle_raw)['descriptive_t95_df4'][1],
                     'raw_by_process': oracle_raw, 'physical_n': 5, 'task_sha256': sha(tp), 'result_sha256': sha(rp)})
        for lhs, rhs in [(c, control) for c in CONFIGS for control in CONTROLS] + [
            ('ppo_with_heldout_embedding', 'ppo_embedding_zeroed_same_capacity'),
            ('ppo_hardware_mask', 'ppo_without_hardware_mask')]:
            for metric in ['raw_gain_percent', 'accepted_gain_percent']:
                a, b = methods[lhs][metric]['values'], methods[rhs][metric]['values']
                s = summary(x - y for x, y in zip(a, b))
                contrasts.append({'context': cid, 'lhs': lhs, 'rhs': rhs, 'metric': metric,
                                  'mean_difference_pp': s['mean'], 'ci95_lo_pp': s['descriptive_t95_df4'][0],
                                  'ci95_hi_pp': s['descriptive_t95_df4'][1], 'paired_process_differences_pp': s['values'],
                                  'physical_n': 5, 'descriptive_not_multiplicity_adjusted': True})
        mask_rows.append({'context': cid, 'supported_actions': safe, 'all_actions_supported': all(safe),
                          'all_16_coalitions_valid_and_equivalent': True,
                          'masked_seed_masks': variant_masks[CONFIGS[0]], 'unmasked_seed_masks': variant_masks[CONFIGS[1]],
                          'mask_toggle_equal_by_seed': variant_masks[CONFIGS[0]] == variant_masks[CONFIGS[1]],
                          'total_training_invalid_attempts': sum(r['training_invalid_attempts'] for r in result['ppo']),
                          'validity_effect_identifiable': False if all(safe) else None,
                          'minimum_eval_quality_retention': min(o[m]['quality_retention'] for o in outcomes for m in range(16))})
        contexts.append({'context': cid, 'task_id': task['task_id'], 'device_id': task['device_id'],
                         'task_path': str(tp.relative_to(ROOT)), 'task_sha256': sha(tp),
                         'result_path': str(rp.relative_to(ROOT)), 'result_sha256': sha(rp),
                         'raw_sources': refs, 'methods': methods, 'budget_at_most': 2,
                         'quality_n_unique_eval_samples': 100,
                         'uncertainty_unit': '5 physical processes after averaging 5 seeds within process',
                         'embedding_provenance': [r['embedding_provenance'] for r in task['action_metadata']]})
    assert len(fits) == 160 and len(seed_rows) == 800
    macro = []
    for name in CONFIGS + CONTROLS:
        values = [c['methods'][name]['raw_gain_percent']['mean'] for c in contexts]
        macro.append({'method': name, 'eight_context_equal_weight_mean_raw_pct': st.mean(values),
                      'eight_context_values': values, 'population_inference': False})
    output = {'analysis_freeze_sha256': sha(HERE / 'analysis-freeze.json'), 'contexts': contexts,
              'policy_fits': len(fits), 'unique_hardware_eval_processes': 40,
              'policy_seed_x_process_scored_records': len(seed_rows),
              'saved_policy_reloads': 160 if reload_policies else 0,
              'paired_contrasts': contrasts, 'mask_domain_audit': mask_rows, 'macro_descriptive': macro,
              'scope': 'New offline fits on physical calibration tables, not historical FITAS reproduction. Equal intervention budget is not equal training compute.'}
    dump('planner-full-panel.json', output)
    csvout('planner-context-panel.csv', flat)
    csvout('planner-seedwise-process.csv', seed_rows)
    csvout('planner-seedwise-summary.csv', seed_stats)
    csvout('planner-paired-contrasts.csv', contrasts)
    csvout('planner-fit-manifest.csv', fits)
    csvout('planner-mask-domain.csv', mask_rows)
    return output


def migration():
    tracepath = METHODS / 'bridge-early-results/results/a100-to-910b-resnet18-b1.json'
    trace = read(tracepath)
    early = ROOT / 'v2-实证增强/evidence/early-tasks'
    sourcepath = early / 'resnet18_b1_imagenet_val10class_v2_2c4332169b.json'
    targetpath = early / 'resnet18_b1_imagenet_val10class_v2_bac88cfcaa.json'
    assert sha(sourcepath) == trace['source_task_sha256'] and sha(targetpath) == trace['target_task_sha256']
    source, sblocks, srefs, ssafe = validate_task(sourcepath)
    target, tblocks, trefs, tsafe = validate_task(targetpath)
    for refs, archived in [(srefs, trace['source_artifacts']), (trefs, trace['target_artifacts'])]:
        assert [(r['run_id'], r['sha256']) for r in refs] == [(r['run_id'], r['sha256']) for r in archived]
    sys.path.insert(0, str(METHODS))
    from migration_bridge import run_bridge
    replay = run_bridge(source, target, sblocks['train'], tblocks['train'], budget=2,
                        total_permutations=500, seed=42, reference_permutations=0)
    for key in ['source_physical_mean_ms', 'target_physical_mean_ms', 'physical_migration_gap_ms',
                'source_operator_path_ms', 'target_operator_path_ms', 'operator_surrogate_gap_ms',
                'unattributed_physical_gap_residual_ms']:
        close(replay['context'][key], trace['context'][key])
    priority = replay['action_priority']
    assert priority['selected_mask'] == trace['action_priority']['selected_mask'] == 6
    operator_rows = []
    for op, value in replay['operator_estimate']['values'].items():
        close(value, trace['operator_estimate']['values'][op])
        operator_rows.append({'operator_id': op, 'source_node_profile_ms': replay['context']['source_node_costs_ms'][op],
                              'target_node_profile_ms': replay['context']['target_node_costs_ms'][op],
                              'migration_phi_ms': value,
                              'pair_mean_se_ms': replay['operator_estimate']['standard_errors'][op],
                              'source_profile_run_ids': replay['context']['source_profile_run_ids'],
                              'target_profile_run_ids': replay['context']['target_profile_run_ids'],
                              'source_raw_sha256': [r['sha256'] for r in srefs[:3]],
                              'target_raw_sha256': [r['sha256'] for r in trefs[:3]],
                              'trace_sha256': sha(tracepath)})
    actions = []
    for row, stored in zip(priority['actions'], trace['action_priority']['actions']):
        for k, v in row.items():
            if isinstance(v, float):
                close(v, stored[k])
            else:
                assert v == stored[k], k
        profiles = [next(a for a in r['action_profiles'] if a['id'] == row['action_id']) for r in tblocks['train']]
        base = st.mean(latency(a['baseline']) for a in profiles)
        candidate = st.mean(latency(a['candidate']) for a in profiles)
        close(base, row['target_local_baseline_ms'])
        close(candidate, row['target_local_candidate_ms'])
        close(base - candidate, row['raw_target_headroom_ms'])
        actions.append({**row, 'target_raw_sha256': [r['sha256'] for r in trefs[:3]],
                        'selected': row['action_id'] in priority['selected_action_ids']})
    outcomes = []
    for s, t, sr, tr, old in zip(sblocks['eval'], tblocks['eval'], srefs[3:], trefs[3:], trace['physical_eval']):
        assert s['run_index'] == t['run_index'] == old['paired_run_index']
        for field in ['task_id', 'model_id', 'input_shape', 'weights_state_sha256', 'input_tensor_sha256', 'dataset_manifest_sha256', 'dataset_id', 'model_configuration']:
            assert s[field] == t[field], field
        sbase = next(r for r in s['coalitions'] if r['mask'] == 0)
        tcand = next(r for r in t['coalitions'] if r['mask'] == 6)
        sq, smap = quality(sbase)
        tq, tmap = quality(tcand)
        assert set(smap) == set(tmap) and all(smap[i][0] == tmap[i][0] for i in smap)
        gate = strict_gate(t, 6, tsafe)
        global_q = tq / sq
        accepted = gate['accepted'] and global_q >= .99
        s_ms = latency(sbase)
        row = {'source_run_id': s['run_id'], 'target_run_id': t['run_id'], 'run_index': s['run_index'],
               'source_raw_path': sr['path'], 'source_raw_sha256': sr['sha256'],
               'target_raw_path': tr['path'], 'target_raw_sha256': tr['sha256'],
               'source_baseline_ms': s_ms, 'target_baseline_ms': gate['baseline_ms'],
               'requested_target_candidate_ms': gate['candidate_ms'], 'selected_mask': 6,
               'physical_gap_before_ms': gate['baseline_ms'] - s_ms,
               'physical_gap_after_requested_ms': gate['candidate_ms'] - s_ms,
               'raw_target_gain_percent': gate['raw_gain_percent'],
               'accepted_target_gain_percent': gate['raw_gain_percent'] if accepted else 0.0,
               'source_top1': sq, 'target_baseline_top1': gate['baseline_top1'], 'target_candidate_top1': tq,
               'source_relative_retention': global_q, 'target_relative_retention': gate['quality_retention'],
               'accepted': accepted, 'reason': gate['reason'] if global_q >= .99 else 'cross_device_quality_below_source_threshold',
               'fallback': 'none' if accepted else ('target_baseline' if gate['baseline_top1'] / sq >= .99 else 'original_source_baseline'),
               'unique_quality_samples': len(smap), 'weights_state_sha256': s['weights_state_sha256'],
               'eval_input_tensor_sha256': s['input_tensor_sha256'], 'trace_sha256': sha(tracepath)}
        close(row['source_baseline_ms'], old['source_physical_baseline_ms'])
        close(row['requested_target_candidate_ms'], old['target_measured_candidate_ms'])
        close(row['raw_target_gain_percent'], old['raw_target_optimization_benefit_percent'])
        assert accepted == old['migration_accepted']
        outcomes.append(row)
    metrics = {k: summary(r[k] for r in outcomes) for k in ['source_baseline_ms', 'target_baseline_ms',
               'requested_target_candidate_ms', 'physical_gap_before_ms', 'physical_gap_after_requested_ms',
               'raw_target_gain_percent', 'accepted_target_gain_percent']}
    result = {'case_selection': 'User-prespecified ResNet-18 batch1, regardless of outcome',
              'identity': {k: replay['context'][k] for k in ['task_id', 'model_id', 'input_shape', 'source_device_id', 'target_device_id', 'weights_state_sha256', 'dataset_id', 'dataset_manifest_sha256', 'model_configuration']},
              'sources': {'trace_path': str(tracepath.relative_to(ROOT)), 'trace_sha256': sha(tracepath),
                          'source_task_path': str(sourcepath.relative_to(ROOT)), 'source_task_sha256': sha(sourcepath),
                          'target_task_path': str(targetpath.relative_to(ROOT)), 'target_task_sha256': sha(targetpath),
                          'source_raw': srefs, 'target_raw': trefs},
              'calibration_diagnosis': replay['context'], 'operator_attribution_count': len(operator_rows),
              'operator_attribution': operator_rows, 'action_priority': priority, 'action_headroom_raw_audit': actions,
              'selection_freeze': {'mask': 6, 'actions': priority['selected_action_ids'], 'budget_at_most': 2,
                  'archived_selection_trace_sha256': sha(tracepath), 'selection_evidence': 'source/target calibration1-3 only',
                  'procedure': 'run_bridge computes selection before caller scores physical_eval; CPU replay reproduced it',
                  'prospective_hardware_run_claim': False,
                  'qualification': 'Existing outcome tables were collected already; this proves calibration-only offline selection, not a newly timestamped prospective intervention.'},
              'heldout_same_chain_outcomes': outcomes, 'heldout_summary': metrics,
              'quality': {'threshold': .99, 'all_source_and_target_retention': [r['source_relative_retention'] for r in outcomes],
                          'unique_eval_images': 100, 'repeated_images_not_independent_accuracy_samples': True,
                          'accepted_blocks': sum(r['accepted'] for r in outcomes), 'total_blocks': 5},
              'gate_interpretation': 'Applied offline to measured heldout tables. Accepted-score zeros represent the baseline fallback rule; no deployed rollback execution is established.',
              'nonconflation': ['A100 target-side PPO 2.097% uses a different selection and target; absent from this migration result.',
                               'Fresh NPU run101-105 is an independent followup, not the original run4-8 chain.',
                               'Hybrid operator path game is non-executable; its residual to physical model gap remains explicit.',
                               'This is independent revalidation, not recovered original FITAS evidence.']}
    dump('migration-resnet18-b1-complete.json', result)
    csvout('migration-operators.csv', operator_rows)
    csvout('migration-actions-headroom.csv', actions)
    csvout('migration-heldout-process.csv', outcomes)
    return result


def v4_crosscheck(panel_result):
    p = ROOT / 'v4-提分条件补强/case/all-context-recomputation.json'
    previous = read(p)
    checked = 0
    for c in panel_result['contexts']:
        t = read(ROOT / c['task_path'])
        v = next(x for x in previous['original_all16_contexts']
                 if (x['device'], x['model'], x['batch']) == (t['device_id'], t['model_id'], t['input_shape'][0]))
        for name in CONFIGS + ['latency', 'uniform_at_most_2', 'uniform_exact_2']:
            for a, b in zip(c['methods'][name]['raw_gain_percent']['values'], v['plans'][name]['raw_gain_percent']['values']):
                close(a, b)
                checked += 1
        for a, b in zip(c['methods']['same_evidence_exhaustive_search']['raw_gain_percent']['values'],
                        v['plans']['calibration_exhaustive']['raw_gain_percent']['values']):
            close(a, b)
            checked += 1
    result = {'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'status': 'passed',
              'raw_process_values_crosschecked': checked,
              'qualification': 'Same canonical raw tables; agreement is an arithmetic check, not new evidence or replication.'}
    dump('v4-crosscheck.json', result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reload-policies', action='store_true')
    args = ap.parse_args()
    inputs = freeze_check()
    p = panel(args.reload_policies)
    m = migration()
    v4 = v4_crosscheck(p)
    report = {'status': 'passed', 'input_hashes_verified': len(inputs), 'fit_checkpoint_hashes_verified': 160,
              'fit_checkpoint_reloads_verified': p['saved_policy_reloads'],
              'raw_hardware_processes_in_planner_panel': 64, 'eval_hardware_processes_in_panel': 40,
              'migration_same_chain_source_and_target_processes': 16,
              'migration_operator_attributions_replayed': m['operator_attribution_count'],
              'v4_raw_values_crosschecked': v4['raw_process_values_crosschecked'],
              'analysis_freeze_sha256': sha(HERE / 'analysis-freeze.json'), 'analyzer_sha256': sha(__file__),
              'mutation_scope': 'v7-剩余条件实证闭环/planning only',
              'no_new_hardware_measurements': True, 'no_retraining': True,
              'mask_claim': 'All actions supported; zero effect cannot establish benefit on invalid actions.'}
    dump('recomputation-verification.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
