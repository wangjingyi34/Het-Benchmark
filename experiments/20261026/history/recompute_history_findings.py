#!/usr/bin/env python3
"""Read-only historical arithmetic/configuration audit; outputs are analysis, not new measurements."""
from pathlib import Path
import argparse, ast, collections, csv, hashlib, json, math, random

HERE = Path(__file__).resolve().parent
CODE = HERE / 'source-data'
OUTPUT = HERE / 'results'
UPSTREAM = json.loads((HERE / 'source-manifest.json').read_text())
SOURCE_MAP = {x['upstream_path']: x for x in UPSTREAM['files']}

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def source(rel, locator):
    p = CODE / rel
    entry = SOURCE_MAP[rel]
    assert digest(p) == entry['bundled_sha256'], rel
    return {'path': 'source-data/' + rel, 'sha256': digest(p), 'public_url': entry['upstream_url'], 'upstream_sha256': entry['upstream_sha256'], 'locator': locator}

def load(rel):
    return json.loads((CODE / rel).read_text())

def write(name, data):
    (OUTPUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def assignments(path):
    out = {}
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                pass
    return out

def main():
    for entry in UPSTREAM['files']:
        assert digest(HERE / entry['path']) == entry['bundled_sha256'], entry['path']
    OUTPUT.mkdir(parents=True, exist_ok=True)
    models = load('data/model_dataset.json')['models']
    schema = load('data/paper_schema_counts.json')
    ids = sorted(m['model_id'] for m in models)
    random.Random(schema['split']['seed']).shuffle(ids)
    ntrain, nval, _ = schema['split']['model_counts']
    split = {'train': sorted(ids[:ntrain]), 'validation': sorted(ids[ntrain:ntrain+nval]), 'test': sorted(ids[ntrain+nval:])}
    modelmap = {m['model_id']: m for m in models}
    stats = {part: {'models': len(mid), 'operator_instances': sum(len(modelmap[x]['operators']) for x in mid), 'categories': dict(collections.Counter(modelmap[x]['category'] for x in mid))} for part, mid in split.items()}
    splitdoc = {'status': 'reconstructed_from_released_deterministic_recipe_not_original_run_attestation', 'seed': schema['split']['seed'], 'split': split, 'group_statistics': stats, 'sources': [source('src/split_mohkg.py','lines 13-22; 35-52'), source('data/model_dataset.json','$.models[*].model_id/operators/category'), source('data/paper_schema_counts.json','$.split')]}
    write('original-source-model-split.json', splitdoc)
    graph = load('data/moh_kg.json')
    perf = [(i,e) for i,e in enumerate(graph['edges']) if e['edge_type']=='r_perf']
    observed = {'declared_canonical_counts': schema, 'actual_legacy_counts': dict(collections.Counter(e['edge_type'] for e in graph['edges'])), 'actual_legacy_total': len(graph['edges']), 'legacy_hardware': [n for n in graph['nodes'] if n['node_type']=='hardware'], 'legacy_performance_edges': [{'json_pointer': f'/edges/{i}', **e} for i,e in perf], 'legacy_performance_edges_are_family_level': all(e['source_id'].startswith('op_type_') for i,e in perf), 'canonical5000_per_edge_records_recovered': False, 'opportunity_count_arithmetic': len(modelmap) and sum(len(m['operators']) for m in models)*schema['node_counts']['hardware'], 'opportunity_denominator_is_not_observed_profile_count': True, 'sources': [source('data/moh_kg.json','$.metadata; $.statistics; $.edges'),source('data/paper_schema_counts.json','$.edge_counts; $.node_counts')]}
    write('original-graph-observed-counts.json', observed)
    # Exact reproduction of the old CSV from its deterministic source arithmetic.
    regen = assignments(CODE/'src/regenerate_tables.py')
    old = list(csv.DictReader((CODE/'results/table8_cross_platform_prediction.csv').open()))
    rng = random.Random(42)
    reproduced = []
    for model in regen['representative_models']:
        for platform, ratio in regen['platform_ratios'].items():
            actual = regen['base_latencies'].get(model, 5.0)/ratio
            predicted = actual*(1+rng.uniform(-.08,.08))
            row = {'model': model, 'source_platform': 'CUDA', 'target_platform': platform, 'predicted_latency_ms': f'{predicted:.2f}', 'actual_latency_ms': f'{actual:.2f}', 'prediction_error_percent': f'{abs(predicted-actual)/actual*100:.2f}'}
            reproduced.append(row)
    write('old50-prediction-reproduction.json', {'row_count': len(old), 'model_count': len(set(r['model'] for r in old)), 'platforms': sorted(set(r['target_platform'] for r in old)), 'all_rows_exact_match': old==reproduced, 'reported_error_column_mean_percent': sum(float(r['prediction_error_percent']) for r in old)/len(old), 'recomputed_from_rounded_latency_mean_percent': sum(abs(float(r['predicted_latency_ms'])-float(r['actual_latency_ms']))/float(r['actual_latency_ms'])*100 for r in old)/len(old), 'rows': reproduced, 'provenance_class': 'deterministic_ratio_and_random_perturbation_demo', 'eligible_for_original_four_test_model_or_five_hardware_LOPO': False, 'sources': [source('results/table8_cross_platform_prediction.csv','header and all 50 rows'), source('src/regenerate_tables.py','lines 202-247')]})
    table = list(csv.DictReader((CODE/'results/paper_tables/prediction_mre.csv').open()))
    fam = {r['operator_family']:r for r in table}
    common=['MatMul','Conv2D','Attention']; hist=common+['LayerNorm','Activation']; eight=[r['operator_family'] for r in table if r['operator_family']!='Overall']
    arithmetic = {'five_family_equal_macro': sum(float(fam[x]['test']) for x in hist)/len(hist), 'five_family_set':hist, 'eight_family_equal_macro':sum(float(fam[x]['test']) for x in eight)/len(eight), 'eight_family_set':eight, 'common3_test_equal_macro':sum(float(fam[x]['test']) for x in common)/3, 'common3_roofline_equal_macro':sum(float(fam[x]['roofline']) for x in common)/3, 'original_micro_MRE_certified':False, 'source':source('results/paper_tables/prediction_mre.csv','test/roofline columns; named family subsets; no per-query weights available')}
    write('historical-overall-recomputation.json',arithmetic)
    cross=load('results/cross_platform_results.json'); specs=assignments(CODE/'experiments/exp_cross_platform_eval.py')['HARDWARE_SPECS']
    rows=[]
    for pid, spec in specs.items():
        entries=[x['platforms'][pid] for bucket in ['operators','models'] for x in cross[bucket].values()]
        rows.append({'platform_id':pid,'device_name':spec['description'],'source_operator_summary_rows':len(cross['operators']),'source_model_summary_rows':len(cross['models']),'measured_flag_true':sum(x.get('measured') is True for x in entries),'measured_flag_false':sum(x.get('measured') is False for x in entries),'canonical5000_linked_records':None,'per_repeat_raw_observations_found_in_this_file':0,'strict_LOPO_eligible':False,'reason':'No operator-instance/model split binding or raw trace lineage; non-A100 entries additionally use hardware-spec estimates.'})
    with (OUTPUT/'original-five-platform-label-availability.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    write('original-five-platform-label-availability.json',{'rows':rows,'counts_are_source_declarations_not_independent_attestation':True,'sources':[source('results/cross_platform_results.json','$.operators[*].platforms; $.models[*].platforms'),source('experiments/exp_cross_platform_eval.py','lines 29-55; 170-195; 225-241')]})
    derived=[]
    for bucket in ['operators','models']:
        for name,item in cross[bucket].items():
            typ=item.get('type','conv' if 'ResNet' in name else 'mixed');a100=item['platforms']['A100_80GB']['latency_ms']
            for pid,values in item['platforms'].items():
                if pid=='A100_80GB':continue
                s=specs[pid]
                if typ in ['matmul','conv','ffn','linear']:ratio=.8*s['compute_ratio']+.2*s['memory_bw_ratio']
                elif typ in ['layernorm','softmax','embedding','activation']:ratio=.1*s['compute_ratio']+.9*s['memory_bw_ratio']
                else:ratio=.5*s['compute_ratio']+.5*s['memory_bw_ratio']
                calc=a100/ratio
                derived.append({'pointer':f'/{bucket}/{name}/platforms/{pid}','source_case_type':typ,'platform_id':pid,'a100_summary_ms':a100,'calculated_ratio':ratio,'calculated_ms':calc,'stored_ms':values['latency_ms'],'measured_flag':values['measured'],'equal_with_1e_12_relative_tolerance':math.isclose(calc,values['latency_ms'],rel_tol=1e-12)})
    write('original-cross-platform-derived-reproduction.json',{'scope':'Offline arithmetic from archived hardware-spec-ratio source; not measurement or LOPO','derived_record_count':len(derived),'all_measured_flags_false':all(r['measured_flag'] is False for r in derived),'all_values_reproduced':all(r['equal_with_1e_12_relative_tolerance'] for r in derived),'records':derived,'sources':[source('experiments/exp_cross_platform_eval.py','lines81-111;132-167;184-195;225-241'),source('results/cross_platform_results.json','$.operators;$.models')]})
    configs=[
      {'name':'MatMul 4096x4096','source_key':'matmul_4096','inputs':[[4096,4096]],'operation':'torch.mm(x,x.T)','layout':'default contiguous x; transposed view for right operand','batch':None,'kernel':None,'heads':None},
      {'name':'Conv2D 224x224','source_key':'conv2d_224','inputs':[[1,64,224,224]],'operation':'Conv2d(64,128,3,padding=1)','layout':'default contiguous NCHW','batch':1,'kernel':[3,3],'channels_in':64,'channels_out':128,'heads':None},
      {'name':'Attention sequence 512','source_key':'attention_512','inputs':[[1,512,768]],'operation':'MultiheadAttention(768,12,batch_first=True), m(x,x,x)[0]','layout':'batch,sequence,embedding','batch':1,'heads':12,'head_dim':64},
      {'name':'LayerNorm width 768','source_key':'layernorm_768','inputs':[[32,512,768]],'operation':'LayerNorm(768)','layout':'default contiguous','batch':32},
      {'name':'Softmax 512x512','source_key':'softmax_512x512','inputs':[[8,12,512,512]],'operation':'Softmax(dim=-1)','layout':'default contiguous','batch':8},
      {'name':'Linear width 4096','source_key':'linear_4096','inputs':[[32,512,768]],'operation':'Linear(768,3072)','layout':'default contiguous','batch':32,'printed_name_matches_dimensions':False}]
    for c in configs:c.update({'dtype':'float16','source_config_only':True,'timing':{'warmup':20,'repeats':100,'synchronize':'torch.cuda.synchronize before/after wall clock'},'source':source('experiments/exp_cross_platform_eval.py','lines 59-77; 132-167; 170-176')})
    write('original-six-operator-source-configurations.json',{'run_identity_certified':False,'rows':configs,'boundary':'Recovered source declarations. The printed Linear width 4096 label disagrees with 768->3072 in the candidate source, and no run/config hash binds Table 4 to this source. Attention in cross_platform_profiler.py is a different batch=8 candidate.'})
    baseline=load('results/baseline_real_comparison.json'); mobile=baseline['mlperf']['MobileNetV2']['batch_1']['timing']['mean_ms'];mobile_cross=cross['models']['MobileNetV2']['platforms']['A100_80GB']['latency_ms']
    mobilefinding={'paper_surrogate_ms':4.441,'source_baseline_mean_ms':mobile,'source_rounds_to_surrogate':round(mobile,3)==4.441,'paper_model_latency_ms':4.35,'candidate_cross_platform_baseline_ms':mobile_cross,'candidate_rounds_to_4_35':round(mobile_cross,2)==4.35,'same_run_certified':False,'source_model_config':{'model':'torchvision.models.mobilenet_v2(weights=None)','shape':[1,3,224,224],'dtype':'float16','eval':True,'no_grad':True,'warmup':10,'repeats':50},'mixed_ratio_rule':'0.5*compute_ratio + 0.5*memory_bw_ratio; equal to attention case by heuristic, not a statement that MobileNet contains attention','sources':[source('results/paper_tables/model_latency_ms.csv','MobileNetV2 row'),source('results/paper_tables/surrogate_speedup.csv','MobileNetV2 row'),source('results/baseline_real_comparison.json','$.mlperf.MobileNetV2.batch_1.timing.mean_ms'),source('results/cross_platform_results.json','$.models.MobileNetV2.platforms'),source('experiments/exp_cross_platform_eval.py','lines 102-108; 206-241'),source('experiments/exp_baseline_real.py','lines 207-210; 218-230')]}
    mobilefinding['baseline_evidence_class']='source_declared_hardware_timing_summary_without_independent_capture_authentication'
    mobilefinding['baseline_declared_metadata']={k:baseline[k] for k in ['experiment','timestamp','hardware','cuda_version','pytorch_version']}
    mobilefinding['capture_provenance_audit']={'timing_source':'exp_baseline_real.py:24-48 uses CUDA synchronization and perf_counter, but exports summary statistics only','stored_repeat_vector':False,'run_id':False,'recorded_code_hash':False,'recorded_input_or_weight_hashes':False,'immutable_script_to_run_binding':False,'independently_authenticated_capture':False,'claim_allowed':'4.441 equals the rounded declared historical JSON mean; this is numerical provenance only','claim_prohibited':'The source summary proves a physical run, or proves 4.35 and4.441 have the same run/configuration'}
    mobilefinding['sources'].append(source('experiments/exp_baseline_real.py','lines 24-48; 414-431'))
    write('mobilenet-source-reconciliation.json',mobilefinding)
    fitas=load('results/fitas_migration_results.json');cost=fitas['migration_summary']['Hardware Cost ($/month)'];before=cost['Before (A100)'];after=cost['After (Ascend 910B)']
    write('fitas-source-reconciliation.json',{'assumed_monthly_costs_usd':[before,after],'arithmetic_reduction_percent':(before-after)/before*100,'observed_cost_receipt_present':False,'real_measurements_count':len(fitas['real_measurements']),'latency_improvement_percent_in_current_source':fitas['migration_summary']['End-to-End Latency (hours)']['Improvement (%)'],'original15_6_certified':False,'physical_case_rerun_requirements':['releaseable exact workload/model+weights/input hashes','A100 PCIe80GB and Ascend910B run identity and complete software stacks','same quality target and demand constraint','paired per-repeat before/after latency and action ledger','actual cost basis or clearly stated scenario pricing'], 'sources':[source('results/fitas_migration_results.json','$.real_measurements; $.hardware_specs; $.migration_summary'),source('experiments/exp_fitas_migration.py','lines 49-102; 112; 193-209; 253-255')]})
    family_counts = {part: dict(collections.Counter(op['type'] for mid in mids for op in modelmap[mid]['operators'])) for part, mids in split.items()}
    write('original-catalog-family-balance.json', {'scope': 'Released catalog declarations, not physical labels', 'source_model_split': split, 'operator_family_counts_by_group': family_counts, 'source': source('data/model_dataset.json', '/models/*/operators/*/type'), 'original_performance_label_balance_certified': False})
    print(json.dumps({'old50_exact_match': old==reproduced, 'derived32_exact_match': all(r['equal_with_1e_12_relative_tolerance'] for r in derived), 'models':len(models),'operator_instances':sum(len(m['operators']) for m in models),'original_micro_MRE_certified':False},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'results');args=parser.parse_args();OUTPUT=args.output;main()
