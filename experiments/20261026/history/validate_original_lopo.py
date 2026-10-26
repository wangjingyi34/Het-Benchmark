#!/usr/bin/env python3
"""Validate original-five-platform evidence and strict source-model/platform splits.

This tool never creates observations, fills missing labels, or launches hardware.
Evidence integrity is necessary but cannot by itself authenticate the instrument.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

PLATFORMS = {
    'A100_80GB': 'NVIDIA A100 80GB PCIe',
    'Ascend_910B': 'Huawei Ascend 910B',
    'MLU370_X8': 'Cambricon MLU370-X8',
    'Intel_GPU_Max': 'Intel GPU Max 1550',
    'Intel_Xeon_8380': 'Intel Xeon 8380',
}
STATUSES = {'measured', 'calibrated', 'estimated', 'failed', 'unsupported', 'missing'}
ORIGINAL_CATALOG_SHA256 = '5be0c5aac386bc366552cb7672ebeadbd03177e795bfe84866d9acfdc28b1245'
FEATURES = {'operator_family', 'input_shapes', 'output_shapes', 'dtype', 'layout',
            'kernel_size', 'stride', 'padding', 'groups', 'channels_in', 'channels_out',
            'attention_heads', 'head_dimension', 'batch_size', 'sequence_length',
            'flops', 'bytes_read', 'bytes_written', 'hardware_peak_flops',
            'hardware_memory_bandwidth', 'hardware_memory_capacity'}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def positive(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0

def pointer(document, locator):
    if locator == '': return document
    if not isinstance(locator, str) or not locator.startswith('/'): raise ValueError('JSON pointer must begin with /')
    node = document
    for key in locator[1:].split('/'):
        key = key.replace('~1','/').replace('~0','~')
        node = node[int(key)] if isinstance(node,list) else node[key]
    return node

class Audit:
    def __init__(self):
        self.errors = []
        self.counts = collections.Counter()
        self.cache = {}

    def fail(self, code, at, detail):
        self.counts[code] += 1
        if len(self.errors) < 150:
            self.errors.append({'code': code, 'at': at, 'detail': detail})

    def artifact(self, ref, at):
        if not isinstance(ref,dict) or not all(k in ref for k in ['path','sha256','pointer']):
            self.fail('MISSING_SOURCE_BINDING',at,'Require path, sha256 and exact JSON pointer');return None
        p = Path(ref['path'])
        if not p.is_absolute(): p = Path(__file__).resolve().parent / p
        try:
            if not p.is_file(): raise ValueError('Source file unavailable')
            if str(p) not in self.cache:
                raw=p.read_bytes();self.cache[str(p)]=(hashlib.sha256(raw).hexdigest(),json.loads(raw))
            digest, document = self.cache[str(p)]
            if digest != ref['sha256']: raise ValueError('SHA256 mismatch')
            return pointer(document, ref['pointer'])
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            self.fail('SOURCE_INTEGRITY',at,str(e));return None

def expected_split(model_ids):
    ids=sorted(set(model_ids));random.Random(42).shuffle(ids)
    return {'train':sorted(ids[:27]),'validation':sorted(ids[27:30]),'test':sorted(ids[30:])}

def measured_record(record, audit):
    rid=record.get('record_id','<missing>')
    samples=record.get('samples_ms',[])
    if not isinstance(samples,list) or len(samples)<5 or not all(positive(x) for x in samples):
        audit.fail('RAW_REPEATS_REQUIRED',rid,'At least five positive finite physical repeats; a mean/std alone is insufficient')
    if record.get('statistic') not in {'mean','median'} or not positive(record.get('latency_ms')):
        audit.fail('MEASUREMENT_UNITS',rid,'Explicit positive latency_ms and mean/median statistic required')
    elif isinstance(samples,list) and samples and all(positive(x) for x in samples):
        value=statistics.mean(samples) if record['statistic']=='mean' else statistics.median(samples)
        if not math.isclose(value,record['latency_ms'],rel_tol=1e-10,abs_tol=1e-10):audit.fail('AGGREGATION_MISMATCH',rid,'Recorded aggregate differs from samples')
    config=record.get('configuration',{})
    if not isinstance(config,dict) or not all(k in config for k in ['input_shapes','output_shapes','dtype','layout','operator_parameters']):
        audit.fail('INCOMPLETE_CONFIGURATION',rid,'Require concrete input/output shapes, dtype, layout and operator_parameters')
    else:
        if not all(isinstance(config[k],str) and config[k] for k in ['dtype','layout']) or not isinstance(config['operator_parameters'],dict):
            audit.fail('INCOMPLETE_CONFIGURATION',rid,'dtype/layout must be explicit nonempty strings; operator_parameters must be an object')
        for shapes in [config['input_shapes'],config['output_shapes']]:
            if not isinstance(shapes,list) or not all(isinstance(s,list) and all(isinstance(n,int) and not isinstance(n,bool) and n>0 for n in s) for s in shapes):
                audit.fail('SYMBOLIC_SHAPE',rid,'All shape dimensions must be positive integers')
        if record.get('configuration_sha256')!=canonical_sha(config):audit.fail('CONFIG_HASH',rid,'Configuration hash mismatch')
    for key in ['run_id','captured_at_utc','model_weights_sha256','input_sha256','software_environment_sha256']:
        if not record.get(key):audit.fail('INCOMPLETE_RUN_IDENTITY',rid,key)
    for key in ['model_weights_sha256','input_sha256','software_environment_sha256']:
        value=record.get(key,'')
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):audit.fail('INVALID_DIGEST',rid,key)
    if record.get('parent_record_ids'):audit.fail('MEASURED_HAS_DERIVATION',rid,'Derived records cannot be labeled measured')
    raw=audit.artifact(record.get('raw_evidence'),rid+'/raw_evidence')
    if isinstance(raw,dict):
        for key in ['record_id','status','platform_id','source_model_id','operator_instance_id','run_id','captured_at_utc','samples_ms','configuration_sha256','model_weights_sha256','input_sha256','software_environment_sha256']:
            if raw.get(key)!=record.get(key):audit.fail('RAW_RECORD_MISMATCH',rid,key)
    elif raw is not None:audit.fail('RAW_RECORD_TYPE',rid,'Pointer must identify a record object')
    hardware=audit.artifact(record.get('hardware_evidence'),rid+'/hardware_evidence')
    if isinstance(hardware,dict):
        pid=record.get('platform_id')
        if hardware.get('platform_id')!=pid or hardware.get('device_name')!=PLATFORMS.get(pid):audit.fail('WRONG_HARDWARE',rid,'Exact original platform identity required')
        if hardware.get('partition')!='full_device':audit.fail('PARTITION_SUBSTITUTION',rid,'MIG or other partition is a distinct runtime environment')
        for key in ['device_variant','driver_version','runtime_version','framework_version','host_inventory_sha256']:
            if not hardware.get(key):audit.fail('HARDWARE_ENVIRONMENT_INCOMPLETE',rid,key)
    elif hardware is not None:audit.fail('HARDWARE_RECORD_TYPE',rid,'Pointer must identify a hardware object')

def validate_fold(fold, records, modelpart, audit):
    target=fold.get('held_out_platform')
    if target not in PLATFORMS:audit.fail('WRONG_HOLDOUT_PLATFORM',str(target),'Require one original hardware platform')
    sets={k:set(fold.get(k+'_record_ids',[])) for k in ['train','validation','test']}
    if any(sets[a]&sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):audit.fail('SPLIT_OVERLAP',str(target),'Record IDs cross split roles')
    for part,ids in sets.items():
        if not ids:audit.fail('EMPTY_FOLD_PARTITION',str(target),part)
        expected={rid for rid,r in records.items() if r.get('status')=='measured' and modelpart.get(r.get('source_model_id'))==part and ((r.get('platform_id')==target) if part=='test' else (r.get('platform_id')!=target))}
        if ids!=expected:audit.fail('FOLD_COVERAGE_MISMATCH',str(target),part+' must contain every eligible measured record, no selection by test performance')
        for rid in ids:
            r=records.get(rid)
            if r is None:audit.fail('UNKNOWN_RECORD',str(target),rid);continue
            if r.get('status')!='measured':audit.fail('NONMEASURED_LABEL_USED',str(target),rid)
            if modelpart.get(r.get('source_model_id'))!=part:audit.fail('SOURCE_MODEL_LEAKAGE',str(target),rid)
            if part in {'train','validation'} and r.get('platform_id')==target:audit.fail('HELDOUT_PLATFORM_LABEL_LEAKAGE',str(target),rid)
            if part=='test' and r.get('platform_id')!=target:audit.fail('TEST_PLATFORM_MISMATCH',str(target),rid)
        if part=='test':
            supported={records[x]['source_model_id'] for x in ids if x in records}
            required={m for m,p in modelpart.items() if p=='test'}
            if supported!=required:audit.fail('TEST_MODEL_COVERAGE',str(target),'All four held-out source models require measured test support')
        else:
            available={records[x]['platform_id'] for x in ids if x in records}
            if available!=set(PLATFORMS)-{target}:audit.fail('TRAINING_PLATFORM_COVERAGE',str(target),part+' requires the other four platforms')
    for key in ['normalization_fit_record_ids','feature_selection_fit_record_ids','training_performance_edge_record_ids']:
        if set(fold.get(key,[]))!=sets['train']:audit.fail('FIT_PROVENANCE_MISMATCH',str(target),key+' must equal declared train labels')
    if set(fold.get('hyperparameter_selection_record_ids',[]))!=sets['validation']:audit.fail('HYPERPARAMETER_LEAKAGE',str(target),'Validation labels only')
    if set(fold.get('feature_names',[]))-FEATURES:audit.fail('UNAPPROVED_FEATURE',str(target),'Latency, prediction, fitted target statistics and runtime outcome features are prohibited')
    if not fold.get('feature_names'):audit.fail('FEATURE_CONTRACT_MISSING',str(target),'Nonempty latency-free feature whitelist required')
    for edge in fold.get('training_similarity_edges',[]):
        if not isinstance(edge,dict) or modelpart.get(edge.get('source_model'))!='train' or modelpart.get(edge.get('target_model'))!='train':audit.fail('SIMILARITY_EDGE_LEAKAGE',str(target),str(edge))

def validate(bundle, stage='labels'):
    audit=Audit()
    if bundle.get('scope') not in {'historical_canonical5000','prospective_original_five_platforms'}:audit.fail('COHORT_SCOPE','scope','Original platforms only; five new software stacks do not qualify')
    if set(bundle.get('platform_ids',[]))!=set(PLATFORMS):audit.fail('PLATFORM_SUBSTITUTION','platform_ids','Must be A100 PCIe80GB, Ascend910B, MLU370-X8, GPU Max1550, Xeon8380')
    catalog_ref=bundle.get('source_catalog')
    if not isinstance(catalog_ref,dict) or catalog_ref.get('sha256')!=ORIGINAL_CATALOG_SHA256:audit.fail('CATALOG_SUBSTITUTION','source_catalog','Require the frozen original 34-model catalog bytes')
    models=audit.artifact(bundle.get('source_catalog'),'source_catalog')
    models=models if isinstance(models,list) else []
    modelids={m.get('model_id') for m in models}
    ops={op['op_id']:m['model_id'] for m in models for op in m.get('operators',[])}
    if len(modelids)!=34 or len(ops)!=6244:audit.fail('SOURCE_CATALOG_COUNTS','source_catalog','Expected 34 original models and 6244 original operator instances')
    split=bundle.get('source_model_split',{})
    if split!=expected_split(modelids):audit.fail('SOURCE_MODEL_SPLIT','source_model_split','Require released seed42 deterministic 27/3/4 split')
    modelpart={m:p for p,ids in split.items() for m in ids}
    data=bundle.get('records',[])
    if not isinstance(data,list):audit.fail('RECORDS_TYPE','records','Expected list');data=[]
    records={};opportunities=set();coverage={p:collections.Counter() for p in PLATFORMS}
    for r in data:
        if not isinstance(r,dict):audit.fail('RECORD_TYPE','records','Expected object');continue
        rid=r.get('record_id');pid=r.get('platform_id');oid=r.get('operator_instance_id');status=r.get('status')
        if not isinstance(rid,str) or not rid or rid in records:audit.fail('RECORD_ID','records',str(rid));continue
        records[rid]=r
        if pid not in PLATFORMS:audit.fail('PLATFORM_SUBSTITUTION',rid,str(pid))
        else:coverage[pid][status]+=1
        if oid not in ops or ops.get(oid)!=r.get('source_model_id'):audit.fail('OPERATOR_MODEL_MAPPING',rid,'Operator does not belong to declared source model')
        pair=(oid,pid)
        if pair in opportunities:audit.fail('DUPLICATE_OPPORTUNITY',rid,str(pair))
        opportunities.add(pair)
        if status not in STATUSES:audit.fail('UNKNOWN_STATUS',rid,str(status))
        if status=='measured':measured_record(r,audit)
        else:
            if not r.get('reason'):audit.fail('STATUS_REASON_MISSING',rid,str(status))
            if status in {'failed','unsupported','missing'} and r.get('latency_ms') is not None:audit.fail('NONOBSERVED_HAS_LABEL',rid,str(status))
            if status in {'calibrated','estimated'}:
                if not r.get('parent_record_ids'):audit.fail('DERIVATION_PARENTS_MISSING',rid,str(status))
                audit.artifact(r.get('derivation_evidence'),rid+'/derivation_evidence')
            if status in {'failed','unsupported'}:audit.artifact(r.get('status_evidence'),rid+'/status_evidence')
    for rid,r in records.items():
        for parent in r.get('parent_record_ids',[]):
            if parent not in records:audit.fail('UNKNOWN_PARENT',rid,parent)
        visiting=set();done=set()
        def visit(node):
            if node in visiting:audit.fail('PROVENANCE_CYCLE',rid,node);return
            if node in done:return
            visiting.add(node)
            for parent in records.get(node,{}).get('parent_record_ids',[]):visit(parent)
            visiting.remove(node);done.add(node)
        visit(rid)
    expected={(oid,pid) for oid in ops for pid in PLATFORMS}
    missing=expected-opportunities
    if missing:audit.fail('OPPORTUNITY_LEDGER_INCOMPLETE','records',str(len(missing))+' operator/platform pairs have no recorded status; this is not a measured failure count')
    canonical=bundle.get('canonical_record_ids',[])
    if bundle.get('scope')=='historical_canonical5000':
        if len(canonical)!=5000 or len(set(canonical))!=5000:audit.fail('CANONICAL5000_MISSING','canonical_record_ids','Exactly 5000 distinct original record IDs required')
        for rid in canonical:
            if rid not in records or records[rid].get('status') not in {'measured','calibrated','estimated'}:audit.fail('CANONICAL_LABEL_UNAVAILABLE','canonical_record_ids',rid)
        binding=audit.artifact(bundle.get('original_run_binding'),'original_run_binding')
        if not isinstance(binding,dict):audit.fail('ORIGINAL_RUN_BINDING_MISSING','original_run_binding','Frozen original graph, checkpoint, split and label ledger binding required')
        else:
            expected_bindings={'canonical_record_ids_sha256':canonical_sha(sorted(canonical)), 'source_catalog_sha256':ORIGINAL_CATALOG_SHA256, 'source_model_split_sha256':canonical_sha(split)}
            for k,v in expected_bindings.items():
                if binding.get(k)!=v:audit.fail('ORIGINAL_BINDING_MISMATCH','original_run_binding',k)
            for key in ['checkpoint_sha256','feature_transform_sha256','target_transform_sha256','graph_sha256']:
                value=binding.get(key,'')
                if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):audit.fail('ORIGINAL_BINDING_MISSING_DIGEST','original_run_binding',key)
    if not any(r.get('status')=='measured' for r in records.values()):audit.fail('NO_PHYSICAL_LABELS','records','No eligible physical labels available for LOPO')
    if stage=='folds':
        folds=bundle.get('folds',[])
        if len(folds)!=5 or {f.get('held_out_platform') for f in folds}!=set(PLATFORMS):audit.fail('FIVE_FOLDS_REQUIRED','folds','One fold per original platform')
        for fold in folds:validate_fold(fold,records,modelpart,audit)
    return {'pass':not audit.counts,'stage':stage,'error_counts':dict(audit.counts),'errors_first_150':audit.errors,'record_count':len(records),'expected_opportunities':len(expected),'unrecorded_opportunities':len(missing),'per_platform_recorded_status_counts':{k:dict(v) for k,v in coverage.items()},'evidence_boundary':'Pass validates submitted files, identifiers, arithmetic and declared split membership; it does not independently authenticate hardware or prove training code used the declared inputs.'}

def build_folds(bundle):
    split=bundle['source_model_split'];part={m:p for p,ms in split.items() for m in ms};folds=[]
    for target in PLATFORMS:
        f={'held_out_platform':target,'feature_names':sorted(FEATURES),'training_similarity_edges':[]}
        for p in ['train','validation','test']:
            f[p+'_record_ids']=sorted(r['record_id'] for r in bundle['records'] if r['status']=='measured' and part[r['source_model_id']]==p and ((r['platform_id']==target) if p=='test' else (r['platform_id']!=target)))
        for k in ['normalization_fit_record_ids','feature_selection_fit_record_ids','training_performance_edge_record_ids']:f[k]=f['train_record_ids'][:]
        f['hyperparameter_selection_record_ids']=f['validation_record_ids'][:]
        folds.append(f)
    return folds

def main():
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('--stage',choices=['labels','folds'],default='labels');p.add_argument('--report',type=Path,required=True);p.add_argument('--build-folds',type=Path);args=p.parse_args()
    bundle=json.loads(args.bundle.read_text());result=validate(bundle,args.stage);result.update({'bundle_path':args.bundle.name,'bundle_sha256':sha256(args.bundle),'validator_sha256':sha256(__file__)})
    if args.build_folds:
        if result['pass']:
            bundle['folds']=build_folds(bundle);post=validate(bundle,'folds')
            result['prepared_folds_validation']=post
            if post['pass']:args.build_folds.write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n')
            else:result['pass']=False
        else:result['folds_created']=False
    args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['pass','record_count','error_counts']},ensure_ascii=False));raise SystemExit(0 if result['pass'] else 2)

if __name__=='__main__':main()
