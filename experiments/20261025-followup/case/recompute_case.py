"""Independent stdlib-only retrospective audit. No torch or hardware calls."""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, itertools, json, math, statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
T95 = 2.7764451051977987
MASKS = [m for m in range(16) if bin(m).count('1') <= 2]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def close(a, b): assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9), (a, b)
def stats(v):
    n=len(v); mean=statistics.mean(v); sd=statistics.stdev(v) if n>1 else 0
    e=T95*sd/math.sqrt(n) if n==5 else None
    return {'n_processes':n,'values':v,'mean':mean,'sample_sd':sd,
            'descriptive_t95_df4':[mean-e,mean+e] if e is not None else None}
def signflip(v):
    observed=abs(statistics.mean(v))
    return sum(abs(statistics.mean(s*x for s,x in zip(signs,v))) >= observed-1e-12
               for signs in itertools.product([-1,1],repeat=len(v)))/(2**len(v))
def phi(latency):
    out=[]
    for i in range(4):
        s=0.
        for m in range(16):
            if m&(1<<i): continue
            k=bin(m).count('1'); w=math.factorial(k)*math.factorial(3-k)/24
            s+=w*(latency[m]-latency[m|(1<<i)])
        out.append(s)
    close(sum(out),latency[0]-latency[15]); return out
def top(scores, positive=True):
    return sum(1<<i for i in sorted(range(4),key=lambda i:(-scores[i],i))[:2]
               if not positive or scores[i]>0)
def path_game(game, aggregation='critical_path'):
    nodes=game['base_costs_ms']; pred={k:[] for k in nodes}
    for a,b in game['edges']:
        if a not in pred[b]: pred[b].append(a)
    order=[]; remaining=set(nodes)
    while remaining:
        ready=sorted(n for n in remaining if all(p in order for p in pred[n]))
        assert ready,'cyclic graph'; order.extend(ready);remaining.difference_update(ready)
    result={}
    for m in range(16):
        costs=dict(nodes)
        for i,a in enumerate(game['action_ids']):
            if m&(1<<i):costs.update(game['action_cost_updates_ms'][a])
        end={}
        for n in order:end[n]=costs[n]+max((end[p] for p in pred[n]),default=0)
        overhead=game.get('runtime_overhead_ms',0.)
        if isinstance(overhead,dict): overhead=overhead[str(m)]
        result[m]=(sum(costs.values()) if aggregation=='sum' else max(end.values()))+overhead
    return result
def quadratic(base, calibration):
    residual={m:calibration[m]-base[m] for m in MASKS}
    intercept=residual[0]
    one={i:residual[1<<i]-intercept for i in range(4)}
    two={(i,j):residual[(1<<i)|(1<<j)]-intercept-one[i]-one[j]
         for i in range(4) for j in range(i+1,4)}
    return {m:base[m]+intercept+sum(v for i,v in one.items() if m&(1<<i))+
            sum(v for (i,j),v in two.items() if m&(1<<i) and m&(1<<j)) for m in range(16)}
def quality(row):
    ids=row['sample_ids']; labels=row['labels']; preds=row['predictions']
    assert len(ids)==len(set(ids))==len(labels)==len(preds)==100
    correct=[a==b for a,b in zip(labels,preds)]
    if 'correct' in row:assert correct==row['correct']
    q=sum(correct)/len(correct)
    close(q,row['quality_score']); close(sum(correct),row['correct_count'])
    return q,dict(zip(ids,zip(labels,preds)))
def validate_raw(raw):
    assert {r['mask'] for r in raw['coalitions']}==set(range(16))
    rows={r['mask']:r for r in raw['coalitions']}
    baseq,basemap=quality(rows[0]); lat={}; feasible={}
    for m,r in rows.items():
        q,qmap=quality(r); assert qmap.keys()==basemap.keys()
        assert all(qmap[i][0]==basemap[i][0] for i in qmap)
        assert r['status']=='ok'; samples=r['latency_samples_ms']
        assert len(samples)==30 and all(x>0 and math.isfinite(x) for x in samples)
        lat[m]=statistics.mean(samples); close(lat[m],r['mean_ms'])
        close(statistics.stdev(samples),r['sample_sd_ms'])
        close(q/baseq,r['retention_ratio'])
        feasible[m]=r['equivalence']['passed'] and q/baseq>=.99
    return lat,feasible,basemap
def contrasts(plans, primary):
    result={}
    for name in primary:
        for control in ['latency','uniform_at_most_2','uniform_exact_2','calibration_exhaustive']:
            if name not in plans or control not in plans:continue
            values=[a-b for a,b in zip(plans[name]['raw_gain_percent']['values'],plans[control]['raw_gain_percent']['values'])]
            result[name+' minus '+control]={**stats(values),'two_sided_exact_signflip_p':signflip(values)}
    return result
def main():
    frozen=read(HERE/'offline-analysis-plan.json')
    for row in frozen['inputs']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    rawpaths=sorted((ROOT/'v2-实证增强/evidence/fullmodel_raw').glob('*.json'))
    groups=defaultdict(list); identities={}; raw_count=coal_count=call_count=0
    for p in rawpaths:
        raw=read(p)
        if raw.get('schema_version')!='het.fullmodel.run.v2':continue
        lat,feasible,qmap=validate_raw(raw)
        groups[(raw['device_id'],raw['model_id'],raw['input_shape'][0])].append((p,raw,lat,feasible,qmap))
        identities[p.name]=sha(p); raw_count+=1;coal_count+=len(lat);call_count+=sum(len(c['latency_samples_ms']) for c in raw['coalitions'])
    tasks={}
    for p in sorted((ROOT/'v2-实证增强/evidence/final-tasks').glob('*.json')):
        d=read(p)
        if isinstance(d,dict) and d.get('schema_version')=='het.measured.coalitions.v2':
            tasks[(d['device_id'],d['model_id'],d['input_shape'][0])]=(p,d)
    assert len(groups)==len(tasks)==16 and raw_count==128
    contexts=[]; frozen_masks={}
    for key, entries in sorted(groups.items()):
        entries.sort(key=lambda e:e[1]['run_index']); assert [x[1]['run_index'] for x in entries]==list(range(1,9))
        assert len({e[1]['runtime']['pid'] for e in entries})==8
        p,task=tasks[key]
        for src in task['source_artifacts']:assert identities[Path(src['path']).name]==src['sha256']
        train=entries[:3]; evaluation=entries[3:]
        for a in ['weights_state_sha256','dataset_manifest_sha256']:
            assert len({e[1][a] for e in entries})==1
        assert set(train[0][4]).isdisjoint(evaluation[0][4])
        assert all(set(e[4])==set(train[0][4]) for e in train)
        assert all(set(e[4])==set(evaluation[0][4]) for e in evaluation)
        calibration={m:statistics.mean(e[2][m] for e in train) for m in range(16)}
        base=path_game(task['copa_game']); calibrated=quadratic(base,calibration)
        assert all(v>0 and math.isfinite(v) for v in base.values())
        assert all(v>0 and math.isfinite(v) for v in calibrated.values())
        scores={'copa_dag_uncalibrated':phi(base),'copa_dag_quadratic_calibrated':phi(calibrated),
                'measured_calibration_shapley':phi(calibration)}
        md={r['id']:r for r in task['action_metadata']}
        masks={name:top(values) for name,values in scores.items()}
        masks['latency']=top([md[a]['operator_latency_ms'] for a in task['action_ids']],False)
        masks['local_headroom']=top([md[a]['local_benefit_ms'] for a in task['action_ids']])
        masks['calibration_exhaustive']=min((calibration[m],m) for m in MASKS
            if all(e[3][m] for e in train))[1]
        # Existing full PPO matrix: retain all configurations and seeds; no retraining/selection.
        archive_result=ROOT/'v2-实证增强/methods/planner-matrix-results/all-complete-v1/results'/(
            ('npu' if key[0].startswith('npu') else 'cuda')+'-'+task['task_id']+'.json')
        ppo={}
        if key[0].startswith(('npu','cuda')) and archive_result.exists():
            ar=read(archive_result); assert ar['status']=='completed'
            close(ar['fixed_strategies']['copa_dag']['mask'],masks['copa_dag_uncalibrated'])
            close(ar['fixed_strategies']['dag_quadratic_runtime']['mask'],masks['copa_dag_quadratic_calibrated'])
            for item in ar['ppo']:ppo.setdefault(item['configuration'],[]).append(item['plan']['mask'])
            assert len(ppo)==4 and all(len(v)==5 for v in ppo.values())
        plans={}
        gains=[{m:100*(e[2][0]-e[2][m])/e[2][0] for m in range(16)} for e in evaluation]
        for name,mask in masks.items():
            values=[g[mask] for g in gains]
            plans[name]={'mask':mask,'raw_gain_percent':stats(values),
                         'gate_gain_percent':stats([v if e[3][mask] and v>=0 else 0 for e,v in zip(evaluation,values)])}
        for name, subset in [('uniform_at_most_2',MASKS),('uniform_exact_2',[m for m in MASKS if bin(m).count('1')==2])]:
            plans[name]={'masks':subset,'raw_gain_percent':stats([statistics.mean(g[m] for m in subset) for g in gains])}
        for name in scores:
            subset=[m for m in MASKS if bin(m).count('1')==bin(masks[name]).count('1')]
            plans[name+'_cardinality_matched_random']={'masks':subset,'raw_gain_percent':stats([statistics.mean(g[m] for m in subset) for g in gains])}
        plans['heldout_oracle']={'posthoc':True,'raw_gain_percent':stats([max(g[m] for m in MASKS if e[3][m]) for e,g in zip(evaluation,gains)])}
        for name, seeds in ppo.items():
            plans[name]={'masks_by_seed':seeds,'raw_gain_percent':stats([statistics.mean(g[m] for m in seeds) for g in gains]),
                         'uncertainty_unit':'five physical processes after averaging all five policy seeds'}
        errors={name:{'all16_coalition_mre_percent':statistics.mean(100*abs(pred[m]-e[2][m])/e[2][m] for e in evaluation for m in range(16)),
                      'physical_shapley_mae_ms':statistics.mean(abs(a-b) for e in evaluation for a,b in zip(phi(e[2]),phi(pred)))}
                for name,pred in [('uncalibrated',base),('quadratic_calibrated',calibrated)]}
        con=contrasts(plans,['copa_dag_uncalibrated','copa_dag_quadratic_calibrated','ppo_with_heldout_embedding'])
        for name in scores:
            control=name+'_cardinality_matched_random'
            v=[a-b for a,b in zip(plans[name]['raw_gain_percent']['values'],plans[control]['raw_gain_percent']['values'])]
            con[name+' minus cardinality_matched_random']={**stats(v),'two_sided_exact_signflip_p':signflip(v)}
        contexts.append({'device':key[0],'model':key[1],'batch':key[2],'task_sha256':sha(p),'plans':plans,
                         'paired_contrasts_pp':con,'physical_error_audit':errors,
                         'baseline_eval_ms':stats([e[2][0] for e in evaluation]),
                         'evaluation_quality':quality(evaluation[0][1]['coalitions'][0])[0]})
        frozen_masks[key]=masks
    # Fresh NPU AB/BA: all 20 files, all eleven masks, all 4400 pairs.
    paired_groups=defaultdict(list); paired_count=pairs_count=0; paired_identities={}
    for p in sorted((ROOT/'v2-实证增强/evidence/paired_raw').glob('*.json')):
        raw=read(p)
        if raw.get('schema_version')!='het.fullmodel.paired-confirmation.v1':continue
        assert raw['status']=='completed'; assert {r['mask'] for r in raw['coalitions']}==set(MASKS)
        rows={}; source_eval=groups[(raw['device_id'],raw['model_id'],raw['input_shape'][0])][3][1]
        assert raw['weights_state_sha256']==source_eval['weights_state_sha256']
        assert raw['input_tensor_sha256']==source_eval['input_tensor_sha256']
        a100_key=next(k for k in groups if k[0].startswith('cuda') and k[1:]==(raw['model_id'],raw['input_shape'][0]))
        a100_source=groups[a100_key][3][1]
        assert raw['weights_state_sha256']==a100_source['weights_state_sha256']
        assert raw['input_tensor_sha256']==a100_source['input_tensor_sha256']
        source_mask0=next(r for r in a100_source['coalitions'] if r['mask']==0)
        source_q,source_map=quality(source_mask0)
        for row in raw['coalitions']:
            assert row['status']=='ok' and len(row['pairs'])==20
            assert sum(x['order']=='AB' for x in row['pairs'])==10
            assert sum(x['order']=='BA' for x in row['pairs'])==10
            a=[];b=[]
            for pair in row['pairs']:
                assert pair['status']=='ok' and pair['output_equivalent']
                x=pair['A']['latency_ms'];y=pair['B']['latency_ms'];assert x>0 and y>0
                close(pair['baseline_ms'],x);close(pair['candidate_ms'],y);close(pair['paired_delta_ms'],x-y)
                a.append(x);b.append(y)
            gain=100*(statistics.mean(a)-statistics.mean(b))/statistics.mean(a)
            close(gain,row['summary']['ratio_of_means_gain_percent'])
            aq,am=quality(row['baseline_quality']);bq,bm=quality(row['candidate_quality'])
            assert am.keys()==bm.keys() and all(am[i][0]==bm[i][0] for i in am)
            assert bm.keys()==source_map.keys() and all(bm[i][0]==source_map[i][0] for i in bm)
            feasible=row['candidate_valid'] and bq/aq>=.99
            assert feasible==row['final_candidate_feasible']
            rows[row['mask']]={'raw_gain':gain,'feasible':feasible,'retention':bq/aq,
                'baseline_ms':statistics.mean(a),'candidate_ms':statistics.mean(b),'source_retention':bq/source_q,
                'AB_gain':100*(statistics.mean(x['A']['latency_ms'] for x in row['pairs'] if x['order']=='AB')-statistics.mean(x['B']['latency_ms'] for x in row['pairs'] if x['order']=='AB'))/statistics.mean(x['A']['latency_ms'] for x in row['pairs'] if x['order']=='AB')}
            pairs_count+=20
        paired_groups[(raw['device_id'],raw['model_id'],raw['input_shape'][0])].append((p,raw,rows))
        paired_count+=1;paired_identities[p.name]=sha(p)
    assert paired_count==20 and pairs_count==4400
    fresh=[]
    trace_manifest=read(ROOT/'v2-实证增强/methods/paired-method-results-with-ppo.json')['traces']
    traces=[]
    for item in trace_manifest:
        tp=ROOT/'v2-实证增强/methods'/item['path']
        assert sha(tp)==item['sha256']
        traces.append(read(tp))
    for key, entries in sorted(paired_groups.items()):
        entries.sort(key=lambda e:e[1]['run_index']); assert [e[1]['run_index'] for e in entries]==list(range(101,106))
        assert len({e[1]['runtime']['pid'] for e in entries})==5
        masks=dict(frozen_masks[key]); trace=None
        for d in traces:
            c=d.get('context',{})
            if c.get('model_id')==key[1] and c.get('input_shape',[None])[0]==key[2]:trace=d;break
        assert trace is not None,key
        masks['cross_hardware_dag']=trace['action_priority']['selected_mask']
        for n,v in trace['matched_control_plans'].items():masks[n]=v['selected_mask']
        plans={}
        for name,mask in masks.items():
            v=[e[2][mask]['raw_gain'] if mask else 0 for e in entries]
            plans[name]={'mask':mask,'raw_gain_percent':stats(v),
                'gate_gain_percent':stats([x if e[2][mask]['feasible'] and x>=0 else 0 for e,x in zip(entries,v)])}
        for name,subset in [('uniform_at_most_2',MASKS),('uniform_exact_2',[m for m in MASKS if bin(m).count('1')==2])]:
            plans[name]={'masks':subset,'raw_gain_percent':stats([statistics.mean(e[2][m]['raw_gain'] if m else 0 for m in subset) for e in entries])}
        for name in ['copa_dag_uncalibrated','copa_dag_quadratic_calibrated','cross_hardware_dag']:
            subset=[m for m in MASKS if bin(m).count('1')==bin(masks[name]).count('1')]
            plans[name+'_cardinality_matched_random']={'masks':subset,'raw_gain_percent':stats([statistics.mean(e[2][m]['raw_gain'] if m else 0 for m in subset) for e in entries])}
        original=next(c for c in contexts if (c['device'],c['model'],c['batch'])==key)
        for name,plan in original['plans'].items():
            if 'masks_by_seed' not in plan:continue
            seeds=plan['masks_by_seed'];plans[name]={'masks_by_seed':seeds,'raw_gain_percent':stats([statistics.mean(e[2][m]['raw_gain'] if m else 0 for m in seeds) for e in entries])}
        plans['heldout_oracle']={'posthoc':True,'raw_gain_percent':stats([max(e[2][m]['raw_gain'] if m else 0 for m in MASKS if e[2][m]['feasible']) for e in entries])}
        c=contrasts(plans,['copa_dag_uncalibrated','copa_dag_quadratic_calibrated','cross_hardware_dag','ppo_with_heldout_embedding'])
        # Cross-hardware latency control is distinct from target-side fixed latency when needed.
        for control in ['target_operator_latency','uniform_at_most_2']:
            v=[a-b for a,b in zip(plans['cross_hardware_dag']['raw_gain_percent']['values'],plans[control]['raw_gain_percent']['values'])]
            c['cross_hardware_dag minus '+control]={**stats(v),'two_sided_exact_signflip_p':signflip(v)}
        fresh.append({'device':key[0],'model':key[1],'batch':key[2],'plans':plans,'paired_contrasts_pp':c,
                      'mask0_negative_control_raw_gain_percent':stats([e[2][0]['raw_gain'] for e in entries]),
                      'minimum_quality_retention':min(e[2][m]['retention'] for e in entries for m in MASKS),
                      'minimum_candidate_to_a100_source_quality_retention':min(e[2][m]['source_retention'] for e in entries for m in MASKS)})
    # Content-check release archives without extraction; backups do not add samples.
    import zipfile
    archive_checks=[]
    for rel,expected in [('public-repo/experiments/20261025/archives/full-model-data-and-frozen-tasks.zip',identities),
                         ('public-repo/experiments/20261025/archives/paired-confirmation-data.zip',paired_identities)]:
        p=ROOT/rel; found={}
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                short=Path(name).name
                if short in expected:
                    digest=hashlib.sha256(z.read(name)).hexdigest()
                    assert digest==expected[short],name
                    found[short]=digest
        assert set(found)==set(expected),(rel,len(found),len(expected))
        archive_checks.append({'path':rel,'sha256':sha(p),'matching_unique_raw_files':len(found)})
    result={'role':'Independent AI retrospective audit','plan_sha256':sha(HERE/'offline-analysis-plan.json'),
        'code_sha256':sha(Path(__file__)),'new_hardware_executions':0,'counts':{'fullmodel_blocks':raw_count,
        'fullmodel_contexts':16,'fullmodel_coalitions':coal_count,'fullmodel_timed_calls':call_count,
        'paired_blocks':paired_count,'paired_coalitions':220,'paired_AB_BA_pairs':pairs_count},
        'archive_checks':archive_checks,'original_all16_contexts':contexts,'fresh_npu_all4_contexts':fresh,
        'scope':'All canonical intervention raw traces, no trimming, all original runs and negative effects. Fixed methods fitted only on original calibration. Existing results were known before this retrospective audit. Physical process is uncertainty unit; t95 intervals are conditional descriptive, not familywise/population superiority evidence.'}
    (HERE/'all-context-recomputation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for label,cs in [('original16',contexts),('fresh_npu4',fresh)]:
        with (HERE/(label+'-all-methods.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['device','model','batch','method','mask','gain_percent','sd','t95_low','t95_high']);writer.writeheader()
            for c in cs:
                for name,p in c['plans'].items():
                    s=p['raw_gain_percent']; ci=s['descriptive_t95_df4']; writer.writerow({'device':c['device'],'model':c['model'],'batch':c['batch'],'method':name,'mask':p.get('mask',p.get('masks_by_seed','expectation/oracle')),'gain_percent':s['mean'],'sd':s['sample_sd'],'t95_low':ci[0],'t95_high':ci[1]})
    print(json.dumps({'counts':result['counts'],'archives_verified':len(archive_checks),'output':str(HERE/'all-context-recomputation.json')}))
if __name__=='__main__':main()
