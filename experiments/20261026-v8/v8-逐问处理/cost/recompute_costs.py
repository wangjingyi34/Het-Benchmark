#!/usr/bin/env python3
"""V8 economics: deterministic, stdlib-only recomputation from frozen local inputs.
No network, no hardware execution, no orders. All writes remain alongside this script.
"""
from pathlib import Path
from statistics import mean, stdev, variance
from math import sqrt, ceil, isclose
import csv, hashlib, itertools, json

H = Path(__file__).resolve().parent
S = H / 'sources'
load = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
P = load(H / 'pricing-inputs.json')
FX = P['fx']['cny_per_usd']
T = P['statistical_protocol']['t_critical_df4']
CONTEXTS = list(itertools.product(['resnet18', 'resnet50'], [1, 4]))
OUT = {}

def js(name, value):
    (H/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    OUT[name] = {'sha256':sha(H/name)}

def table(name, rows):
    assert rows
    with (H/name).open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    OUT[name] = {'rows':len(rows),'sha256':sha(H/name)}

def ci(v):
    m=mean(v); d=T*stdev(v)/sqrt(len(v)); return m,m-d,m+d

def ratio_ci(x, y):
    # Independent hardware processes: same run index is not evidence of covariance.
    # Fieller normal-means approximation using conservative df=4 t critical value.
    n=len(x);m=len(y); xm=mean(x);ym=mean(y);q=T*T
    a=xm*xm-q*variance(x)/n;b=-2*xm*ym;c=ym*ym-q*variance(y)/m
    d=b*b-4*a*c
    assert a>0 and d>=0
    return ym/xm,(-b-sqrt(d))/(2*a),(-b+sqrt(d))/(2*a)

def pctl(values, q):
    values=sorted(values);return values[min(len(values)-1,ceil(q*len(values))-1)]

def replay(ms,batch,demand,replicas):
    # 200 deterministic complete batches. All individual requests are retained.
    # Queue and batch formation are computed; network/startup not represented.
    ends=[0.0]*replicas; rt=[]; service=ms/1000
    for k in range(200):
        ready=(k*batch+batch-1)/demand; r=k%replicas
        finish=max(ready,ends[r])+service;ends[r]=finish
        rt.extend((finish-(k*batch+j)/demand)*1000 for j in range(batch))
    return max(rt),pctl(rt,.95)

# Verify original and web snapshot manifests before any analysis.
for r in load(H/'input-snapshot-manifest.json'):
    assert sha(H/r['snapshot_path'])==r['sha256']
for r in load(S/'fetch-manifest.json'):
    assert sha(S/r['file'])==r['sha256']
RAW={}; SERIES={}; process=[]; timing=[]
for model,batch in CONTEXTS:
    for dev in ['cuda','npu']:
        vals=[]
        for run in range(4,9):
            path=S/'measurements'/f'{dev}-{model}-b{batch}-r{run}.json'
            d=load(path);c=next(c for c in d['coalitions'] if c['mask']==0)
            assert d['provenance']==c['provenance']=='physical_measurement'
            assert d['quality_partition']=='evaluation' and c['status']=='ok'
            assert d['protocol']['dtype']=='float32' and d['protocol']['repeats']==30
            assert len(c['latency_samples_ms'])==30
            assert isclose(mean(c['latency_samples_ms']),c['mean_ms'],rel_tol=1e-13)
            assert len(c['labels'])==len(c['predictions'])==len(c['sample_ids'])==100
            assert sum(a==b for a,b in zip(c['labels'],c['predictions']))==c['correct_count']
            assert isclose(c['quality_score'],c['correct_count']/100)
            RAW[dev,model,batch,run]=(d,c);vals.append(c['mean_ms'])
            process.append({'model':model,'batch':batch,'device':dev,'run':run,'mean_ms':c['mean_ms'],'timed_calls':30,'correct_count':c['correct_count'],'quality_sample_count':100,'top1':c['quality_score'],'provenance':'physical_measurement','source_path':str(path.relative_to(H)),'source_sha256':sha(path)})
        SERIES[dev,model,batch]=vals
        m,lo,hi=ci(vals)
        timing.append({'model':model,'batch':batch,'device':dev,'physical_processes':5,'mean_ms':m,'sd_between_process_ms':stdev(vals),'ci95_lower_ms':lo,'ci95_upper_ms':hi,'mean_images_per_second':batch*1000/m,'top1':RAW[dev,model,batch,4][1]['quality_score'],'unique_quality_images':100,'measured_service_sla':False})
    for run in range(4,9):
        da,ca=RAW['cuda',model,batch,run];dn,cn=RAW['npu',model,batch,run]
        for key in ['weights_state_sha256','dataset_manifest_sha256','input_tensor_sha256','input_shape','model_configuration']:
            assert da[key]==dn[key]
        assert ca['sample_ids']==cn['sample_ids'] and ca['labels']==cn['labels']
        assert ca['quality_score']==cn['quality_score']
assert len(process)==40 and len(timing)==8
table('measured-processes.csv',process);table('timing-summary.csv',timing)

# Pure active-compute allocation, independent of service idle time and billing rounding.
def costrow(model,batch,a,n,plan,fx,alpha,hours):
    av=SERIES['cuda',model,batch];nv=SERIES['npu',model,batch]
    am,al,ah=ci(av);nm,nl,nh=ci(nv);r,rl,rh=ratio_ci(av,nv)
    ar=a['usd_per_card_hour']*fx*alpha
    nr=n['hourly_cny'] if plan=='hourly' else n['monthly_cny']/hours
    mult=nr/ar
    return {'model':model,'batch':batch,'a100_provider':a['id'],'npu_tariff':n['id'],'npu_plan':plan,'fx_cny_per_usd':fx,'mig_cost_allocation':alpha,'month_hours':hours,'a100_allocated_cny_per_hour':ar,'npu_effective_cny_per_hour':nr,'target_source_price_ratio':mult,'target_source_latency_ratio':r,'latency_ratio_ci95_lower':rl,'latency_ratio_ci95_upper':rh,'a100_cny_per_image':ar*am/(3.6e6*batch),'npu_cny_per_image':nr*nm/(3.6e6*batch),'a100_cny_per_1000_images':ar*am/(3600*batch),'a100_per_1000_ci95_lower':ar*al/(3600*batch),'a100_per_1000_ci95_upper':ar*ah/(3600*batch),'npu_cny_per_1000_images':nr*nm/(3600*batch),'npu_per_1000_ci95_lower':nr*nl/(3600*batch),'npu_per_1000_ci95_upper':nr*nh/(3600*batch),'npu_cost_saving_percent':100*(1-mult*r),'saving_ci95_lower_percent':100*(1-mult*rh),'saving_ci95_upper_percent':100*(1-mult*rl),'achieves_66_percent':mult*r<=.34,'cost_scope':'active_compute_allocation_before_unquantified_extras','measured_market_mig_rate':False}
primary=[];sensitivity=[]
for model,batch in CONTEXTS:
    for a,n,plan in itertools.product(P['a100_rates'],P['npu_tariffs'],['hourly','monthly']):
        primary.append(costrow(model,batch,a,n,plan,FX,1,720))
        for fx,alpha in itertools.product(P['comparison_policy']['fx_sensitivity'],P['comparison_policy']['mig_allocation_alpha']):
            for hours in ([720] if plan=='hourly' else [672,720,744]):
                sensitivity.append(costrow(model,batch,a,n,plan,fx,alpha,hours))
assert len(primary)==32 and len(sensitivity)==768
table('unit-cost-primary.csv',primary);table('allocation-sensitivity.csv',sensitivity)

breakeven=[]
for model,batch in CONTEXTS:
    av=SERIES['cuda',model,batch];nv=SERIES['npu',model,batch];r,rl,rh=ratio_ci(av,nv)
    for n,plan,alpha in itertools.product(P['npu_tariffs'],['hourly','monthly'],P['comparison_policy']['mig_allocation_alpha']):
        nr=n['hourly_cny'] if plan=='hourly' else n['monthly_cny']/720
        breakeven.append({'model':model,'batch':batch,'npu_tariff':n['id'],'npu_plan':plan,'npu_effective_cny_per_hour':nr,'mig_allocation':alpha,'a100_full_card_usd_hour_break_even':nr*r/(alpha*FX),'a100_full_card_usd_hour_for_66_saving':nr*r/(.34*alpha*FX),'threshold_66_ci95_lower':nr*rl/(.34*alpha*FX),'threshold_66_ci95_upper':nr*rh/(.34*alpha*FX),'maximum_npu_cny_hour_for_66_runpod':.34*alpha*1.79*FX/r,'maximum_npu_cny_hour_for_66_lambda':.34*alpha*2.79*FX/r,'formula':'P_N*t_N/(alpha*P_A*t_A)<=0.34; P_A in CNY/hour; equal batch','tax_adjustment':'multiply cost ratio by (1+tau_N)/(1+tau_A) when quote tax bases are verified'})
table('break-even.csv',breakeven)

# Service procurement: whole orders, billing quantum, uptime, equal demand and quality.
service=[];cfg=P['service_model']
for model,batch in CONTEXTS:
 for demand,budget,endpoint,packing in itertools.product(cfg['demands_per_second'],cfg['response_budgets_ms'],cfg['latency_endpoints'],P['comparison_policy']['packing_sensitivity']):
    av=ci(SERIES['cuda',model,batch]);nv=ci(SERIES['npu',model,batch]);idx=0 if endpoint=='mean' else 2
    at,nt=av[idx],nv[idx]
    repa=ceil(demand*at/(1000*batch*cfg['utilization_cap']));repn=ceil(demand*nt/(1000*batch*cfg['utilization_cap']))
    amax,ap95=replay(at,batch,demand,repa);nmax,np95=replay(nt,batch,demand,repn)
    feasible=max(amax,nmax)<=budget+1e-8
    for a,n,plan,active,policy in itertools.product(P['a100_rates'],P['npu_tariffs'],['hourly','monthly'],cfg['active_hours_per_day'],cfg['billing_policies']):
        acards=ceil(ceil(repa/packing)/a['minimum_cards_per_order'])*a['minimum_cards_per_order'];ncards=repn
        for quantum in (cfg['npu_billing_quanta_seconds_assumed'] if plan=='hourly' else [0]):
            ah=720 if policy=='keep_warm' else 30*ceil(active*3600/a['billing_quantum_seconds']-1e-12)*a['billing_quantum_seconds']/3600
            if plan=='hourly':
                nh=720 if policy=='keep_warm' else 30*ceil(active*3600/quantum-1e-12)*quantum/3600
                nc=ncards*nh*n['hourly_cny']
            else: nh=720;nc=ncards*n['monthly_cny']
            ac=acards*ah*a['usd_per_card_hour']*FX
            requests=demand*active*3600*30
            saving=100*(1-nc/ac)
            service.append({'model':model,'batch':batch,'demand_images_per_second':demand,'response_budget_ms':budget,'latency_endpoint':endpoint,'assumed_mig_slices_per_a100_card':packing,'a100_provider':a['id'],'npu_tariff':n['id'],'npu_plan':plan,'active_hours_per_day':active,'billing_policy':policy,'npu_billing_quantum_seconds_assumed':quantum,'a100_replicas':repa,'npu_replicas':repn,'a100_billed_cards':acards,'npu_billed_cards':ncards,'a100_unused_slice_slots':acards*packing-repa,'a100_billed_hours_per_card':ah,'npu_billed_hours_per_card':nh,'a100_max_modelled_response_ms':amax,'npu_max_modelled_response_ms':nmax,'a100_modelled_p95_ms':ap95,'npu_modelled_p95_ms':np95,'equal_demand_response_model_feasible':feasible,'month_requests':requests,'a100_cny_per_month':ac,'npu_cny_per_month':nc,'a100_service_cny_per_1000_images':ac*1000/requests,'npu_service_cny_per_1000_images':nc*1000/requests,'npu_cost_saving_percent':saving,'achieves_66_within_model':feasible and saving>=66,'measured_service_sla':False,'packing_performance_measured':False,'feasibility_note':'conditional periodic-arrival model; excludes startup,network,random-arrival tails and inter-slice interference'})
table('service-procurement.csv',service)

# Cost the traceable public migration: requested mask6 vs baseline and offline gate.
with (S/'prior/migration-heldout-process.csv').open() as f: mr=list(csv.DictReader(f))
ma=[];mb=[];mc=[];mg=[];migration=[]
for row in mr:
    run=int(row['run_index']);da,ca=RAW['cuda','resnet18',1,run];dn,cn=RAW['npu','resnet18',1,run]
    cand=next(c for c in dn['coalitions'] if c['mask']==6)
    assert sha(S/'measurements'/f'cuda-resnet18-b1-r{run}.json')==row['source_raw_sha256']
    assert sha(S/'measurements'/f'npu-resnet18-b1-r{run}.json')==row['target_raw_sha256']
    assert isclose(cand['mean_ms'],float(row['requested_target_candidate_ms']))
    accept=row['accepted']=='True'; gate=cand['mean_ms'] if accept else cn['mean_ms']
    ma.append(ca['mean_ms']);mb.append(cn['mean_ms']);mc.append(cand['mean_ms']);mg.append(gate)
    for a,n,plan in itertools.product(P['a100_rates'],P['npu_tariffs'],['hourly','monthly']):
        ar=a['usd_per_card_hour']*FX;nr=n['hourly_cny'] if plan=='hourly' else n['monthly_cny']/720
        migration.append({'run':run,'a100_provider':a['id'],'npu_tariff':n['id'],'npu_plan':plan,'source_baseline_ms':ca['mean_ms'],'target_baseline_ms':cn['mean_ms'],'requested_target_mask':6,'requested_target_ms':cand['mean_ms'],'gate_accepted':accept,'gate_replayed_ms':gate,'source_cny_per_1000':ar*ca['mean_ms']/3600,'target_baseline_cny_per_1000':nr*cn['mean_ms']/3600,'requested_target_cny_per_1000':nr*cand['mean_ms']/3600,'gated_target_cny_per_1000':nr*gate/3600,'requested_within_target_cost_saving_percent':100*(1-cand['mean_ms']/cn['mean_ms']),'gated_within_target_cost_saving_percent':100*(1-gate/cn['mean_ms']),'gated_cross_device_cost_saving_percent':100*(1-nr*gate/(ar*ca['mean_ms'])),'quality_top1':cand['quality_score'],'quality_retention_vs_source':cand['quality_score']/ca['quality_score'],'execution_scope':'offline selection and gating on existing physical timing ledger; gate decision latency excluded'})
table('migration-costs.csv',migration)

fb=load(S/'fallback/npu-active-posthoc.json');fa=load(S/'fallback/active-fallback-npu.json')
wall=fb['wall_cost_seconds'];nwall=wall['decision_phase_wall_seconds']+wall['measurement_phase_wall_seconds_including_graph_setup']
assert fa['successful_measurements']==fb['successful_fresh_measurements']==83
assert fa['decision_phase_wall_seconds']==wall['decision_phase_wall_seconds']
assert fa['measurement_phase_wall_seconds_including_graph_setup']==wall['measurement_phase_wall_seconds_including_graph_setup']
fallback=[]
for n in P['npu_tariffs']:
 for plan in ['hourly','monthly']:
    rate=n['hourly_cny'] if plan=='hourly' else n['monthly_cny']/720
    for q in ([1,3600] if plan=='hourly' else [0]):
        minimum=rate*ceil(nwall/q)*q/3600 if q else n['monthly_cny']
        fallback.append({'operation':'910B frozen prediction plus active profiling','scope':'732 queries; 83 Activation queries remeasured','physical_wall_seconds':nwall,'rate_provider':n['id'],'plan':plan,'billing_quantum_seconds_assumed':q,'whole_order_cards':1,'allocated_wall_cost_cny':rate*nwall/3600,'minimum_new_order_charge_cny_excluding_startup':minimum,'money_status':'scenario from frozen time and listed price; not paid invoice','source':'sources/fallback/npu-active-posthoc.json'})
for dev,f in [('A100 MIG','a100mig-reference.json'),('M5 MPS','m5mps-reference.json')]:
    d=load(S/'fallback'/f);assert len(d['rows'])==24 and not d['failures']
    if dev=='A100 MIG':
        for a in P['a100_rates']:
            rate=a['usd_per_card_hour']*FX;q=a['billing_quantum_seconds'];minimum=rate*a['minimum_cards_per_order']*ceil(d['wall_seconds']/q)*q/3600
            fallback.append({'operation':'A100 MIG direct full-MHA support fallback','scope':'24 full-MHA configurations; one reference process','physical_wall_seconds':d['wall_seconds'],'rate_provider':a['id'],'plan':'hourly','billing_quantum_seconds_assumed':q,'whole_order_cards':a['minimum_cards_per_order'],'allocated_wall_cost_cny':rate*d['wall_seconds']/3600,'minimum_new_order_charge_cny_excluding_startup':minimum,'money_status':'one whole-card allocation; order minimum separate; cold start excluded','source':'sources/fallback/'+f})
    else:
        fallback.append({'operation':'M5 MPS direct full-MHA support fallback','scope':'24 full-MHA configurations; one reference process','physical_wall_seconds':d['wall_seconds'],'rate_provider':'no matched public tariff','plan':'not_applicable','billing_quantum_seconds_assumed':'','whole_order_cards':'','allocated_wall_cost_cny':'','minimum_new_order_charge_cny_excluding_startup':'','money_status':'no monetary estimate without applicable tariff','source':'sources/fallback/'+f})
table('fallback-costs.csv',fallback)

# Main scenario examples are specified by fixed predicates, not best outcomes.
def example(provider='lambda',plan='monthly',active=24,demand=1,batch=1,packing=1,quantum=0,budget=10):
    return next(r for r in service if r['model']=='resnet18' and r['batch']==batch and r['a100_provider']==provider and r['npu_tariff']=='ctyun_public' and r['npu_plan']==plan and r['active_hours_per_day']==active and r['demand_images_per_second']==demand and r['assumed_mig_slices_per_a100_card']==packing and r['response_budget_ms']==budget and r['latency_endpoint']=='mean' and r['billing_policy']=='scheduled_release' and r['npu_billing_quantum_seconds_assumed']==quantum)
examples={
'low_demand_lambda_minimum_order':example(),
'low_demand_runpod_single_card':example(provider='runpod'),
'busy_service_lambda':example(demand=1000),
'scheduled_one_hour_lambda_hourly':example(plan='hourly',active=1,quantum=1),
'scheduled_one_hour_lambda_monthly':example(active=1),
'low_demand_batching_infeasible':example(batch=4),
'five_minutes_one_second_quantum':example(provider='runpod',plan='hourly',active=1/12,quantum=1),
'five_minutes_one_hour_quantum':example(provider='runpod',plan='hourly',active=1/12,quantum=3600)}
summary={
'query_date':P['query_date'],'scope':'Measured latency and quality with separately priced cost and service scenarios; no new hardware execution or authenticated rental invoices.',
'source_count':len(load(S/'fetch-manifest.json')),'fx':P['fx'],
'statistical_protocol':{'latency':'95% t interval over five independent physical-process means (30 timed calls within each process), df=4.','ratio':'Fieller interval for independent target/source means, zero cross-hardware covariance and conservative t(df=4); descriptive normal-means approximation. Matched run indices do not establish statistical pairing.','interval_limit':'Intervals quantify process timing variation only; prices, taxes, FX and service tail behavior are separate assumptions; no multiplicity adjustment.'},
'formulas':{'active_compute_cny_per_image':'P_cny_hour*t_ms/(3,600,000*batch)','saving_fraction':'1-(P_N*t_N)/(alpha*P_A*t_A)','service_month':'billed_cards * rounded_billed_hours * hourly_rate, or billed_cards * monthly_fee','66_percent_condition':'P_N*t_N <= 0.34*alpha*P_A*t_A','66_service_condition':'NPU monthly bill <= 0.34*A100 monthly bill, under same feasible request demand, quality and response model'},
'measured_timing':timing,
'counts':{k:v.get('rows') for k,v in OUT.items()},
'normalized_sensitivity':{'scenarios':len(sensitivity),'achieves_66':sum(r['achieves_66_percent'] for r in sensitivity),'positive_savings':sum(r['npu_cost_saving_percent']>0 for r in sensitivity),'saving_range_percent':[min(r['npu_cost_saving_percent'] for r in sensitivity),max(r['npu_cost_saving_percent'] for r in sensitivity)]},
'service_sensitivity':{'scenarios':len(service),'feasible_under_model':sum(r['equal_demand_response_model_feasible'] for r in service),'infeasible_under_model':sum(not r['equal_demand_response_model_feasible'] for r in service),'achieves_66_within_model':sum(r['achieves_66_within_model'] for r in service),'feasible_not_66':sum(r['equal_demand_response_model_feasible'] and not r['achieves_66_within_model'] for r in service),'measured_sla':False},
'fixed_scenario_examples':examples,
'migration':{'source_baseline_mean_ms':mean(ma),'target_baseline_mean_ms':mean(mb),'requested_target_mean_ms':mean(mc),'gate_replayed_mean_ms':mean(mg),'mean_process_requested_gain_percent':mean([100*(1-y/x) for x,y in zip(mb,mc)]),'requested_gain_ratio_of_means_percent':100*(1-mean(mc)/mean(mb)),'mean_process_gated_gain_percent':mean([100*(1-y/x) for x,y in zip(mb,mg)]),'gated_gain_ratio_of_means_percent':100*(1-mean(mg)/mean(mb)),'requested_process_gain_ci95':ci([100*(1-y/x) for x,y in zip(mb,mc)]),'gated_process_gain_ci95':ci([100*(1-y/x) for x,y in zip(mb,mg)]),'gate_accepted':sum(r['accepted']=='True' for r in mr),'physical_processes':5,'retention':1.0,'scope':'Same weights/input/data identity and 100 evaluation images; offline requested mask6/gate replay is independent revalidation, not original FITAS15.6%.'},
'fallback':{'active_npu_queries':732,'active_npu_remeasured':83,'decision_wall_seconds':wall['decision_phase_wall_seconds'],'measurement_wall_seconds':wall['measurement_phase_wall_seconds_including_graph_setup'],'combined_wall_seconds':nwall,'timed_blocks_only_seconds':fb['fresh_timed_blocks_cost_ms_excluding_warmup_and_setup']/1000,'operations':'Full-MHA support fallbacks are separate 24-configuration workloads, not equivalent to the NPU Activation fallback.'},
'historical_claim':{'archived_A100_usd_month':500,'archived_Ascend_usd_month':170,'arithmetic_reduction_percent':66,'real_measurements_records':len(load(S/'prior/fitas_migration_results.json')['real_measurements']),'original_case_authenticated':False,'action':'Replace unconditional measured 66% headline with conditional tariff-based results; archive 500/170 as a budget assumption. New experiment is independently reproducible and not original-case reconstruction.','implied_hours_if_reinterpreted_at_current_quotes':{'runpod_a100_card_hours':500/1.79,'lambda_8card_node_hours':500/(2.79*8),'ctyun_public_npu_hours':170*FX/38.45,'author_reported_npu_hours':170*FX/38},'denominator_warning':'Different implied hours cannot define equal demand, equal delivered work, equal SLA or historical paid spend.'},
'limitations':['The A100 measurement used one MIG 2g.20gb slice, not a full 80GB-card benchmark. Whole-card prices and alpha allocations are explicit opportunity-cost assumptions, not observed MIG tariffs.','Runpod and Lambda product-region capacity, permission to create MIG, simultaneous three-slice throughput and matched geographic region are unverified.','A100 hardware is self-owned according to author input. Public cloud prices are alternatives/opportunity cost, not its observed cash outlay or depreciated TCO.','List-price tax bases, VAT recovery, storage, bandwidth, startup, release/restart overhead and engineering are not fully normalized. No final tax-inclusive total is claimed.','Service calculation uses periodic arrivals, deterministic mean or upper-confidence mean times, 0.8 utilization cap and computed batching/queue delays. It is not a measured p95 service guarantee.','Hourly-to-720h conversion is not a vendor monthly contract. Tianyi monthly plans are committed full-month charges even at low duty cycle.','Five process replications and 100 unique quality images limit precision and generality. The same quality set is not 500 independent images.']}
js('results-summary.json',summary)
# Invariant checks include a real negative result and an infeasible batching case.
assert summary['normalized_sensitivity']['achieves_66']==0
assert examples['low_demand_lambda_minimum_order']['achieves_66_within_model']
assert examples['low_demand_runpod_single_card']['npu_cost_saving_percent']<0
assert not examples['low_demand_batching_infeasible']['equal_demand_response_model_feasible']
assert examples['five_minutes_one_hour_quantum']['npu_cny_per_month']==12*examples['five_minutes_one_second_quantum']['npu_cny_per_month']
assert isclose(summary['migration']['mean_process_requested_gain_percent'],-3.752442225434733,abs_tol=1e-9)
verification={'status':'PASS','physical_raw_files':40,'source_manifests_verified':True,'means_recomputed_from_30_samples':True,'cross_hardware_weights_input_data_identity':True,'quality_counts_recomputed':True,'all_negative_and_infeasible_scenarios_retained':True,'no_hardware_or_network_execution':True,'csv_tables':8,'outputs':OUT}
js('verification.json',verification)
print(json.dumps({'counts':summary['counts'],'normalized':summary['normalized_sensitivity'],'service':summary['service_sensitivity'],'migration':summary['migration'],'examples':{k:{f:r[f] for f in ['a100_cny_per_month','npu_cny_per_month','npu_cost_saving_percent','equal_demand_response_model_feasible']} for k,r in examples.items()}},ensure_ascii=False,indent=2))
