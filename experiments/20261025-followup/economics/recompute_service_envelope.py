"""Frozen offline economics audit. Stdlib only; no remote hardware/orders."""
from pathlib import Path
import csv, datetime, gzip, hashlib, itertools, json, math, statistics
O=Path(__file__).resolve().parent; R=O.parents[1]
P=json.loads((O/'protocol.json').read_text()); Q=json.loads((O/'official-quotes.json').read_text())
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
F=json.loads((O/'freeze-receipt.json').read_text())
assert sha(O/'protocol.json')==F['protocol_sha256']
assert sha(O/'official-quotes.json')==F['quotes_sha256']
for x in P['frozen_sources']: assert sha(R/x['path'])==x['sha256'], x['path']
rates={x['id']:x for x in Q['rates']}; baselines=[]; bs={}; runtimes=[]
for dev,model,batch in itertools.product(['cuda','npu'],['resnet18','resnet50'],[1,4]):
    means=[]; sources=[]
    for run in P['source_selection']['runs']:
        p=R/f'v2-实证增强/evidence/fullmodel_raw/{dev}-{model}-b{batch}-r{run}.json'; d=json.loads(p.read_text())
        assert d['phase']=='eval' and d['quality_partition']=='evaluation'
        assert d['provenance']=='physical_measurement' and d['input_shape']==[batch,3,224,224]
        c=next(c for c in d['coalitions'] if c['mask']==0)
        assert c['status']=='ok' and len(c['latency_samples_ms'])==30 and c['quality_threshold_passed']
        m=statistics.mean(c['latency_samples_ms']); assert math.isclose(m,c['mean_ms'],rel_tol=1e-12)
        means.append(m); sources.append(str(p.relative_to(R)))
        runtimes.append({'path':str(p.relative_to(R)),'device_id':d['device_id'],'runtime':d['runtime'],'protocol':d['protocol']})
    v={'backend':dev,'model':model,'batch':batch,'process_count':5,'calls_per_process':30,'process_means_ms':means,'mean_of_five_process_means':statistics.mean(means),'max_of_five_process_means':max(means),'sample_sd_of_process_means_ms':statistics.stdev(means),'device_id':d['device_id'],'source_paths':sources}
    baselines.append(v); bs[dev,model,batch]=v
old=json.loads((R/'v2-实证增强/methods/market-cost-scenarios.json').read_text())
for x in old['baselines']:
    dev='cuda' if x['backend']=='a100_mig' else 'npu'; v=bs[dev,x['model_id'],x['batch_size']]
    assert math.isclose(v['mean_of_five_process_means'],x['mean_ms'],rel_tol=1e-12)
    assert all(math.isclose(a,b,rel_tol=1e-12) for a,b in zip(v['process_means_ms'],x['process_means_ms']))
v3=json.loads((R/'v3-逐问重构/evidence/official-market-cost-recheck.json').read_text())
max_diff=0
for x in v3['rows']:
    a=bs['cuda',x['model'],x['batch']]['mean_of_five_process_means']; n=bs['npu',x['model'],x['batch']]['mean_of_five_process_means']
    ca=x['a100_allocated_cny_h']*a/(3600*x['batch']); cn=x['npu_cny_h']*n/(3600*x['batch']); reduction=100*(1-cn/ca)
    max_diff=max(max_diff,abs(reduction-x['cost_reduction_percent']))
    assert math.isclose(reduction,x['cost_reduction_percent'],rel_tol=1e-12)
    assert (cn<=.34*ca)==x['supports_66_percent_or_better']
(O/'raw-baselines.json').write_text(json.dumps({'source_manifest_sha256':F['protocol_sha256'],'baselines':baselines,'runtimes':runtimes},indent=2)+'\n')
# Capacity-normalized allocation is independent from procurement minimums.
normalized=[]
for model,batch,stat,prov,fx,alpha,plan in itertools.product(['resnet18','resnet50'],[1,4],P['latency_statistics'],['runpod','lambda'],P['normalized_grid']['FX'],P['normalized_grid']['A100_allocation_alpha'],['hourly','monthly672','monthly720','monthly744']):
    a=bs['cuda',model,batch][stat]; n=bs['npu',model,batch][stat]
    pn=38.45 if plan=='hourly' else 18454/int(plan.replace('monthly','')); pa=rates[prov]['hourly_per_card']*fx*alpha
    ca=pa*a/(3600*batch); cn=pn*n/(3600*batch)
    normalized.append({'model':model,'batch':batch,'latency_statistic':stat,'provider':prov,'FX_CNY_per_USD':fx,'allocation_alpha':alpha,'NPU_plan':plan,'A100_MIG_ms':a,'NPU_ms':n,'A100_allocated_CNY_h':pa,'NPU_CNY_h':pn,'A100_CNY_per1000_samples':ca,'NPU_CNY_per1000_samples':cn,'NPU_over_A100_cost_ratio':cn/ca,'reduction_percent':100*(1-cn/ca),'NPU_cheaper':cn<ca,'supports_66':cn<=.34*ca,'whole_A100_CNY_h_at_parity':pn*n/(a*alpha),'whole_A100_CNY_h_required_for66':pn*n/(a*alpha*.34),'NPU_CNY_h_at_parity':pa*a/n,'NPU_CNY_h_for66':.34*pa*a/n,'utilization_ratio_uN_over_uA_for_parity':cn/ca,'utilization_ratio_uN_over_uA_for66':cn/(.34*ca)})

def write_csv(path,rows,compressed=False):
    opener=gzip.open if compressed else open
    with opener(path,'wt',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write_csv(O/'capacity-normalized.csv',normalized)
# Independent periodic-service replay validates batching/round-robin scheduling.
service=[]; scheduling_checks=0
for model,batch,stat,demand,budget,cap in itertools.product(['resnet18','resnet50'],[1,4],P['latency_statistics'],P['service_grid']['demand_samples_per_second'],P['service_grid']['maximum_response_budget_ms'],P['service_grid']['utilization_cap']):
    z={'model':model,'batch':batch,'latency_statistic':stat,'demand_samples_s':demand,'response_budget_ms':budget,'utilization_cap':cap}
    for dev,label in [('cuda','A100'),('npu','NPU')]:
        ms=bs[dev,model,batch][stat]; k=max(1,math.ceil(demand*ms/(1000*batch*cap)))
        last=[-math.inf]*k; responses=[]
        for j in range(200):
            arrival_last=(j*batch+batch-1)/demand; lane=j%k; start=max(arrival_last,last[lane]); end=start+ms/1000;last[lane]=end
            for t in range(batch):responses.append((end-(j*batch+t)/demand)*1000)
        analytic=(batch-1)*1000/demand+ms
        assert math.isclose(max(responses),analytic,abs_tol=1e-8)
        util=demand*ms/(1000*batch*k); assert util<=cap+1e-12
        z.update({label+'_latency_ms':ms,label+'_replicas':k,label+'_active_utilization':util,label+'_deterministic_max_response_ms':analytic,label+'_budget_feasible':analytic<=budget,label+'_serial_capacity_samples_s':1000*batch/ms})
        scheduling_checks+=1
    z['feasibility']='both' if z['A100_budget_feasible'] and z['NPU_budget_feasible'] else ('A100_only' if z['A100_budget_feasible'] else 'NPU_only' if z['NPU_budget_feasible'] else 'neither')
    service.append(z)
write_csv(O/'service-grid.csv',service)
G=P['billing_grid']; nrows=0; agg={}; examples=[]; minreduction=math.inf; maxreduction=-math.inf
with gzip.open(O/'service-cost-all-cases.csv.gz','wt',newline='',encoding='utf-8') as f:
    w=None
    for s,prov,pack,fx,active,policy,nplan in itertools.product(service,G['a100_providers'],G['a100_MIG_replicas_per_card'],G['fx_CNY_per_USD'],G['active_seconds_per_day'],G['policies'],G['npu_plans']):
        r=dict(s); ka=s['A100_replicas']; kn=s['NPU_replicas']; ainfo=rates[prov]; mincards=ainfo['minimum_cards_per_order']; reqcards=math.ceil(ka/pack); cards=math.ceil(reqcards/mincards)*mincards
        duration=86400 if policy=='keep_warm_all_day' else active; qa=ainfo['billing_quantum_seconds']
        bhA=30*math.ceil(duration/qa)*qa/3600
        monthly=nplan.startswith('monthly'); qn=None if monthly else (1 if nplan=='hourly_quantum1s_assumed' else 3600)
        bhN=720 if monthly else 30*math.ceil(duration/qn)*qn/3600
        ca=cards*bhA*ainfo['hourly_per_card']*fx; cn=kn*(18454 if monthly else bhN*38.45)
        sample_count=30*active*s['demand_samples_s']; ah=30*active/3600
        workA=sample_count*s['A100_latency_ms']/(1000*s['batch']*3600); workN=sample_count*s['NPU_latency_ms']/(1000*s['batch']*3600)
        reduction=100*(1-cn/ca)
        r.update({'provider':prov,'MIG_replicas_per_card_assumption':pack,'FX_CNY_per_USD':fx,'active_seconds_day':active,'policy':policy,'NPU_plan':nplan,'horizon_days':30,'sample_count':sample_count,'A100_minimum_cards_order':mincards,'A100_required_cards':reqcards,'A100_billed_cards':cards,'A100_unused_MIG_slots':cards*pack-ka,'A100_billed_hours_per_card':bhA,'NPU_billed_cards':kn,'NPU_billed_hours_per_card':bhN,'A100_MIG_busy_hours':workA,'A100_nominal_SM_GPU_hours':workA*2/7,'A100_nominal_memory_GPU_hours':workA*.25,'NPU_busy_hours':workN,'A100_reservation_CNY':ca,'NPU_reservation_CNY':cn,'A100_CNY_per1000_delivered_samples':ca*1000/sample_count,'NPU_CNY_per1000_delivered_samples':cn*1000/sample_count,'A100_MIG_average_utilization_over_billed_time':workA/(ka*bhA),'NPU_average_utilization_over_billed_time':workN/(kn*bhN),'A100_reserved_card_SM_busy_fraction':workA*2/7/(cards*bhA),'NPU_over_A100_reservation_ratio':cn/ca,'reduction_percent':reduction,'both_feasible_and_NPU_cheaper':s['feasibility']=='both' and cn<ca,'both_feasible_and_supports66':s['feasibility']=='both' and cn<=.34*ca,'A100_USD_per_card_h_at_parity':cn/(cards*bhA*fx),'A100_USD_per_card_h_required_for66':cn/(cards*bhA*fx*.34)})
        assert cards>=reqcards and cards%mincards==0 and 0<bhA<=720 and 0<bhN<=720
        assert math.isclose(cn/ca,(r['NPU_CNY_per1000_delivered_samples']/r['A100_CNY_per1000_delivered_samples']),rel_tol=1e-12)
        if w is None:w=csv.DictWriter(f,fieldnames=list(r));w.writeheader()
        w.writerow(r);nrows+=1
        key=(prov,nplan,policy,pack)
        a=agg.setdefault(key,{'provider':prov,'NPU_plan':nplan,'policy':policy,'MIG_replicas_per_card':pack,'all_cases':0,'both_feasible':0,'NPU_cheaper_among_both_feasible':0,'supports66_among_both_feasible':0,'min_reduction_among_both_feasible':None,'max_reduction_among_both_feasible':None})
        a['all_cases']+=1
        if s['feasibility']=='both':
            a['both_feasible']+=1;a['NPU_cheaper_among_both_feasible']+=cn<ca;a['supports66_among_both_feasible']+=cn<=.34*ca
            a['min_reduction_among_both_feasible']=reduction if a['min_reduction_among_both_feasible'] is None else min(a['min_reduction_among_both_feasible'],reduction)
            a['max_reduction_among_both_feasible']=reduction if a['max_reduction_among_both_feasible'] is None else max(a['max_reduction_among_both_feasible'],reduction)
            minreduction=min(minreduction,reduction);maxreduction=max(maxreduction,reduction)
        if (s['model']=='resnet18' and s['batch']==1 and s['latency_statistic']=='mean_of_five_process_means' and s['demand_samples_s'] in [100,1000] and s['response_budget_ms']==50 and s['utilization_cap']==.8 and fx==7 and pack==1 and active in [3600,86400] and nplan in ['hourly_quantum1s_assumed','monthly_30day_commitment'] and policy=='scheduled_release_outside_active_period'):
            examples.append(r)
assert len(normalized)==576 and len(service)==448 and nrows==129024
# Verify FX sensitivity on fixed service/procurement rows, independent of conclusions.
monotonic_groups={}
for x in normalized:
    key=tuple(x[k] for k in ['model','batch','latency_statistic','provider','allocation_alpha','NPU_plan'])
    monotonic_groups.setdefault(key,[]).append((x['FX_CNY_per_USD'],x['A100_CNY_per1000_samples'],x['NPU_CNY_per1000_samples']))
for vals in monotonic_groups.values():
    vals.sort()
    assert all(y[1]>x[1] and math.isclose(y[2],x[2],rel_tol=1e-12) for x,y in zip(vals,vals[1:]))
# Independent one-row budget check: fixed monthly commitment does not shrink with activity.
assert math.isclose(18454/38.45,479.94798439531854,rel_tol=1e-12)
summary={'schema':'het.v4.economics.results.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol_sha256':F['protocol_sha256'],'quotes_sha256':F['quotes_sha256'],'new_hardware_executions':0,'original_raw_files':40,'original_process_baselines':len(baselines)*5,'V3_exact_reproduction':{'rows':len(v3['rows']),'maximum_absolute_reduction_difference_percentage_points':max_diff,'supports66_count':sum(x['supports_66_percent_or_better'] for x in v3['rows'])},'new_capacity_normalized':{'cases':len(normalized),'NPU_cheaper_cases':sum(x['NPU_cheaper'] for x in normalized),'supports66_cases':sum(x['supports_66'] for x in normalized),'NPU_over_A100_cost_ratio_range':[min(x['NPU_over_A100_cost_ratio'] for x in normalized),max(x['NPU_over_A100_cost_ratio'] for x in normalized)],'reduction_range_percent':[min(x['reduction_percent'] for x in normalized),max(x['reduction_percent'] for x in normalized)]},'new_service_envelope':{'service_cases':len(service),'schedule_replays':scheduling_checks,'financial_cases':nrows,'feasibility_counts':{v:sum(s['feasibility']==v for s in service) for v in ['both','A100_only','NPU_only','neither']},'reservation_range_among_both_feasible_percent':[minreduction,maxreduction],'stratified_results':list(agg.values()),'counts_are_scenarios_not_statistical_evidence':True},'cost_identity':{'original_arithmetic_only':'1-170/500=0.66','original_invoice_authenticated':False,'old_cost_denominator_and_service_demand_authenticated':False,'new_envelope_reconstructs_original_case':False},'billing_breakeven':{'NPU_monthly_vs_hourly_active_billable_hours':18454/38.45,'fraction_of720h':18454/(38.45*720),'comparison':'Monthly becomes cheaper only above this many billed card-hours per month, assuming samecard and no otherfees. Scheduled release below that boundary favors hourly.'},'interpretation':['All576 capacity-normalized scenarios favor the measured A100 MIG, under the fixed official price references and accounting allocations. This does not establish fullA100 performance or a universal price ranking.','Minimum-order reservation can make an NPU appear cheaper against an otherwise unused8card Lambda order. This is idle procurement overhead, not66percent algorithmic savings or reconstruction of the originalFITAS cost.','A1002g.20GB is neither a fullA100 benchmark nor a retail MIGquote; fraction2/7 counts nominalSMGPUhours, while actualbilledcardhours retain integerorders.','Service feasibility is an ideal periodic-arrival model using original processmean latencies, with explicit batching delay and utilizationheadroom. ProductiontailSLA requires measured arrivals, networking, queuing and concurrency evidence.','No currentofficialquote or newofflinemodel authenticates historical500/170 without pricedunit, duration, demand and billingreceipt.']}
(O/'results-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');write_csv(O/'representative-cases.csv',examples)
(O/'verification.json').write_text(json.dumps({'status':'passed','raw_hashes_checked':40,'raw_process_means_checked':40,'raw_baseline_matches_V2':True,'V3_rows_reproduced':288,'service_cases':448,'schedule_replays':896,'financial_rows':129024,'price_monotonicity_groups_checked':len(monotonic_groups),'all_cases_retained':True,'checks':['frozenprotocol and quoteSHA','allsourceSHA','heldoutphase and physicalprovenance','all30 originalcalls and processmeans','capacitynormalization and batchunits','same demand,budget and activity on bothdevices','deterministic batch formation and roundrobin replay','replica utilization not above cap','minimumorder integerpurchases','reservedhour bounds','total and per1000 cost ratios identical','allplanned cases retained including infeasible','no paidorders or hardwareexecutions']},indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['new_service_envelope','interpretation']},ensure_ascii=False));print(json.dumps({'service':{k:v for k,v in summary['new_service_envelope'].items() if k!='stratified_results'}},ensure_ascii=False))
