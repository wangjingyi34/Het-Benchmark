from pathlib import Path
import json,hashlib,csv,math
P=Path(__file__).resolve().parent;B=P.parent;V=B/'20261026-v8'/'v8-逐问处理'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prices=json.loads((V/'cost/pricing-inputs.json').read_text())
rows=list(csv.DictReader((V/'cost/timing-summary.csv').open()));d={(r['model'],int(r['batch']),r['device']):r for r in rows}
fx=prices['fx']['cny_per_usd'];pn=38.45; p_runpod=1.79*fx;p_lambda=2.79*fx
out=[]
for model in ['resnet18','resnet50']:
 for batch in [1,4]:
  ta=float(d[model,batch,'cuda']['mean_ms'])/1000;tn=float(d[model,batch,'npu']['mean_ms'])/1000;r=tn/ta
  out.append({'model':model,'batch':batch,'source_measured':'A100 MIG2g20GB','t_a_s':ta,'t_n_s':tn,'t_n_over_t_a':r,'N_A_price_ratio_runpod_alpha1':pn/p_runpod,'N_A_price_ratio_lambda_alpha1':pn/p_lambda,'N_A_active_cost_ratio_runpod_alpha1':pn*r/p_runpod,'N_A_active_cost_ratio_lambda_alpha1':pn*r/p_lambda,'A_full_card_USD_hour_required_for_66_at_alpha1':pn*r/(.34*fx),'N_CNY_hour_max_for_66_runpod_alpha1':.34*p_runpod/r,'runpod_unit_A_CNY_per_image':p_runpod*ta/(3600*batch),'N_unit_CNY_per_image':pn*tn/(3600*batch)})
ca=8*2.79*fx*720;cn=18454;cr=1.79*fx*720
service={'conditional_lambda8_CNY_month':ca,'npu_CNY_month':cn,'runpod1_CNY_720hours':cr,'savings_vs_lambda8_percent':100*(1-cn/ca),'savings_vs_runpod1_percent':100*(1-cn/cr),'source_bill_required_for_66_CNY':cn/.34,'equal_feasible_demand_assumption':'1image/s,720h,periodic arrival,batch1,quality>=.99paired retention; excludes startup/network; deployment suitability is an assumption, not measured cloud SLA.'}
assert service['savings_vs_lambda8_percent']>=66 and service['savings_vs_runpod1_percent']<0
for r in out:assert r['N_A_active_cost_ratio_runpod_alpha1']>1 and r['N_A_active_cost_ratio_lambda_alpha1']>1
# Numerical controls complement, but do not replace, the analytic proof.
from itertools import product
checked=0
for truth in product([1.,1.25,1.5], repeat=3):
 for errors in product([-.25,0.,.25], repeat=3):
  prediction=[t+e for t,e in zip(truth, errors)]
  picked=min(range(3), key=lambda i:prediction[i])
  assert truth[picked]-min(truth)<=.5+1e-12
  checked+=1
tight={'truth':[1.5,1.0],'prediction':[1.25,1.25],'epsilon':.25,'tie_pick':0,'regret':.5}
assert tight['truth'][0]-min(tight['truth'])==2*tight['epsilon']
files=[V/'cost/pricing-inputs.json',V/'cost/timing-summary.csv']
result={'source_hashes':{str(f.relative_to(B)):sha(f) for f in files},'source_measurement_commit':'35e91a692716fd890ab80c90f3da9de9a3f4127e','cost_rows':out,'service':service,'regret_controls':checked,'tight_example':tight,'scope':'Conditional recomputation using prior measurements and frozen tariffs. No new hardware run. No historical-result authentication. The eight-card procurement example is not a per-inference efficiency result.'}
(P/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'contexts':len(out),'regret_controls':checked,'scope':result['scope']},indent=2))
