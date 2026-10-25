"""New offline PPO planner trained on measured calibration coalition tables.

Held-out tables are used only after an action sequence has been selected using
calibration observations. This is not the submitted PPO implementation, and
provenance field validation alone does not authenticate a physical experiment.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from copa_v2 import exact_shapley, finite, game_from_json, improvement_game, valid_players


def mean_latency(record):
    if record['status'] != 'ok': raise ValueError('latency unavailable for non-ok record')
    values = record.get('latency_samples_ms')
    if not isinstance(values,list) or not values: raise ValueError('raw latency vector missing')
    values = [finite(v,'latency sample') for v in values]
    if min(values) <= 0: raise ValueError('latency samples must be positive')
    return statistics.mean(values)


def quality_score(record):
    if record.get('quality_evaluated') is not True: raise ValueError('quality not evaluated')
    q = finite(record.get('quality_score'),'quality score')
    if not 0 <= q <= 1: raise ValueError('quality score outside [0,1]')
    return q


class MeasuredTable:
    def __init__(self,payload,allow_unit_fixtures=False):
        self.payload = payload
        if payload.get('schema_version') != 'het.measured.coalitions.v2': raise ValueError('wrong schema')
        if not payload.get('task_id') or not payload.get('device_id'): raise ValueError('missing task/device ID')
        self.actions = valid_players(payload['action_ids']); self.n = len(self.actions)
        if self.n > 12: raise ValueError('finite table planner capped at 12 actions')
        qd = payload.get('quality_definition',{})
        if not qd.get('metric') or not qd.get('dataset_id'): raise ValueError('quality metric/dataset required')
        self.threshold = finite(qd.get('threshold'),'quality threshold')
        if not 0 <= self.threshold <= 1: raise ValueError('quality threshold outside [0,1]')
        self.mode = qd.get('mode')
        if self.mode not in ['absolute','retention_ratio']: raise ValueError('quality mode must be explicit')
        metadata = payload.get('action_metadata',[])
        self.metadata = {m['id']:m for m in metadata}
        if len(self.metadata) != self.n or set(self.metadata) != set(self.actions): raise ValueError('action metadata IDs differ')
        self.supported = [self.metadata[a].get('hardware_supported') is True for a in self.actions]
        self.splits = {}; run_ids = set()
        for split in ['train','eval']:
            blocks = payload.get(split)
            if not isinstance(blocks,list) or not blocks: raise ValueError(f'{split} blocks missing')
            parsed = []
            for block in blocks:
                rid = block.get('run_id')
                if not isinstance(rid,str) or not rid or rid in run_ids: raise ValueError('duplicate/missing run ID across splits')
                run_ids.add(rid)
                provenance = block.get('provenance')
                if provenance != 'physical_measurement' and not (allow_unit_fixtures and provenance == 'UNIT_FIXTURE'):
                    raise ValueError('physical measurement provenance required')
                records = {}
                for rec in block.get('coalitions',[]):
                    mask = rec.get('mask')
                    if isinstance(mask,bool) or not isinstance(mask,int) or not 0 <= mask < (1 << self.n) or mask in records:
                        raise ValueError('invalid/duplicate coalition mask')
                    if rec.get('status') not in ['ok','unsupported','failed','oom']: raise ValueError('unknown status')
                    if not rec.get('artifact_id'): raise ValueError('artifact reference required')
                    if rec['status']=='ok': mean_latency(rec)
                    records[mask] = rec
                if set(records) != set(range(1 << self.n)): raise ValueError('all coalition statuses must be retained')
                baseline = records[0]
                mean_latency(baseline); bq = quality_score(baseline)
                if self.mode=='retention_ratio' and bq <= 0: raise ValueError('positive baseline quality required')
                if 'baseline_quality' in block and abs(finite(block['baseline_quality'],'block baseline')-bq)>1e-9:
                    raise ValueError('block baseline quality disagrees with mask0')
                parsed.append({'run_id':rid,'records':records,'quality_partition':block.get('quality_partition'),
                               'quality_sample_ids':block.get('quality_sample_ids',[]),'provenance':provenance})
            self.splits[split] = parsed
        train_ids={b['run_id'] for b in self.splits['train']}; eval_ids={b['run_id'] for b in self.splits['eval']}
        if payload.get('copa_game'):
            profile_runs=set(payload['copa_game'].get('profile_run_ids',[]))
            if not profile_runs or profile_runs-train_ids:
                raise ValueError('DAG COPA profiles must identify calibration-only runs')
        for a,m in self.metadata.items():
            profile_runs=set(m.get('profile_run_ids',[]))
            if profile_runs & eval_ids or profile_runs - train_ids: raise ValueError('action metadata profile leaks eval/unknown run')
            ep=m.get('embedding_provenance',{})
            if ep.get('used_inference_latency_labels') is True: raise ValueError('embedding consumes evaluation latency labels')
        train_samples={s for b in self.splits['train'] for s in b['quality_sample_ids']}
        eval_samples={s for b in self.splits['eval'] for s in b['quality_sample_ids']}
        if train_samples & eval_samples: raise ValueError('quality sample IDs overlap train/eval')
        self.quality_sample_disjointness_checked = bool(train_samples and eval_samples)
        self.unit_fixtures = any(b['provenance']=='UNIT_FIXTURE' for v in self.splits.values() for b in v)

    def calibrated_observation(self,mask):
        """Uses training records only, including while selecting an eval plan."""
        rows=[b['records'][mask] for b in self.splits['train']]
        if any(r['status']!='ok' for r in rows): return None
        q=[]
        for block,row in zip(self.splits['train'],rows):
            try:
                score=quality_score(row)
                if self.mode=='retention_ratio':score/=quality_score(block['records'][0])
                q.append(score)
            except ValueError:return None
        return statistics.mean(mean_latency(r) for r in rows),statistics.mean(q)

    def final_gate(self,block,mask,require_latency_nonregression=True):
        r=block['records'][mask]; base=block['records'][0]; baseline=mean_latency(base)
        raw_latency=None; raw_gain=None; normalized_quality=None
        reason='ok'
        if any(mask & (1<<i) and not self.supported[i] for i in range(self.n)): reason='hardware_unsupported'
        elif r['status']!='ok':reason=r['status']
        else:
            raw_latency=mean_latency(r);raw_gain=100*(baseline-raw_latency)/baseline
            try:
                normalized_quality=quality_score(r)
                if self.mode=='retention_ratio':normalized_quality/=quality_score(base)
                if normalized_quality < self.threshold:reason='quality_below_threshold'
                elif require_latency_nonregression and raw_latency>baseline:reason='latency_regression'
            except ValueError:reason='quality_evidence_invalid'
        accepted=reason=='ok'
        return {'run_id':block['run_id'],'requested_mask':mask,'accepted':accepted,'reason':reason,
                'raw_latency_ms':raw_latency,'raw_benefit_percent':raw_gain,'normalized_quality':normalized_quality,
                'accepted_mask':mask if accepted else 0,'accepted_latency_ms':raw_latency if accepted else baseline,
                'accepted_benefit_percent':raw_gain if accepted else 0.0,
                'rollback_to_baseline':not accepted,'quality_mode':self.mode,'threshold':self.threshold,
                'baseline_quality':quality_score(base),'artifact_id':r.get('artifact_id')}


class CalibrationEnv:
    def __init__(self,table:MeasuredTable,budget,hardware_mask=True,with_embedding=False,drop_embedding=False):
        if not isinstance(budget,int) or not 0 < budget <= table.n: raise ValueError('invalid action budget')
        if drop_embedding and not with_embedding:raise ValueError('drop-embedding control requires actual embedding metadata')
        self.table=table;self.budget=budget;self.hardware_mask=hardware_mask;self.with_embedding=with_embedding
        self.drop_embedding=drop_embedding
        self.embedding=[]
        if with_embedding:
            embeddings=[table.metadata[a].get('graph_embedding') for a in table.actions]
            if any(not isinstance(e,list) or not e for e in embeddings) or len({len(e) for e in embeddings})!=1:
                raise ValueError('real frozen same-dimension graph embeddings required')
            self.embedding=[finite(v,'embedding') for e in embeddings for v in e]
            if not any(abs(v)>1e-12 for v in self.embedding):raise ValueError('all-zero placeholder embedding forbidden')
            for action in table.actions:
                provenance=table.metadata[action].get('embedding_provenance',{})
                if not provenance.get('checkpoint_sha256') or provenance.get('used_inference_latency_labels') is not False:
                    raise ValueError('embedding checkpoint and label-free inference provenance required')
                training_models=provenance.get('training_model_ids')
                if not isinstance(training_models,list) or not training_models:
                    raise ValueError('embedding training model identities required')
                if table.payload.get('model_id') in training_models:
                    raise ValueError('target model was used to train purported heldout embedding')
                if not provenance.get('split_id') or table.payload.get('model_id') not in provenance.get('test_model_ids',[]):
                    raise ValueError('target model must belong to the declared heldout test split')
                if table.payload.get('model_id') in provenance.get('validation_model_ids',[]):
                    raise ValueError('target model belonged to embedding validation/tuning split')
            if drop_embedding:self.embedding=[0.0]*len(self.embedding)
        self.base_cal=table.calibrated_observation(0)
        if self.base_cal is None:raise ValueError('calibration baseline invalid')
        # These static features must come from calibration metadata, not eval.
        self.static=[]
        for a in table.actions:
            m=table.metadata[a]
            self.static.extend([float(m.get('quality_delta_prior',0.0)),
                                float(m.get('operator_latency_ms',0.0))/self.base_cal[0],
                                float(m.get('local_benefit_ms',0.0))/self.base_cal[0]])
        if not all(math.isfinite(v) for v in self.static):raise ValueError('nonfinite static features')
        self.block=None;self.mask=0;self.steps=0;self.done=False

    def reset(self,block):
        if block not in self.table.splits['train']:raise ValueError('training environment cannot reset to eval block')
        # A deterministic calibration MDP uses the unweighted measured means
        # across all training blocks. No hidden per-episode run context.
        self.block=None;self.mask=0;self.steps=0;self.done=False
        return self.state()

    def state(self):
        obs=self.table.calibrated_observation(self.mask)
        latency,q=obs if obs is not None else (self.base_cal[0],0.0)
        values=[float(bool(self.mask&(1<<i))) for i in range(self.table.n)]
        values += [(self.budget-self.steps)/self.budget,q,latency/self.base_cal[0]]
        values += self.static+self.embedding
        return torch.tensor(values,dtype=torch.float32)

    def valid_mask(self):
        available=[not self.done and self.steps<self.budget and not self.mask&(1<<i)
                   and (self.table.supported[i] or not self.hardware_mask) for i in range(self.table.n)]
        return torch.tensor(available+[True],dtype=torch.bool)  # stop/no-op always legal

    def step(self,action):
        if self.done:raise ValueError('episode already terminal')
        if action==self.table.n:self.done=True;return self.state(),0.0,True,{'stop':True}
        if not 0 <= action < self.table.n or not self.valid_mask()[action]:raise ValueError('illegal action')
        before=self.table.calibrated_observation(self.mask);nxt=self.mask|(1<<action)
        self.mask=nxt;self.steps+=1
        after=self.table.calibrated_observation(nxt)
        if not self.table.supported[action] or after is None:
            self.done=True
            return self.state(),-2.0,True,{'invalid':True,'status':'hardware_or_calibration_evidence_invalid'}
        reward=(before[0]-after[0])/self.base_cal[0]
        q=after[1]
        if q<self.table.threshold:reward-=2.0*(self.table.threshold-q)
        self.done=self.steps>=self.budget
        return self.state(),reward,self.done,{'invalid':False,'normalized_quality':q}


class ActorCritic(nn.Module):
    def __init__(self,dim,actions):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(dim,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.actor=nn.Linear(64,actions);self.critic=nn.Linear(64,1)
    def forward(self,state,mask):
        z=self.body(state);logits=self.actor(z).masked_fill(~mask,-1e9)
        return Categorical(logits=logits),self.critic(z).squeeze(-1)


def train_ppo(table,budget=2,seed=42,total_steps=4096,hardware_mask=True,with_embedding=False,
              batch_steps=256,epochs=5,lr=0.003,clip=0.2,gae_lambda=0.95,drop_embedding=False):
    torch.manual_seed(seed);rng=random.Random(seed)
    env=CalibrationEnv(table,budget,hardware_mask,with_embedding,drop_embedding)
    env.reset(table.splits['train'][0]);model=ActorCritic(len(env.state()),table.n+1)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr);history=[];count=0;invalid_count=0;episodes=0
    while count<total_steps:
        states=[];masks=[];actions=[];oldlogs=[];returns=[];advantages=[]
        while len(states)<batch_steps and count<total_steps:
            env.reset(table.splits['train'][0]);episode=[]
            while not env.done:
                state=env.state();mask=env.valid_mask()
                with torch.no_grad():dist,val=model(state,mask);action=dist.sample();lp=dist.log_prob(action)
                _,reward,done,info=env.step(int(action));invalid_count+=bool(info.get('invalid'))
                episode.append((state,mask,action,lp,val,reward,done));count+=1
            # gamma=1 for finite-horizon normalized latency delta; no artificial
            # discount preference for obtaining an identical gain earlier.
            gae=0.0;next_value=0.0
            for state,mask,action,lp,value,reward,done in reversed(episode):
                delta=reward+(0.0 if done else next_value)-float(value)
                gae=delta+(0.0 if done else gae_lambda*gae)
                states.append(state);masks.append(mask);actions.append(action);oldlogs.append(lp)
                advantages.append(gae);returns.append(gae+float(value));next_value=float(value)
            episodes+=1
        st=torch.stack(states);ma=torch.stack(masks);ac=torch.stack(actions);ol=torch.stack(oldlogs)
        ret=torch.tensor(returns);adv=torch.tensor(advantages)
        adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
        for _ in range(epochs):
            dist,value=model(st,ma);ratio=torch.exp(dist.log_prob(ac)-ol)
            actor=-torch.minimum(ratio*adv,torch.clamp(ratio,1-clip,1+clip)*adv).mean()
            critic=((value-ret)**2).mean();entropy=dist.entropy().mean()
            loss=actor+0.5*critic-0.01*entropy
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),0.5);optimizer.step()
        history.append({'steps':count,'loss':float(loss.detach()),'actor_loss':float(actor.detach()),
                        'critic_loss':float(critic.detach()),'entropy':float(entropy.detach())})
    return model,{'seed':seed,'steps':count,'episodes':episodes,'training_invalid_attempts':invalid_count,
                  'hardware_mask':hardware_mask,'with_embedding':with_embedding,'drop_embedding':drop_embedding,'history':history,
                  'training_run_ids':[b['run_id'] for b in table.splits['train']],
                  'environment':'deterministic MDP of unweighted physical training-run means; no hidden run context or synthetic latency noise',
                  'eval_access_during_training':False,'unit_fixture_data':table.unit_fixtures}


def select_plan(model,table,budget,hardware_mask=True,with_embedding=False,drop_embedding=False):
    # Selection deliberately never receives an eval block.
    env=CalibrationEnv(table,budget,hardware_mask,with_embedding,drop_embedding)
    env.reset(table.splits['train'][0]);sequence=[]
    for _ in range(budget+1):
        with torch.no_grad():dist,_=model(env.state(),env.valid_mask());a=int(dist.logits.argmax())
        if a==table.n:break
        sequence.append(table.actions[a]);env.mask|=1<<a;env.steps+=1
        if env.steps>=budget:break
    return {'mask':env.mask,'actions':sequence,'selection_source':'calibration_only'}


def evaluate_plan(table,mask,require_latency_nonregression=True):
    return [table.final_gate(b,mask,require_latency_nonregression) for b in table.splits['eval']]


def fixed_rank_mask(table,field,budget,positive_only=False):
    if any(table.metadata[a].get(field) is None for a in table.actions):return None
    ordered=sorted(range(table.n),key=lambda i:(-finite(table.metadata[table.actions[i]][field],field),i))
    chosen=[i for i in ordered if table.supported[i] and (not positive_only or table.metadata[table.actions[i]][field]>0)][:budget]
    return sum(1<<i for i in chosen)


def calibration_search(table,budget,greedy=False):
    """Non-RL comparators given the same measured calibration evidence as PPO."""
    base=table.calibrated_observation(0)
    def valid(mask):
        if any(mask&(1<<i) and not table.supported[i] for i in range(table.n)):return None
        obs=table.calibrated_observation(mask)
        return obs if obs is not None and obs[1]>=table.threshold else None
    if greedy:
        mask=0;current=base[0]
        for _ in range(budget):
            candidates=[]
            for i in range(table.n):
                if mask&(1<<i):continue
                nxt=mask|(1<<i);obs=valid(nxt)
                if obs is not None:candidates.append((obs[0],nxt))
            if not candidates:break
            latency,nxt=min(candidates)
            if latency>=current:break
            mask=nxt;current=latency
        return mask
    candidates=[]
    for mask in range(1<<table.n):
        if mask.bit_count()>budget:continue
        obs=valid(mask)
        if obs is not None:candidates.append((obs[0],mask))
    return min(candidates)[1] if candidates else None


def attribution_rankings(table,budget):
    out={}
    # Exact measured calibration Shapley is expensive diagnostic evidence, not
    # a low-query predictor. It is explicitly named separately from DAG COPA.
    values={}
    for mask in range(1<<table.n):
        obs=table.calibrated_observation(mask)
        if obs is None:break
        values[mask]=obs[0]
    if len(values)==1<<table.n:
        scores=exact_shapley(table.actions,lambda m:values[0]-values[m])
        out['measured_calibration_shapley']=scores
    if table.payload.get('copa_game'):
        payload=dict(table.payload['copa_game']);payload['action_ids']=list(table.actions)
        dag=game_from_json(payload)
        out['copa_dag']=exact_shapley(table.actions,improvement_game(dag))
        payload['aggregation']='sum'
        out['copa_sum']=exact_shapley(table.actions,improvement_game(game_from_json(payload)))
        payload['aggregation']='critical_path';payload['runtime_overhead_ms']=0.0
        out['copa_no_runtime']=exact_shapley(table.actions,improvement_game(game_from_json(payload)))
    for name,record in out.items():
        order=sorted(range(table.n),key=lambda i:(-record['values'][table.actions[i]],i))
        eligible=[i for i in order if table.supported[i] and record['values'][table.actions[i]]>0][:budget]
        record['mask']=sum(1<<i for i in eligible)
    return out


def uniform_random_expectation(table,budget,at_most=False):
    safe=[i for i in range(table.n) if table.supported[i]]
    sizes=range(min(budget,len(safe))+1) if at_most else [min(budget,len(safe))]
    masks=[sum(1<<i for i in combination) for size in sizes for combination in itertools.combinations(safe,size)]
    rows=[]
    for block in table.splits['eval']:
        outcomes=[table.final_gate(block,m) for m in masks]
        raw=[o['raw_benefit_percent'] for o in outcomes if o['raw_benefit_percent'] is not None]
        rows.append({'run_id':block['run_id'],'subset_count':len(masks),
                     'raw_benefit_percent':statistics.mean(raw) if len(raw)==len(masks) else None,
                     'accepted_benefit_percent':statistics.mean(o['accepted_benefit_percent'] for o in outcomes),
                     'rejection_fraction':statistics.mean(not o['accepted'] for o in outcomes),
                     'budget_semantics':'uniform subsets of size<=k, including no-op' if at_most else 'uniform subsets of size exactly k',
                     'interpretation':'exact uniform-subset expectation, not independent random-run measurements'})
    return rows


def feasible_oracle(table,budget):
    rows=[]
    for block in table.splits['eval']:
        outcomes=[table.final_gate(block,m) for m in range(1<<table.n) if m.bit_count()<=budget]
        valid=[o for o in outcomes if o['accepted']]
        if valid:
            best=min(valid,key=lambda o:(o['accepted_latency_ms'],o['requested_mask']))
            rows.append({**best,'scope':'posthoc heldout feasible oracle; never a selectable predictor'})
        else:
            rows.append({'run_id':block['run_id'],'accepted':False,'reason':'no_feasible_plan',
                         'scope':'posthoc heldout oracle found no feasible plan, including baseline'})
    return rows
