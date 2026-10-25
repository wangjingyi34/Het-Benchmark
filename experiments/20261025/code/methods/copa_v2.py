"""New auditable COPA reference. Positive values retain the caller's game meaning.

A migration game with target costs gives positive regression attribution. A
measured improvement game baseline - latency gives positive improvement. These
are deliberately separate constructions, never silently sign-flipped.
"""
from __future__ import annotations

import itertools
import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric')
    return float(value)


def valid_players(players: Sequence[str]):
    players = tuple(players)
    if not players or len(set(players)) != len(players) or not all(isinstance(p, str) and p for p in players):
        raise ValueError('players must have distinct nonempty string IDs')
    return players


def exact_shapley(players: Sequence[str], value: Callable[[int], float]):
    players = valid_players(players); n = len(players)
    if n > 16: raise ValueError('exact enumeration capped at 16 players')
    values = {m: finite(value(m), f'value[{m}]') for m in range(1 << n)}
    out = {}
    for i, name in enumerate(players):
        phi = 0.0
        for m in range(1 << n):
            if m & (1 << i): continue
            k = m.bit_count()
            weight = math.factorial(k) * math.factorial(n-k-1) / math.factorial(n)
            phi += weight * (values[m | (1 << i)] - values[m])
        out[name] = phi
    return {'values': out, 'endpoint_delta': values[(1 << n)-1] - values[0],
            'efficiency_residual': sum(out.values()) - values[(1 << n)-1] + values[0],
            'coalition_evaluations': len(values)}


def estimate_shapley(players: Sequence[str], value: Callable[[int], float],
                     total_permutations=500, seed=42, antithetic=True):
    players = valid_players(players); n = len(players)
    if not isinstance(total_permutations, int) or total_permutations < 1:
        raise ValueError('positive total permutation budget required')
    if antithetic and total_permutations % 2:
        raise ValueError('antithetic total permutation budget must be even')
    rng = random.Random(seed); cache = {}; units = []; permutations = []
    def v(m):
        if m not in cache: cache[m] = finite(value(m), f'value[{m}]')
        return cache[m]
    def marginal(order):
        result = [0.0] * n; m = 0
        for i in order:
            nxt = m | (1 << i); result[i] = v(nxt) - v(m); m = nxt
        return result
    count = total_permutations // 2 if antithetic else total_permutations
    for _ in range(count):
        order = list(range(n)); rng.shuffle(order)
        a = marginal(order); permutations.append({'order': order, 'marginals': a})
        if antithetic:
            reverse = list(reversed(order)); b = marginal(reverse)
            permutations.append({'order': reverse, 'marginals': b})
            units.append([(x+y)/2 for x,y in zip(a,b)])
        else: units.append(a)
    means = {p: statistics.mean(u[i] for u in units) for i,p in enumerate(players)}
    se = {p: statistics.stdev(u[i] for u in units)/math.sqrt(count) if count > 1 else None
          for i,p in enumerate(players)}
    return {'values': means, 'standard_errors': se, 'total_permutations': total_permutations,
            'independent_units': count, 'unit': 'pair_mean' if antithetic else 'permutation',
            'unit_values': units, 'permutations': permutations, 'seed': seed,
            'coalition_evaluations': len(cache),
            'efficiency_residual': sum(means.values()) - v((1 << n)-1) + v(0)}


class ActionDAGGame:
    """Coalition counterfactual for action-specific node-cost updates on a DAG.

    Runtime is a surrogate: critical_path or sum. Graph edges do not prove
    parallel physical execution. Overhead must be supplied explicitly and may
    vary by coalition. Overhead calibration data/provenance belongs to caller.
    """
    def __init__(self, action_ids, base_costs: Mapping[str, float], edges,
                 action_cost_updates: Mapping[str, Mapping[str, float]],
                 runtime_overhead: Callable[[int], float] | float = 0.0,
                 aggregation='critical_path'):
        self.action_ids = valid_players(action_ids)
        self.base = {str(k): finite(v, f'cost {k}') for k,v in base_costs.items()}
        if not self.base or any(v < 0 for v in self.base.values()):
            raise ValueError('nonempty nonnegative node costs required')
        if aggregation not in ['critical_path','sum']: raise ValueError('unknown aggregation')
        self.aggregation = aggregation; self.runtime_overhead = runtime_overhead
        self.pred = {k: [] for k in self.base}; successors = {k: [] for k in self.base}
        edge_set = set()
        for a,b in edges:
            if a not in self.base or b not in self.base or a == b: raise ValueError('invalid DAG edge')
            if (a,b) in edge_set: continue
            edge_set.add((a,b)); self.pred[b].append(a); successors[a].append(b)
        indegree = {k: len(v) for k,v in self.pred.items()}
        ready = sorted(k for k,v in indegree.items() if not v); order = []
        while ready:
            node = ready.pop(0); order.append(node)
            for b in successors[node]:
                indegree[b] -= 1
                if not indegree[b]: ready.append(b); ready.sort()
        if len(order) != len(self.base): raise ValueError('dependency graph is cyclic')
        self.order = order; self.updates = {}; touched = set()
        if set(action_cost_updates) != set(self.action_ids): raise ValueError('updates must cover actions exactly')
        for action, updates in action_cost_updates.items():
            self.updates[action] = {}
            for node,cost in updates.items():
                if node not in self.base or node in touched: raise ValueError('unknown or overlapping action node')
                val = finite(cost, f'candidate cost {node}')
                if val < 0: raise ValueError('negative candidate cost')
                self.updates[action][node] = val; touched.add(node)
        self._cache = {}

    def __call__(self, mask):
        if isinstance(mask,bool) or not isinstance(mask,int) or not 0 <= mask < (1 << len(self.action_ids)):
            raise ValueError('coalition mask out of range')
        if mask in self._cache:return self._cache[mask]
        total=self.raw_runtime(mask)
        if total <= 0: raise ValueError('total surrogate runtime must be positive')
        self._cache[mask]=total
        return total

    def raw_runtime(self,mask):
        """Unclipped prediction retained for validation and failure accounting."""
        if isinstance(mask,bool) or not isinstance(mask,int) or not 0 <= mask < (1 << len(self.action_ids)):
            raise ValueError('coalition mask out of range')
        costs = dict(self.base)
        for i,action in enumerate(self.action_ids):
            if mask & (1 << i): costs.update(self.updates[action])
        if self.aggregation == 'sum': total = sum(costs.values())
        else:
            end = {}
            for node in self.order:
                end[node] = costs[node] + max((end[p] for p in self.pred[node]), default=0.0)
            total = max(end.values())
        overhead = self.runtime_overhead(mask) if callable(self.runtime_overhead) else self.runtime_overhead
        total += finite(overhead, 'runtime overhead')
        return total


def game_from_json(payload):
    overhead = payload.get('runtime_overhead_ms', 0.0)
    if isinstance(overhead, dict):
        table = {int(k): finite(v, f'overhead[{k}]') for k,v in overhead.items()}
        required = set(range(1 << len(payload['action_ids'])))
        if set(table) != required: raise ValueError('explicit coalition overhead table incomplete')
        overhead = lambda m: table[m]
    return ActionDAGGame(payload['action_ids'], payload['base_costs_ms'], payload['edges'],
                         payload['action_cost_updates_ms'], overhead,
                         payload.get('aggregation','critical_path'))


def improvement_game(runtime):
    baseline = finite(runtime(0), 'baseline runtime')
    return lambda mask: baseline - runtime(mask)


def signed_l1_and_ranks(reference, predicted):
    if set(reference) != set(predicted): raise ValueError('score IDs differ')
    players = sorted(reference)
    ref = sorted(players,key=lambda p:(-reference[p],p)); pred = sorted(players,key=lambda p:(-predicted[p],p))
    return {'signed_mae_ms': statistics.mean(abs(reference[p]-predicted[p]) for p in players),
            'sign_agreement': statistics.mean((reference[p]>0)-(reference[p]<0) ==
                                             (predicted[p]>0)-(predicted[p]<0) for p in players),
            'reference_rank': ref, 'predicted_rank': pred,
            'top1_match': ref[0] == pred[0],
            'top2_overlap':len(set(ref[:2]) & set(pred[:2]))/min(2,len(players))}
