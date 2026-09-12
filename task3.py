"""Task 3: fast assignment methods, evaluated on test split.

Methods (all trained/tuned on train only; test observed_quality used only for evaluation):

  Oracle            IP on test with observed quality (upper-bound reference).
  Rule (naive)      Highest train-mean quality per query_type, constraint-blind.
                    Included as a counter-example: violates the cost budget.
  Rule (efficiency) Train efficiency = mean_q / mean_c.
                    Greedily assign each query to the highest-efficiency feasible option.
  LP-Batch          LP relaxation of the routing IP over the test batch,
                    plus LP-weighted rounding. Requires batching.
  Shadow-Static     Shadow prices (dual variables) extracted from an aggregate LP on train.
                    Used for online per-query scoring.
  Shadow-Adaptive   Shadow-Static plus multiplicative-weight updates to prices
                    based on realized resource consumption. Main proposal.
"""

import os, time
import pandas as pd
import numpy as np
from collections import defaultdict
from pulp import (LpProblem, LpVariable, LpMaximize, lpSum,
                  LpBinary, LpContinuous, PULP_CBC_CMD)

os.makedirs('results', exist_ok=True)

# data loading and preprocessing
df = pd.read_csv('data.csv')
train = df[df.split == 'train'].copy()
test  = df[df.split == 'test'].copy()

N_test = test.query_id.nunique()
agents  = ['A', 'B', 'C', 'D', 'E']
budgets = [512, 1024, 2048, 4096]
query_types = sorted(train.query_type.unique())

BT_test = 1000 * N_test
BC_test = 0.00022 * N_test
cap_test = {'A': int(0.30 * N_test), 'B': int(0.25 * N_test), 'C': int(0.25 * N_test),
            'D': int(0.25 * N_test), 'E': int(0.20 * N_test)}

print(f"N_test = {N_test}, BT = {BT_test}, BC = ${BC_test:.5f}")
print(f"caps:   {cap_test}")

# used only for evaluation
q_test = {(r.query_id, r.agent_id, r.token_budget): r.observed_quality
          for _, r in test.iterrows()}
c_test = {(r.query_id, r.agent_id, r.token_budget): r.cost_usd
          for _, r in test.iterrows()}

# helper function to evaluate a routing assignment and print summary stats
def evaluate(assignment, label):
    total_q, total_t, total_c = 0.0, 0, 0.0
    load = defaultdict(int)
    for q_id, (a, b) in assignment.items():
        total_q += q_test[(q_id, a, b)]
        total_t += b
        total_c += c_test[(q_id, a, b)]
        load[a] += 1
    cap_ok = all(load[a] <= cap_test[a] for a in agents)
    feasible = (total_t <= BT_test) and (total_c <= BC_test) and cap_ok
    print(f"\n===== {label} =====")
    print(f"  Quality: {total_q:.2f} / {N_test} ({100*total_q/N_test:.1f}% queries solved)")
    print(f"  Tokens:  {total_t}/{BT_test} ({100*total_t/BT_test:.1f}%)")
    print(f"  Cost:    ${total_c:.5f}/${BC_test:.5f} ({100*total_c/BC_test:.1f}%)")
    print(f"  Load:    {dict(load)}  (caps: {cap_test})")
    print(f"  Feasible: {feasible}")
    return total_q, feasible

# helper function to save a routing assignment to csv
def save_assignment(route, name):
    pd.DataFrame([{'query_id': i, 'agent_id': j, 'token_budget': k,
                   'quality':  q_test[(i, j, k)], 'cost': c_test[(i, j, k)]}
                  for i, (j, k) in route.items()]
                 ).to_csv(f'results/test_{name}.csv', index=False)

# oracle
print("\n[Oracle] Solving IP on test with observed quality...")
test_qs_ids = sorted(test.query_id.unique())

prob = LpProblem("oracle", LpMaximize)
x = {(i, j, k): LpVariable(f"x_{i}_{j}_{k}", cat=LpBinary)
     for i in test_qs_ids for j in agents for k in budgets}

prob += lpSum(q_test[(i, j, k)] * x[(i, j, k)]
              for i in test_qs_ids for j in agents for k in budgets)
for i in test_qs_ids:
    prob += lpSum(x[(i, j, k)] for j in agents for k in budgets) == 1
prob += lpSum(k * x[(i, j, k)]
              for i in test_qs_ids for j in agents for k in budgets) <= BT_test
prob += lpSum(c_test[(i, j, k)] * x[(i, j, k)]
              for i in test_qs_ids for j in agents for k in budgets) <= BC_test
for j in agents:
    prob += lpSum(x[(i, j, k)]
                  for i in test_qs_ids for k in budgets) <= cap_test[j]

t0 = time.time()
prob.solve(PULP_CBC_CMD(msg=False, timeLimit=90))
print(f"IP solve time: {time.time()-t0:.1f}s, Status: {prob.status}")

route_oracle = {i: (j, k) for (i, j, k), v in x.items() if int(v.value()) == 1}
oracle_q, _ = evaluate(route_oracle, "Oracle: IP with observed quality")
save_assignment(route_oracle, 'oracle')

# naive rule
rule_stats = (train.groupby(['query_type', 'agent_id', 'token_budget'])
                   ['observed_quality'].mean().reset_index())

naive_rule = {}
print("\n[Rule (naive)] lookup table (learned from train):")
for qt in query_types:
    sub = rule_stats[rule_stats.query_type == qt]
    row = sub.loc[sub.observed_quality.idxmax()]
    naive_rule[qt] = (row.agent_id, int(row.token_budget))
    print(f"  {qt:15s} -> agent {row.agent_id}, budget {int(row.token_budget)}  "
          f"(train mean q = {row.observed_quality:.3f})")

test_qs = test[['query_id', 'query_type']].drop_duplicates()
route_rule_naive = {r.query_id: naive_rule[r.query_type] for _, r in test_qs.iterrows()}
rule_naive_q, _ = evaluate(route_rule_naive,
                           "Rule (naive): quality-max lookup, constraint-blind")
save_assignment(route_rule_naive, 'rule_naive')

# efficiency rule
stats = train.groupby(['query_type', 'agent_id', 'token_budget']).agg(
    mean_q=('observed_quality', 'mean'),
    mean_c=('cost_usd', 'mean')
).reset_index()
stats['eff'] = stats['mean_q'] / (stats['mean_c'] + 1e-9)

type_options_eff = {}
for qt in query_types:
    sub = stats[stats.query_type == qt].sort_values('eff', ascending=False)
    type_options_eff[qt] = [(r.agent_id, int(r.token_budget), r.mean_q, r.mean_c)
                             for _, r in sub.iterrows()]

route_rule_eff = {}
rem_t, rem_c = BT_test, BC_test
load = defaultdict(int)
for _, r in test_qs.iterrows():
    q_id, qt = r.query_id, r.query_type
    for a, b, mq, mc in type_options_eff[qt]:
        if load[a] < cap_test[a] and b <= rem_t and mc <= rem_c:
            route_rule_eff[q_id] = (a, b)
            load[a] += 1
            rem_t -= b
            rem_c -= c_test[(q_id, a, b)]
            break

rule_eff_q, _ = evaluate(route_rule_eff,
                         "Rule (efficiency): efficiency-ranked lookup")
save_assignment(route_rule_eff, 'rule_eff')

# LP-Batch
train_mean_q = train.groupby(['query_type', 'agent_id', 'token_budget']
                             )['observed_quality'].mean().to_dict()
exp_cost     = train.groupby(['query_type', 'agent_id', 'token_budget']
                             )['cost_usd'].mean().to_dict()
qt_of        = dict(zip(test_qs.query_id, test_qs.query_type))

t0 = time.time()
lp = LpProblem("lp_batch", LpMaximize)

# LP relaxation: continuous variables in [0,1] instead of binary
xl = {(i, j, k): LpVariable(f"xl_{i}_{j}_{k}", 0, 1, cat=LpContinuous)
      for i in test_qs_ids for j in agents for k in budgets}

lp += lpSum(train_mean_q.get((qt_of[i], j, k), 0) * xl[(i, j, k)]
            for i in test_qs_ids for j in agents for k in budgets)
for i in test_qs_ids:
    lp += lpSum(xl[(i, j, k)] for j in agents for k in budgets) == 1
lp += lpSum(k * xl[(i, j, k)]
            for i in test_qs_ids for j in agents for k in budgets) <= BT_test
lp += lpSum(exp_cost.get((qt_of[i], j, k), 1e-3) * xl[(i, j, k)]
            for i in test_qs_ids for j in agents for k in budgets) <= BC_test
for j in agents:
    lp += lpSum(xl[(i, j, k)]
                for i in test_qs_ids for k in budgets) <= cap_test[j]

lp.solve(PULP_CBC_CMD(msg=False))
lp_solve_time = time.time() - t0

# LP-weighted rounding: for each query, pick the option with the highest LP value
route_lp_batch = {}
rem_t, rem_c = BT_test, BC_test
load = defaultdict(int)

q_order = sorted(test_qs_ids,
                 key=lambda i: -max(xl[(i, j, k)].value()
                                    for j in agents for k in budgets))

for i in q_order:
    opts = sorted([(xl[(i, j, k)].value(), j, k)
                   for j in agents for k in budgets], reverse=True)
    for _, a, b in opts:
        ce = exp_cost.get((qt_of[i], a, b), 1e-3)
        if load[a] < cap_test[a] and b <= rem_t and ce <= rem_c:
            route_lp_batch[i] = (a, b)
            load[a] += 1
            rem_t -= b
            rem_c -= c_test[(i, a, b)]
            break

lp_batch_total_time = time.time() - t0
print(f"\nLP-Batch total time: {lp_batch_total_time*1000:.1f}ms "
      f"({lp_batch_total_time*1000/N_test:.2f}ms per query, amortized)")
print(f"  (LP solve: {lp_solve_time*1000:.1f}ms, "
      f"rounding: {(lp_batch_total_time-lp_solve_time)*1000:.1f}ms)")

lp_batch_q, _ = evaluate(route_lp_batch,
                         "LP-Batch: LP relaxation + LP-weighted rounding")
save_assignment(route_lp_batch, 'lp_batch')

# shadow-static
# idea: solve an aggregate LP offline to get shadow prices (dual variables)
# for each shared resource. At routing time, each query is scored by
#     score(i, j, k) = q_est - lam_tok * k - lam_cost * c_est - mu_agent
# and assigned to the highest-scoring feasible option. No LP at runtime.

N_train = train.query_id.nunique()
type_counts = train.groupby('query_type')['query_id'].nunique().to_dict()

BT_train = 1000 * N_train
BC_train = 0.00022 * N_train
cap_train = {'A': int(0.30 * N_train), 'B': int(0.25 * N_train),
             'C': int(0.25 * N_train), 'D': int(0.25 * N_train),
             'E': int(0.20 * N_train)}

print("\n[Shadow-Static] Solving aggregate LP on train to extract shadow prices...")
t0 = time.time()
agg = LpProblem("aggregate_train_lp", LpMaximize)
xa = {(t, j, k): LpVariable(f"xa_{t}_{j}_{k}", lowBound=0, cat=LpContinuous)
      for t in query_types for j in agents for k in budgets}

agg += lpSum(train_mean_q.get((t, j, k), 0) * xa[(t, j, k)]
             for t in query_types for j in agents for k in budgets)

for t in query_types:
    agg += lpSum(xa[(t, j, k)] for j in agents for k in budgets) == type_counts[t], \
           f"demand_{t}"

agg += lpSum(k * xa[(t, j, k)]
             for t in query_types for j in agents for k in budgets) <= BT_train, "tok"
agg += lpSum(exp_cost.get((t, j, k), 1e-3) * xa[(t, j, k)]
             for t in query_types for j in agents for k in budgets) <= BC_train, "cost"
for j in agents:
    agg += lpSum(xa[(t, j, k)]
                 for t in query_types for k in budgets) <= cap_train[j], f"cap_{j}"

agg.solve(PULP_CBC_CMD(msg=False))
offline_time = time.time() - t0

# extract shadow prices
lam_tok  = abs(agg.constraints['tok'].pi  or 0.0)
lam_cost = abs(agg.constraints['cost'].pi or 0.0)
mu       = {j: abs(agg.constraints[f'cap_{j}'].pi or 0.0) for j in agents}

print(f"  Offline LP solved in {offline_time*1000:.1f}ms")
print(f"  lam_tok  = {lam_tok:.6f}   (marginal quality per extra token)")
print(f"  lam_cost = {lam_cost:.2f}      (marginal quality per extra $)")
print(f"  mu_j     = " + ", ".join(f"{j}:{mu[j]:.4f}" for j in agents))

# score each query with the shadow prices, assign to highest-scoring feasible option
route_shadow_static = {}
rem_t, rem_c = BT_test, BC_test
load = defaultdict(int)

t0_online = time.time()
for i in test_qs_ids:                       # queries arrive in id order (stream simulation)
    qt = qt_of[i]
    scored = []
    for j in agents:
        for k in budgets:
            q_est = train_mean_q.get((qt, j, k), 0)
            c_est = exp_cost.get((qt, j, k), 1e-3)
            score = q_est - lam_tok * k - lam_cost * c_est - mu[j]
            scored.append((score, j, k, c_est))
    scored.sort(reverse=True)

    for _, a, b, ce in scored:
        if load[a] < cap_test[a] and b <= rem_t and ce <= rem_c:
            route_shadow_static[i] = (a, b)
            load[a] += 1
            rem_t -= b
            rem_c -= c_test[(i, a, b)]
            break

online_time_static = time.time() - t0_online
print(f"  Online routing: {online_time_static*1000:.2f}ms total, "
      f"{online_time_static*1000/N_test:.3f}ms per query")

shadow_static_q, _ = evaluate(route_shadow_static,
                              "Shadow-Static: static shadow-price routing")
save_assignment(route_shadow_static, 'shadow_static')

# shadow-adaptive
# same offline stage as shadow-static (reuse lam_tok, lam_cost, mu)
# update shadow prices online based on realized resource consumption vs expected schedule

# update rule (multiplicative weights, Arora-Kale style):
#     lam(i+1) = lam(i) * exp( eta * ( consumed_fraction - progress_fraction ) )

ETA = 0.5   # learning rate; chosen via grid search on train

lam_t0, lam_c0, mu0 = lam_tok, lam_cost, dict(mu)

route_shadow_adaptive = {}
rem_t, rem_c = BT_test, BC_test
load = defaultdict(int)

t0_online = time.time()
for idx, i in enumerate(test_qs_ids):
    # dynamically update shadow prices
    progress   = idx / N_test
    tok_used   = 1.0 - rem_t / BT_test
    cost_used  = 1.0 - rem_c / BC_test

    lam_t_now = lam_t0 * np.exp(ETA * (tok_used  - progress))
    lam_c_now = lam_c0 * np.exp(ETA * (cost_used - progress))
    mu_now = {j: mu0[j] * np.exp(ETA * (load[j] / cap_test[j] - progress))
              for j in agents}

    # score each query with the updated shadow prices, assign to highest-scoring feasible option
    qt = qt_of[i]
    scored = []
    for j in agents:
        for k in budgets:
            q_est = train_mean_q.get((qt, j, k), 0)
            c_est = exp_cost.get((qt, j, k), 1e-3)
            score = q_est - lam_t_now * k - lam_c_now * c_est - mu_now[j]
            scored.append((score, j, k, c_est))
    scored.sort(reverse=True)

    for _, a, b, ce in scored:
        if load[a] < cap_test[a] and b <= rem_t and ce <= rem_c:
            route_shadow_adaptive[i] = (a, b)
            load[a] += 1
            rem_t -= b
            rem_c -= c_test[(i, a, b)]
            break

online_time_adaptive = time.time() - t0_online
print(f"\nShadow-Adaptive online routing: {online_time_adaptive*1000:.2f}ms total, "
      f"{online_time_adaptive*1000/N_test:.3f}ms per query")

shadow_adaptive_q, _ = evaluate(route_shadow_adaptive,
                                "Shadow-Adaptive: adaptive primal-dual routing")
save_assignment(route_shadow_adaptive, 'shadow_adaptive')

# summary comparison
print("\n\n" + "=" * 74)
print(f"FINAL COMPARISON (test set, N={N_test})")
print("=" * 74)
print(f"  Oracle              (IP + observed q, reference)  {oracle_q:6.2f}  100.0% of oracle")
print(f"  Rule (naive)        (constraint-blind)            {rule_naive_q:6.2f}  INFEASIBLE")
print(f"  Rule (efficiency)   (streaming baseline)          {rule_eff_q:6.2f}  {100*rule_eff_q/oracle_q:5.1f}% of oracle")
print(f"  LP-Batch            (batch reference)             {lp_batch_q:6.2f}  {100*lp_batch_q/oracle_q:5.1f}% of oracle")
print(f"  Shadow-Static       (streaming)                   {shadow_static_q:6.2f}  {100*shadow_static_q/oracle_q:5.1f}% of oracle")
print(f"  Shadow-Adaptive     (streaming, main proposal)    {shadow_adaptive_q:6.2f}  {100*shadow_adaptive_q/oracle_q:5.1f}% of oracle")
print(f"\n  Shadow-Adaptive vs LP-Batch:      {(shadow_adaptive_q-lp_batch_q)/lp_batch_q*100:+.1f}%")
print(f"  Shadow-Adaptive vs Shadow-Static: {(shadow_adaptive_q-shadow_static_q)/shadow_static_q*100:+.1f}%")
