import numpy as np
import pandas as pd
from collections import defaultdict
from pulp import (LpProblem, LpVariable, LpMaximize, lpSum,
                  LpContinuous, PULP_CBC_CMD)

# data processing
df = pd.read_csv('data.csv')
train = df[df.split == 'train'].copy()

N       = train.query_id.nunique()
agents  = ['A', 'B', 'C', 'D', 'E']
budgets = [512, 1024, 2048, 4096]
qtypes  = sorted(train.query_type.unique())

BT  = 1000 * N
BC  = 0.00022 * N
cap = {'A': int(0.30*N), 'B': int(0.25*N), 'C': int(0.25*N),
       'D': int(0.25*N), 'E': int(0.20*N)}

q_tr = {(r.query_id, r.agent_id, r.token_budget): r.observed_quality
        for _, r in train.iterrows()}
c_tr = {(r.query_id, r.agent_id, r.token_budget): r.cost_usd
        for _, r in train.iterrows()}
qt_of = dict(zip(train.query_id, train.query_type))
ids   = sorted(train.query_id.unique())

train_mean_q = train.groupby(['query_type','agent_id','token_budget']
                             )['observed_quality'].mean().to_dict()
exp_cost     = train.groupby(['query_type','agent_id','token_budget']
                             )['cost_usd'].mean().to_dict()
type_counts  = train.groupby('query_type')['query_id'].nunique().to_dict()

# repeat the aggregate LP from task3.py
agg = LpProblem("agg", LpMaximize)
xa = {(t, j, k): LpVariable(f"xa_{t}_{j}_{k}", 0, cat=LpContinuous)
      for t in qtypes for j in agents for k in budgets}

agg += lpSum(train_mean_q.get((t, j, k), 0) * xa[(t, j, k)]
             for t in qtypes for j in agents for k in budgets)
for t in qtypes:
    agg += lpSum(xa[(t, j, k)] for j in agents for k in budgets) == type_counts[t], f"d_{t}"
agg += lpSum(k * xa[(t, j, k)]
             for t in qtypes for j in agents for k in budgets) <= BT, "tok"
agg += lpSum(exp_cost.get((t, j, k), 1e-3) * xa[(t, j, k)]
             for t in qtypes for j in agents for k in budgets) <= BC, "cost"
for j in agents:
    agg += lpSum(xa[(t, j, k)]
                 for t in qtypes for k in budgets) <= cap[j], f"cap_{j}"

# to get the dual variables
agg.solve(PULP_CBC_CMD(msg=False))
lam_t0 = abs(agg.constraints['tok'].pi  or 0.0)
lam_c0 = abs(agg.constraints['cost'].pi or 0.0)
mu0    = {j: abs(agg.constraints[f'cap_{j}'].pi or 0.0) for j in agents}

# grid search for eta in Shadow-Adaptive
def run_shadow_adaptive(eta):
    route = {}
    rem_t, rem_c = BT, BC
    load = defaultdict(int)
    for idx, i in enumerate(ids):
        p  = idx / N
        ct = 1.0 - rem_t / BT
        cc = 1.0 - rem_c / BC
        lam_t = lam_t0 * np.exp(eta * (ct - p))
        lam_c = lam_c0 * np.exp(eta * (cc - p))
        mu    = {j: mu0[j] * np.exp(eta * (load[j] / cap[j] - p)) for j in agents}

        qt = qt_of[i]
        scored = []
        for j in agents:
            for k in budgets:
                q_est = train_mean_q.get((qt, j, k), 0)
                c_est = exp_cost.get((qt, j, k), 1e-3)
                s = q_est - lam_t * k - lam_c * c_est - mu[j]
                scored.append((s, j, k, c_est))
        scored.sort(reverse=True)

        for _, a, b, ce in scored:
            if load[a] < cap[a] and b <= rem_t and ce <= rem_c:
                route[i] = (a, b)
                load[a] += 1
                rem_t -= b
                rem_c -= c_tr[(i, a, b)]
                break
    total_q = sum(q_tr[(i, a, b)] for i, (a, b) in route.items())
    return total_q

print("===== Grid search: eta for Shadow-Adaptive (evaluated on train) =====")
print(f"{'eta':>5}  {'train Q':>10}")
for eta in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
    tq = run_shadow_adaptive(eta)
    print(f"{eta:>5.1f}  {tq:>10.2f}")
