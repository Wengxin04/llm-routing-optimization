import pandas as pd
import time
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, PULP_CBC_CMD
import os
os.makedirs('results', exist_ok=True)

# load data
df = pd.read_csv('data.csv')
train = df[df.split == 'train'].copy()
N = train.query_id.nunique()

queries = sorted(train.query_id.unique())
agents  = ['A', 'B', 'C', 'D', 'E']
budgets = [512, 1024, 2048, 4096]

q = {}    # observed quality
t = {}    # token budget
c = {}    # cost in USD$
for _, r in train.iterrows():
    key = (r.query_id, r.agent_id, r.token_budget)
    q[key] = r.observed_quality
    t[key] = r.token_budget
    c[key] = r.cost_usd

# capacities per agent
cap = {'A': int(0.30*N), 'B': int(0.25*N), 'C': int(0.25*N),
       'D': int(0.25*N), 'E': int(0.20*N)}

def solve(BT, BC, label):
    print(f"\n=== {label}: BT={BT}, BC=${BC:.5f} ===")
    prob = LpProblem(f"routing_{label}", LpMaximize)

    # decision variables x_{ijk}
    x = {(i, j, k): LpVariable(f"x_{i}_{j}_{k}", cat=LpBinary)
         for i in queries for j in agents for k in budgets}

    prob += lpSum(q[(i,j,k)] * x[(i,j,k)] for i in queries for j in agents for k in budgets), "total_quality"

    # constraint (1) each query assigned exactly once
    for i in queries:
        prob += lpSum(x[(i, j, k)] for j in agents for k in budgets) == 1, f"assign_{i}"

    # constraint (2) total token budget
    prob += lpSum(t[(i,j,k)] * x[(i,j,k)] for i in queries for j in agents for k in budgets) <= BT, "token_budget"

    # constraint (3) total cost budget
    prob += lpSum(c[(i,j,k)] * x[(i,j,k)] for i in queries for j in agents for k in budgets) <= BC, "cost_budget"

    # constraint (4) per-agent capacity
    for j in agents:
        prob += lpSum(x[(i,j,k)] for i in queries for k in budgets) <= cap[j], f"cap_{j}"

    # solve with time limit
    start = time.time()
    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=90))
    print(f"Solve time: {time.time()-start:.1f}s, Status: {prob.status}, "
          f"Objective: {prob.objective.value():.4f}")

    # report
    used_tokens = sum(t[k] * int(v.value()) for k, v in x.items())
    used_cost   = sum(c[k] * int(v.value()) for k, v in x.items())
    agent_load  = {j: sum(int(x[(i,j,k)].value()) for i in queries for k in budgets)
                   for j in agents}
    budget_use  = {b: sum(int(x[(i,j,b)].value()) for i in queries for j in agents)
                   for b in budgets}
    print(f"  tokens used: {used_tokens}/{BT} ({100*used_tokens/BT:.1f}%)")
    print(f"  cost used:   ${used_cost:.5f}/${BC:.5f} ({100*used_cost/BC:.1f}%)")
    print(f"  agent load:  {agent_load}  (caps: {cap})")
    print(f"  budget mix:  {budget_use}")

    # save assignment to csv
    rows = [{'query_id': i, 'agent_id': j, 'token_budget': k,
             'quality': q[(i,j,k)], 'cost': c[(i,j,k)]}
            for (i,j,k), v in x.items() if int(v.value()) == 1]
    pd.DataFrame(rows).sort_values('query_id').to_csv(
        f'results/assignment_{label}.csv', index=False)
    print(f"saved -> results/assignment_{label}.csv")
    return prob, x

# run all three scenarios
if __name__ == "__main__":
    BT_default, BC_default = 1000 * N, 0.00022 * N
    solve(BT_default, BC_default, "default")
    solve(0.7 * BT_default, 0.7 * BC_default, "tight")
    solve(1.5 * BT_default, 1.5 * BC_default, "loose")
