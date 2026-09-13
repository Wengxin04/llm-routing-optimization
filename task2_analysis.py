import pandas as pd
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, PULP_CBC_CMD

data = pd.read_csv('data.csv')
qtype = data[['query_id', 'query_type']].drop_duplicates()

# oracle upper bound on train data
train = data[data.split == 'train']
oracle = train.groupby('query_id')['observed_quality'].max().sum()
print(f"Oracle upper bound (train): {oracle:.4f}")
print(f"  = sum of per-query max quality, ignoring all constraints\n")

objectives = {'tight': 185.4146, 'default': 211.8813, 'loose': 218.0283}
for name, obj in objectives.items():
    gap = (oracle - obj) / oracle * 100
    print(f"  {name:8s}: {obj:6.4f}  ({gap:4.2f}% below oracle)")

# scenario breakdown
for scenario in ['tight', 'default', 'loose']:
    a = pd.read_csv(f'results/assignment_{scenario}.csv')
    a = a.merge(qtype, on='query_id')
    print(f"\n===== {scenario.upper()} =====")
    print(f"Total quality: {a['quality'].sum():.2f}")
    print("\nAgent × query_type (rows=agent, cols=query_type):")
    print(a.groupby(['agent_id','query_type']).size().unstack(fill_value=0))
    print("\nAgent × token_budget:")
    print(a.groupby(['agent_id','token_budget']).size().unstack(fill_value=0))

# save summary
summary_rows = []
for scenario in ['tight', 'default', 'loose']:
    a = pd.read_csv(f'results/assignment_{scenario}.csv').merge(qtype, on='query_id')
    a['scenario'] = scenario
    summary_rows.append(a)
pd.concat(summary_rows).to_csv('results/all_assignments.csv', index=False)
print("Saved results/all_assignments.csv")

# find the saturation point where the objective reaches the oracle upper bound
train_df = data[data.split == 'train'].copy()
N        = train_df.query_id.nunique()
queries  = sorted(train_df.query_id.unique())
agents   = ['A', 'B', 'C', 'D', 'E']
budgets  = [512, 1024, 2048, 4096]

q = {}; t = {}; c = {}
for _, r in train_df.iterrows():
    key = (r.query_id, r.agent_id, r.token_budget)
    q[key] = r.observed_quality
    t[key] = r.token_budget
    c[key] = r.cost_usd

cap = {'A': int(0.30 * N), 'B': int(0.25 * N), 'C': int(0.25 * N),
       'D': int(0.25 * N), 'E': int(0.20 * N)}
BT_default = 1000 * N
BC_default = 0.00022 * N

def solve_at_multiplier(m):
    """Solve the routing IP with token and cost budgets scaled by m."""
    prob = LpProblem(f"sweep_m_{m}", LpMaximize)
    x = {(i, j, k): LpVariable(f"x_{i}_{j}_{k}", cat=LpBinary)
         for i in queries for j in agents for k in budgets}
    prob += lpSum(q[key] * x[key] for key in x)
    for i in queries:
        prob += lpSum(x[(i, j, k)] for j in agents for k in budgets) == 1
    prob += lpSum(t[key] * x[key] for key in x) <= m * BT_default
    prob += lpSum(c[key] * x[key] for key in x) <= m * BC_default
    for j in agents:
        prob += lpSum(x[(i, j, k)] for i in queries for k in budgets) <= cap[j]
    prob.solve(PULP_CBC_CMD(msg=False, timeLimit=90))
    return prob.objective.value()

print("\n===== Budget-multiplier sweep =====")
print(f"{'multiplier':>10} {'objective':>10} {'% oracle':>9}")
for m in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    obj = solve_at_multiplier(m)
    print(f"{m:>10.2f} {obj:>10.4f} {100*obj/oracle:>8.2f}%")

# realized quality by (agent, query_type) in the tight scenario
print("\n===== Realized quality: agent × query_type (tight scenario) =====")
tight_assign = pd.read_csv('results/assignment_tight.csv').merge(qtype, on='query_id')
realized = (tight_assign.groupby(['agent_id', 'query_type'])['quality']
            .agg(count='count', mean_q='mean')
            .round(3))
print(realized)
