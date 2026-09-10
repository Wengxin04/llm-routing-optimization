import pandas as pd

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
