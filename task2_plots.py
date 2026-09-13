import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

Path('figures').mkdir(exist_ok=True)

# style
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        10,
    'axes.labelsize':   10,
    'axes.titlesize':   11,
    'legend.fontsize':  9,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'savefig.dpi':      150,
    'savefig.bbox':     'tight',
})

# load data
data  = pd.read_csv('data.csv')
qtype = data[['query_id', 'query_type']].drop_duplicates()

scenarios       = ['tight', 'default', 'loose']
scenario_labels = {'tight': 'Tight (0.7×)', 'default': 'Default', 'loose': 'Loose (1.5×)'}
objectives      = {'tight': 185.41, 'default': 211.88, 'loose': 218.03}
oracle          = 218.03   # sum of per-query max quality on train (from task2_analysis.py)

assignments = {s: pd.read_csv(f'results/assignment_{s}.csv').merge(qtype, on='query_id')
               for s in scenarios}

# figure 1: Token-budget allocation shifts with resource tightness
budgets = [512, 1024, 2048, 4096]
colors  = ['#c6dbef', '#6baed6', '#2171b5', '#08306b']   # blues, dark = high budget

fig, ax = plt.subplots(figsize=(5.5, 3.2))
bottom = np.zeros(3)
for i, b in enumerate(budgets):
    counts = np.array([assignments[s]['token_budget'].eq(b).sum() for s in scenarios])
    bars = ax.bar(
        [scenario_labels[s] for s in scenarios],
        counts,
        bottom=bottom,
        label=str(b),
        color=colors[i],
        edgecolor='white',
        linewidth=0.6,
    )
    for j, (bar, c) in enumerate(zip(bars, counts)):
        if c >= 8:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom[j] + c / 2,
                str(c),
                ha='center', va='center', fontsize=8,
                color='white' if i >= 2 else '#333',
            )
    bottom += counts

ax.set_ylabel('Number of queries')
ax.set_title('Token-budget allocation shifts with resource tightness')
ax.legend(title='max_tokens', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
plt.savefig('figures/token_budget_mix.png')
plt.close()
print("Saved figures/token_budget_mix.png")

# figure 2: Agent × query_type heatmaps
qtypes_order = ['coding', 'knowledge', 'math_comp', 'math_olympiad', 'multihop_qa']
qtype_short  = {'coding': 'code', 'knowledge': 'know', 'math_comp': 'm_cmp',
                'math_olympiad': 'm_oly', 'multihop_qa': 'm_qa'}

fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
for ax, s in zip(axes, scenarios):
    a = assignments[s]
    pivot = (a.groupby(['agent_id', 'query_type']).size()
              .unstack(fill_value=0)
              .reindex(index=['A', 'B', 'C', 'D', 'E'],
                       columns=qtypes_order, fill_value=0))

    ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=30)
    ax.set_xticks(range(len(qtypes_order)))
    ax.set_xticklabels([qtype_short[q] for q in qtypes_order], rotation=45, ha='right')
    ax.set_yticks(range(5))
    ax.set_yticklabels(['A', 'B', 'C', 'D', 'E'])
    ax.set_title(scenario_labels[s])
    ax.set_xlabel('query type')

    for i in range(5):
        for j in range(len(qtypes_order)):
            val = pivot.iloc[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=8, color='white' if val > 18 else '#333')

axes[0].set_ylabel('Agent')
plt.suptitle('Assignments by agent × query type', y=1.02)
plt.savefig('figures/agent_by_qtype.png')
plt.close()
print("Saved figures/agent_by_qtype.png")
