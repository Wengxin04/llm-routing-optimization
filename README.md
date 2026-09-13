# LLM Agent Routing under Token and Cost Budgets

This repository formulates LLM query-to-agent assignment as a 0-1 integer program under token, cost, and per-agent capacity constraints, evaluates it under three budget scenarios, and develops fast streaming methods for real-time deployment. The main methodological contribution is an adaptive primal-dual streaming router (Shadow-Adaptive) that outperforms a batched LP-relaxation baseline by 5.6% at ~0.03 ms per-query decision time while remaining fully streaming.

Full formulation, results, and proposed extension: see [`Wengxin_Xu_summary.pdf`](./Wengxin_Xu_summary.pdf).

---

## Environment

Python 3.10 or later. Install dependencies:

```bash
pip install pulp pandas numpy matplotlib
```

The CBC solver is bundled with PuLP, no separate install required.

Reference environment (used to produce the report numbers): Python 3.14.7, PuLP 3.3.2, pandas 3.0.5, numpy 2.5.3, matplotlib 3.11.2.

---

## Reproducing the report

Run the scripts in the order below. Each writes outputs to `results/` or `figures/` (created automatically). Approximate runtimes on a modern laptop are noted.

### Step 1: Task 2 — solve the IP under three budget scenarios (~4 min)

```bash
python task2_solver.py
```

Solves the routing IP on the train split (N = 266) under default, tight (0.7×), and loose (1.5×) budgets. Writes `results/assignment_{tight,default,loose}.csv`. Prints objective, resource usage, agent load, and token-budget mix per scenario. Reproduces Table 1 in the report.

### Step 2: Task 2 findings (~2 min)

```bash
python task2_analysis.py
```

Reads the assignment CSVs from Step 1 and produces:

- Oracle upper bound and each scenario's gap to it
- Agent × query-type and agent × token-budget breakdowns
- Budget-multiplier sweep (locates the saturation point at m ≈ 1.3, supporting finding (i) in Section 3)
- Realized-quality breakdown by (agent, query-type) in the tight scenario (supports the "~0.20 realized quality" claim in finding (iii))

### Step 3: Task 2 figures (~5 sec)

```bash
python task2_plots.py
```

Reads the assignment CSVs and writes `figures/token_budget_mix.png` and `figures/agent_by_qtype.png` (Figures 1 and 2 in the report).

### Step 4: Task 3 — six routing methods evaluated on test (~5 sec)

```bash
python task3.py
```

Runs Oracle, Rule (naive), Rule (efficiency), LP-Batch, Shadow-Static, and Shadow-Adaptive on the held-out test split (N = 109). Prints per-method quality, resource usage, agent load, feasibility, and decision latency. Writes `results/test_{method}.csv` for each method. Reproduces Table 2 in the report.

### Step 5: Task 3 hyperparameter tuning (~5 sec)

```bash
python task3_eta_search.py
```

Grid search over η ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0} for the Shadow-Adaptive multiplicative-weights update, evaluated on train. Supports the choice of η = 0.5 in the report.

---

## Repository structure

| File | Purpose | Supports report |
|---|---|---|
| `task2_solver.py` | Solves the routing IP under three budget scenarios | Task 1-2, Table 1 |
| `task2_analysis.py` | Findings, multiplier sweep, realized-quality breakdown | Task 2 findings (i), (iii) |
| `task2_plots.py` | Generates Figures 1 and 2 | Task 2 figures |
| `task3.py` | Six routing methods, evaluated on test | Task 3, Table 2 |
| `task3_eta_search.py` | Grid search for MW learning rate η | Task 3, choice of η = 0.5 |
| `data.csv` | Dataset: 375 queries × 5 agents × 4 token budgets, with quality / cost / latency | input |
| `queries.jsonl` | Prompt text and reference answers for all queries | input |
| `Wengxin_Xu_summary.pdf` | Summary report (formulation, results, extension) | — |

Auto-generated directories:

- `results/` — assignment CSVs from `task2_solver.py` and `task3.py`
- `figures/` — PNGs from `task2_plots.py`

---

## Notes

- All methods are deterministic. No random seeds are used or required.
- The CBC solve time limit is 90 seconds on Task 1–2 instances; the returned solutions are optimal or within a very small gap.
- Reported per-query decision latencies (fractions of a millisecond) may vary by 5–10% across runs due to system-level noise. The numbers in the report are from a single reference run.

---

## AI tools disclosure

Claude (Anthropic) was used interactively during development for concept discussion, code review, and prose editing. All formulations, algorithm designs, and numerical results were verified by the author before submission.
