# Course Scheduling with Constraint Optimization

---

## Project Overview
This project implements a complete **three-stage heuristic pipeline** for university course scheduling.

The goal is to generate feasible schedules while minimizing soft constraint violations.

The system includes:

1. **Baseline CSP Solver**
   - Backtracking Search
   - MRV (Minimum Remaining Values)
   - LCV (Least Constraining Value)
   - Forward Checking

2. **Enhancement 1**
   - Iterative Deepening with Random Restarts
   - Randomized MRV tie-breaking
   - Backtrack cutoffs
   - Solution diversity

3. **Enhancement 2**
   - Guided Local Search (GLS)
   - Adaptive penalty weights
   - Local minima escape
   - Soft constraint optimization

---

## Folder Structure
```text
CourseSchedulingHeuristics/
│
├── scheduler.py
├── enhancement1_restarts.py
├── enhancement2_gls.py
├── experiments_restart.py
├── experiments_gls.py
├── constraint_checker.py
├── data_loader.py
├── display.py
├── requirements.txt
├── README.md
│
├── figures/
│   ├── baseline_vs_enh1.png
│   ├── scaling_results.png
│   ├── diversity_results.png
│   ├── gls_convergence_curve.png
│   └── gls_results_comparison.png
```

---

## Requirements
- matplotlib>=3.7.0
- numpy>=1.24.0

Install dependencies:

```
Check for requirements.txt
```

---

## How to Run

### Baseline CSP
```bash
python scheduler.py
```

---

### Enhancement 1
```bash
python enhancement1_restarts.py
```

---

### Enhancement 2 (GLS Pipeline)
```bash
python enhancement2_gls.py
```

---

### All Experiments
```bash
python experiments.py
```

---

### GLS Experiments + Graphs
```bash
python experiments_gls.py
```

This generates additional figures in the `figures/` folder.

---

## Test Cases
The project includes multiple datasets:

- **TC1 Easy**
- **TC2 Medium**
- **TC3 Hard**
- **Hard 100 Course Dataset**

The `hard_100.json` dataset is specifically designed to demonstrate clear performance differences between:

- Baseline
- Enhancement 1
- Enhancement 2

---

## Output Figures
The `figures` folder includes:

- baseline vs enhancement comparison
- scaling analysis
- solution diversity
- GLS convergence graph
- GLS result comparison graph

---

## Experimental Highlights
- **96.7% reduction** in backtracking  
  `1226 → 41`

- **72.2% improvement** in GLS optimization on TC1

- **63.5% improvement** on TC2

- Hard 100 dataset added for large-scale evaluation

---

## Future Work
- AC-3 / MAC propagation
- Simulated Annealing
- ITC benchmark datasets
- Larger real-world university scheduling data