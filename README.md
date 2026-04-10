# CourseSchedulingHeuristics
# Course Scheduling with Constraint Optimization
## CS 57200: Heuristic Problem Solving — Project Milestone 2
### Deekshitha Reddy Muppidi | Track B: Optimization / Planning

---

## Project Overview

This project implements a course scheduling system using Constraint
Satisfaction Problem (CSP) techniques and heuristic search. Given a
set of courses, rooms, timeslots, and instructors, the solver finds a
complete schedule satisfying all hard constraints.

---

## File Structure

```
course_scheduler/
├── scheduler.py              # BASELINE — Backtracking CSP (MRV + LCV + FC)
├── enhancement1_restarts.py  # ENHANCEMENT 1 — Iterative Deepening with Restarts
├── experiments.py            # M2 Experimental Evaluation (all 3 experiments)
├── constraint_checker.py     # Hard constraint validation (H1-H5)
├── data_loader.py            # Problem loader + 3 test cases (small/medium/large)
├── display.py                # Schedule and stats output
├── requirements.txt          # No external dependencies
└── README.md                 # This file
```

---

## Requirements

- Python 3.7 or higher
- No external libraries required — pure Python standard library

---

## How to Run

### Run the Baseline

```bash
python scheduler.py
```

Runs the deterministic CSP solver on the small test case (10 courses).
Outputs the schedule, solver stats, and hard constraint validation.

### Run Enhancement 1

```bash
python enhancement1_restarts.py
```

Runs Iterative Deepening with Restarts on the small test case.
Shows per-run table with seeds, backtracks, and results.

### Run All M2 Experiments

```bash
python experiments.py
```

Runs all 3 experiments across all 3 test cases:
- Experiment 1: Baseline vs Enhancement 1
- Experiment 2: Performance scaling (10, 20, 30 courses)
- Experiment 3: Restart parameter sensitivity

---

## Algorithms Implemented

### Baseline — Backtracking CSP (`scheduler.py`)
Implements BACKTRACKING-SEARCH from AIMA 4th Ed., Fig 6.5 with:
- **MRV**: Minimum Remaining Values — assign most constrained course first
- **LCV**: Least Constraining Value — try least constraining triple first
- **FC**: Forward Checking — prune domains after each assignment

Path cost: number of course assignments made (one unit per course).

### Enhancement 1 — Iterative Deepening with Restarts (`enhancement1_restarts.py`)
Runs the CSP solver multiple times with:
- **Randomized MRV tie-breaking** — different search path each run
- **Backtrack cutoff** — abandons unproductive runs early
- **Best solution tracking** — keeps best result across all runs

Path cost: total backtracks accumulated across all restart runs.

---

## Hard Constraints Enforced (H1-H5)

| ID  | Constraint            | Description                                          |
|-----|-----------------------|------------------------------------------------------|
| H1  | Instructor Conflict   | No instructor teaches two courses at the same time   |
| H2  | Room Conflict         | No room hosts two courses at the same time           |
| H3  | Room Capacity         | Room capacity >= course enrollment                   |
| H4  | Qualification         | Instructor must be qualified to teach the course     |
| H5  | Curriculum Conflict   | No student group has two courses at the same time    |

---

## Test Cases

| Test Case | Courses | Rooms | Timeslots | Instructors | Curricula |
|-----------|---------|-------|-----------|-------------|-----------|
| Small     | 10      | 4     | 10        | 4           | 3         |
| Medium    | 20      | 6     | 15        | 5           | 4         |
| Large     | 30      | 8     | 20        | 6           | 5         |

---

## References

1. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. Chapters 4 & 6.
2. Schaerf, A. (1999). A survey of automated timetabling. *Artificial Intelligence Review*, 13(2), 87–127.
3. Bonutti et al. (2012). Benchmarking curriculum-based course timetabling. *Annals of Operations Research*, 194(1), 59–70.
4. Muller, T. (2009). ITC-2007 solver description. *Journal of Scheduling*, 12(4), 397–409.
5. ITC-2007 Dataset: http://www.cs.qub.ac.uk/itc2007
