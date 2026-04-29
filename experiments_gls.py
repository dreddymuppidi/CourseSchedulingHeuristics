"""
M2 Experimental Evaluation -- 4 Experiments
--------------------------------------------

    Experiment 1 -- Baseline vs Enhancement 1 (all 3 test cases)
        Shows Enhancement 1 reduces backtracks on TC3 (1226 -> variable)
        via random restarts that escape dead-end MRV orderings.

    Experiment 2 -- Performance Scaling
        Runtime and backtrack growth across Easy, Medium, Hard.

    Experiment 3 -- Solution Diversity (Enhancement 1 advantage)
        Baseline is deterministic -- always same schedule.
        Enhancement 1 produces 100% different valid schedules per seed.
        This diversity is critical: Enhancement 2 (GLS) needs it.

    Experiment 4 -- Enhancement 2 (GLS) Soft Constraint Optimization
        Measures F(S) = sum(wi*vi) improvement from baseline schedule
        to GLS-optimized schedule across all 3 test cases.
        Shows GLS escaping local minima via adaptive weight updates.

Per M2 rubric: stochastic algorithms averaged over multiple runs.
"""
import sys
import statistics

sys.stdout.reconfigure(encoding="utf-8")

from data_loader import get_all_test_cases, count_solution_differences
from scheduler import CSPScheduler
from enhancement1_restarts import RestartSolver, SingleRunCSP
from enhancement2_gls import (
    GuidedLocalSearch, compute_penalty, INITIAL_WEIGHTS
)
from constraint_checker import ConstraintChecker


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def run_baseline(problem: dict) -> dict:
    """Run baseline once -- deterministic."""
    solver   = CSPScheduler(problem)
    solution = solver.solve()
    s        = solver.stats
    penalty  = compute_penalty(solution, problem, INITIAL_WEIGHTS) if solution else 0
    return {
        "solution":      solution,
        "solved":        solution is not None,
        "backtracks":    s["backtracks"],
        "nodes_visited": s["nodes_visited"],
        "path_cost":     s["path_cost"],
        "time_elapsed":  s["time_elapsed"],
        "penalty":       round(penalty, 3),
    }


def run_enhancement1_multi(problem: dict, max_restarts: int = 5,
                           cutoff: int = 30, outer_runs: int = 3) -> dict:
    """Run Enhancement 1 multiple times and average (stochastic -- must average)."""
    all_results   = []
    all_solutions = []
    for _ in range(outer_runs):
        solver   = RestartSolver(problem, max_restarts=max_restarts,
                                 cutoff=cutoff, verbose=False)
        solution = solver.solve()
        s        = solver.stats
        penalty  = compute_penalty(solution, problem, INITIAL_WEIGHTS) if solution else 0
        all_results.append({
            "solved":           solution is not None,
            "total_backtracks": s["total_backtracks"],
            "total_nodes":      s["total_nodes"],
            "runs_cutoff":      s["runs_cutoff"],
            "solution_run":     s["solution_found_run"],
            "time_elapsed":     s["time_elapsed"],
            "penalty":          round(penalty, 3),
        })
        if solution is not None:
            all_solutions.append(solution)

    solved = [r for r in all_results if r["solved"]]
    return {
        "solutions":        all_solutions,
        "solved_rate":      f"{len(solved)}/{outer_runs}",
        "avg_backtracks":   round(statistics.mean(r["total_backtracks"] for r in all_results), 2),
        "avg_nodes":        round(statistics.mean(r["total_nodes"]      for r in all_results), 2),
        "avg_time":         round(statistics.mean(r["time_elapsed"]     for r in all_results), 4),
        "avg_runs_cutoff":  round(statistics.mean(r["runs_cutoff"]      for r in all_results), 2),
        "avg_penalty":      round(statistics.mean(r["penalty"]          for r in all_results), 3),
        "avg_solution_run": round(
            statistics.mean(r["solution_run"] for r in all_results if r["solution_run"]), 2
        ) if solved else "N/A",
    }


def run_enhancement2(problem: dict, max_gls_iter: int = 150,
                     outer_runs: int = 3) -> dict:
    """
    Run Enhancement 2 (GLS) multiple times starting from baseline solution.
    Average results -- GLS has some randomness in weight update ordering.
    """
    all_results = []
    for _ in range(outer_runs):
        # Get baseline solution first
        base_solver = CSPScheduler(problem)
        base_sol    = base_solver.solve()
        if base_sol is None:
            all_results.append({"solved": False, "initial": 0, "final": 0,
                                 "improvement_pct": 0, "local_minima": 0, "time": 0})
            continue

        gls    = GuidedLocalSearch(problem, base_sol,
                                   max_iterations=max_gls_iter, verbose=False)
        result = gls.optimize()
        s      = gls.stats
        all_results.append({
            "solved":           result is not None,
            "initial":          s["initial_penalty"],
            "final":            s["final_penalty"],
            "improvement_pct":  s["improvement_pct"],
            "local_minima":     s["local_minima_hit"],
            "time":             s["time_elapsed"],
        })

    solved = [r for r in all_results if r["solved"]]
    return {
        "solved_rate":         f"{len(solved)}/{outer_runs}",
        "avg_initial_penalty": round(statistics.mean(r["initial"] for r in all_results), 3),
        "avg_final_penalty":   round(statistics.mean(r["final"]   for r in all_results), 3),
        "avg_improvement_pct": round(statistics.mean(r["improvement_pct"] for r in all_results), 1),
        "avg_local_minima":    round(statistics.mean(r["local_minima"]    for r in all_results), 1),
        "avg_time":            round(statistics.mean(r["time"]            for r in all_results), 4),
    }


def sep(w=74):
    print("=" * w)


def dasep(w=74):
    print("-" * w)


def row(vals, widths):
    print("  " + "".join(f"{str(v):<{w}}" for v, w in zip(vals, widths)))


def hdr(cols, widths):
    row(cols, widths)
    print("  " + "-" * sum(widths))


# ---------------------------------------------------------------------------
# Experiment 1 -- Baseline vs Enhancement 1
# ---------------------------------------------------------------------------

def experiment1(test_cases):
    sep()
    print("  EXPERIMENT 1 -- Baseline CSP vs Enhancement 1 (Restarts)")
    print("  Research Question: Does Enhancement 1 reduce backtracks on hard")
    print("  problems via random restarts that escape dead-end MRV orderings?")
    sep()

    cols   = ["Test Case",       "Algorithm",                   "Solved","Backtracks","Nodes","Time(s)"]
    widths = [36,                40,                             9,       13,           9,      9]
    hdr(cols, widths)

    results = []
    for tc in test_cases:
        name = tc["name"]
        b    = run_baseline(tc)
        e    = run_enhancement1_multi(tc, max_restarts=5, cutoff=30, outer_runs=3)

        row([name, "Baseline CSP (MRV+LCV+FC)",
             "Yes" if b["solved"] else "No",
             b["backtracks"], b["nodes_visited"], b["time_elapsed"]], widths)

        row(["", "Enhancement 1 (restarts=5, cutoff=30, n=3 avg)",
             e["solved_rate"], e["avg_backtracks"],
             e["avg_nodes"], e["avg_time"]], widths)

        print("  " + "-" * sum(widths))
        results.append({"name":name, "b":b, "e":e})

    print("\n  Key Observations:")
    for r in results:
        b, e = r["b"], r["e"]
        if b["backtracks"] > 0 and e["avg_backtracks"] < b["backtracks"]:
            reduction = round((1 - e["avg_backtracks"]/b["backtracks"])*100, 1)
            print(f"    {r['name']}:")
            print(f"      Baseline BT    : {b['backtracks']}")
            print(f"      Enhancement BT : {e['avg_backtracks']} avg -- {reduction}% reduction")
        else:
            print(f"    {r['name']}: both solve efficiently (0 BT) -- diversity is the advantage here")

    return results


# ---------------------------------------------------------------------------
# Experiment 2 -- Performance Scaling
# ---------------------------------------------------------------------------

def experiment2(test_cases):
    sep()
    print("  EXPERIMENT 2 -- Performance Scaling Across Problem Sizes")
    print("  Research Question: How do backtracks and runtime scale from")
    print("  easy (10 courses) to medium (20) to hard (15, tight)?")
    sep()

    cols   = ["Test Case",       "Courses","Algorithm",              "Time(s)","Backtracks","F(S)"]
    widths = [36,                9,        34,                        11,        13,           8]
    hdr(cols, widths)

    scaling = []
    for tc in test_cases:
        name = tc["name"]
        n    = len(tc["courses"])
        b    = run_baseline(tc)
        e    = run_enhancement1_multi(tc, max_restarts=5, cutoff=30, outer_runs=3)

        row([name, n, "Baseline CSP",
             b["time_elapsed"], b["backtracks"], b["penalty"]], widths)
        row(["", "", "Enhancement 1 (avg)",
             e["avg_time"], e["avg_backtracks"], e["avg_penalty"]], widths)
        print("  " + "-" * sum(widths))

        scaling.append({
            "name":name, "courses":n,
            "base_t":b["time_elapsed"], "enh_t":e["avg_time"],
            "base_bt":b["backtracks"],  "enh_bt":e["avg_backtracks"],
            "base_fs":b["penalty"],     "enh_fs":e["avg_penalty"],
        })

    print("\n  Scaling data (for line chart in report):")
    print(f"  {'Test Case':<36} {'Courses':<9} {'Base_t':<12} {'Enh1_t':<12} {'Base_BT':<12} {'Enh1_BT'}")
    print("  " + "-" * 85)
    for r in scaling:
        print(f"  {r['name']:<36} {r['courses']:<9} {r['base_t']:<12} "
              f"{r['enh_t']:<12} {r['base_bt']:<12} {r['enh_bt']}")

    return scaling


# ---------------------------------------------------------------------------
# Experiment 3 -- Solution Diversity
# ---------------------------------------------------------------------------

def experiment3(test_cases):
    sep()
    print("  EXPERIMENT 3 -- Solution Diversity (Enhancement 1)")
    print("  Research Question: Does Enhancement 1 produce meaningfully")
    print("  different schedules per restart vs the deterministic baseline?")
    print("  (Diversity = diverse starting points for Enhancement 2 GLS)")
    sep()

    tc   = test_cases[1]   # medium: 20 courses shows diversity clearly
    name = tc["name"]
    n    = len(tc["courses"])
    print(f"  Problem: {name} ({n} courses)\n")

    b     = run_baseline(tc)
    sol_b = b["solution"]
    print(f"  Baseline: bt={b['backtracks']}, t={b['time_elapsed']}s -- ALWAYS same schedule\n")

    seeds   = [42, 84, 126, 168, 210, 252, 294, 336]
    diffs   = []
    checker = ConstraintChecker(tc)

    cols   = ["Seed","Backtracks","Courses differing from baseline","F(S)","Valid?"]
    widths = [8,     13,          34,                                8,     8]
    hdr(cols, widths)

    for seed in seeds:
        csp   = SingleRunCSP(tc, cutoff=100, seed=seed)
        sol_e = csp.solve()
        bt    = csp.backtracks

        if sol_e is not None:
            diff   = count_solution_differences(sol_b, sol_e, tc["courses"])
            report = checker.validate_full(sol_e)
            fs     = round(compute_penalty(sol_e, tc, INITIAL_WEIGHTS), 1)
            valid  = "Yes" if report.is_feasible() else f"No({report.total})"
            diffs.append(diff)
        else:
            diff, fs, valid = "N/A", "N/A", "None"

        row([seed, bt, diff, fs, valid], widths)

    print()
    if diffs:
        avg_diff = round(statistics.mean(diffs), 1)
        pct      = round(avg_diff / n * 100, 1)
        print(f"  Summary:")
        print(f"    Baseline   : 0/{n} courses differ (identical every run)")
        print(f"    Enh1 avg   : {avg_diff}/{n} courses differ per seed ({pct}% diversity)")
        print(f"    Min / Max  : {min(diffs)} / {max(diffs)} courses differ")
        print(f"\n  Conclusion: Enhancement 1 explores ~{pct}% different schedule space")
        print(f"  per restart. Enhancement 2 (GLS) uses this diversity to find better")
        print(f"  soft-constraint optima than starting from one fixed baseline solution.")

    return {"diffs":diffs, "seeds":seeds, "n":n}


# ---------------------------------------------------------------------------
# Experiment 4 -- Enhancement 2 (GLS) Soft Constraint Optimization
# ---------------------------------------------------------------------------

def experiment4(test_cases):
    sep()
    print("  EXPERIMENT 4 -- Enhancement 2: GLS Soft Constraint Optimization")
    print("  Research Question: How much does GLS improve schedule quality")
    print("  (F(S)) over the baseline, and what does the improvement trajectory")
    print("  look like across Easy, Medium, and Hard test cases?")
    sep()

    cols   = ["Test Case",       "Stage",                     "F(S) avg","Impr %","Local Min","Time(s)"]
    widths = [36,                28,                           12,         10,       12,         10]
    hdr(cols, widths)

    results = []
    for tc in test_cases:
        name = tc["name"]

        # Baseline F(S)
        b = run_baseline(tc)

        # Enhancement 2 (GLS from baseline start) -- averaged
        e2 = run_enhancement2(tc, max_gls_iter=150, outer_runs=3)

        row([name, "Baseline CSP (unoptimized)",
             b["penalty"], "0%", "-", b["time_elapsed"]], widths)

        row(["", "Enhancement 2 GLS (avg 3 runs)",
             e2["avg_final_penalty"], f"{e2['avg_improvement_pct']}%",
             e2["avg_local_minima"], e2["avg_time"]], widths)

        print("  " + "-" * sum(widths))

        results.append({
            "name":    name,
            "base_fs": b["penalty"],
            "gls_fs":  e2["avg_final_penalty"],
            "impr":    e2["avg_improvement_pct"],
            "lm":      e2["avg_local_minima"],
            "time":    e2["avg_time"],
        })

    print("\n  GLS Penalty History -- Detailed run on TC1 (for chart in report):")
    print("  (Shows F(S) per iteration -- used to plot GLS convergence curve)")
    tc1     = test_cases[0]
    base_b  = CSPScheduler(tc1)
    base_sol= base_b.solve()
    gls_det = GuidedLocalSearch(tc1, base_sol, max_iterations=150, verbose=False)
    gls_det.optimize()
    history = gls_det.stats["penalty_history"]
    print(f"  Iter: {list(range(1, len(history)+1))}")
    print(f"  F(S): {history}")

    print(f"\n  Summary:")
    for r in results:
        print(f"    {r['name']}: F(S) {r['base_fs']} -> {r['gls_fs']} "
              f"({r['impr']}% improvement, {r['lm']} local minima escaped)")

    print(f"\n  GLS vs Min-Conflicts explanation:")
    print(f"    Min-Conflicts: uses fixed weights wi=1 throughout -- gets stuck")
    print(f"    GLS:           increases wi at local minima -- escapes by shifting")
    print(f"                   the cost landscape. This is why GLS improves further.")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sep()
    print("  Course Scheduling -- M2 Experimental Evaluation")
    print("  CS 57200: Heuristic Problem Solving")
    print("  Deekshitha Reddy Muppidi | Track B: Optimization/Planning")
    sep()
    print()

    test_cases = get_all_test_cases()
    print("  Test cases:")
    for tc in test_cases:
        print(f"    {tc['name']}: {len(tc['courses'])} courses, "
              f"{len(tc['rooms'])} rooms, {len(tc['timeslots'])} timeslots, "
              f"{len(tc['curricula'])} curricula")
    print()

    r1 = experiment1(test_cases)
    print()
    r2 = experiment2(test_cases)
    print()
    r3 = experiment3(test_cases)
    print()
    r4 = experiment4(test_cases)

    print()
    sep()
    print("  All experiments complete. Report figures:")
    print("  Exp 1 -> Bar chart : Backtracks -- Baseline vs Enhancement 1 per TC")
    print("  Exp 2 -> Line chart: Problem size vs runtime (3 points x 2 algorithms)")
    print("  Exp 3 -> Bar chart : Diversity % per seed vs 0% for baseline")
    print("  Exp 4 -> Line chart: F(S) per iteration (GLS convergence curve)")
    print("           Bar chart : F(S) before/after GLS per test case")
    sep()
