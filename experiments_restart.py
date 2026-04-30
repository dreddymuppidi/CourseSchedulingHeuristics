"""
experiments.py
==============
CS 57200: Heuristic Problem Solving -- Project Milestone 2
Track B: Optimization / Planning
Student: Deekshitha Reddy Muppidi

M2 Experimental Evaluation
---------------------------
Three experiments demonstrating the difference between the
Baseline CSP solver and Enhancement 1 (Iterative Deepening
with Restarts).

Includes automatic figure generation for M2 report.
"""

import sys
import statistics
import os

# Try to import matplotlib, warn if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Run 'pip install matplotlib' to generate figures.")

sys.stdout.reconfigure(encoding="utf-8")

from data_loader import get_all_test_cases, count_solution_differences
from scheduler import CSPScheduler
from enhancement1_restarts import RestartSolver, SingleRunCSP
from constraint_checker import ConstraintChecker


# ---------------------------------------------------------------------------
# Figure output directory
# ---------------------------------------------------------------------------

FIGURES_DIR = "figures"

def ensure_figures_dir():
    """Create figures directory if it doesn't exist."""
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
        print(f"[figures] Created directory: {FIGURES_DIR}")


# ---------------------------------------------------------------------------
# Graph Generation Functions
# ---------------------------------------------------------------------------

def generate_experiment1_chart(results, save=True):
    """
    Generate bar chart for Experiment 1:
    Baseline vs Enhancement 1 - Backtracks and Time
    
    Creates two subplots:
        - Left: Backtracks comparison
        - Right: Time comparison
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[figures] Skipping Experiment 1 chart - matplotlib not installed")
        return None
    
    ensure_figures_dir()
    
    test_names = [r["name"].split("(")[0].strip() for r in results]
    base_backtracks = [r["b"]["backtracks"] for r in results]
    enh_backtracks = [r["e"]["avg_backtracks"] for r in results]
    base_time = [r["b"]["time_elapsed"] for r in results]
    enh_time = [r["e"]["avg_time"] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: Backtracks
    x = range(len(test_names))
    width = 0.35
    
    bars1 = ax1.bar([i - width/2 for i in x], base_backtracks, width, 
                    label='Baseline CSP', color='#2E86AB', edgecolor='black')
    bars2 = ax1.bar([i + width/2 for i in x], enh_backtracks, width,
                    label='Enhancement 1 (Restarts)', color='#A23B72', edgecolor='black')
    
    ax1.set_ylabel('Number of Backtracks', fontsize=11)
    ax1.set_title('Experiment 1: Backtrack Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=15, ha='right')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, base_backtracks):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(base_backtracks)*0.01,
                f'{val}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, enh_backtracks):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(base_backtracks)*0.01,
                f'{val}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 2: Time
    bars3 = ax2.bar([i - width/2 for i in x], base_time, width,
                    label='Baseline CSP', color='#2E86AB', edgecolor='black')
    bars4 = ax2.bar([i + width/2 for i in x], enh_time, width,
                    label='Enhancement 1 (Restarts)', color='#A23B72', edgecolor='black')
    
    ax2.set_ylabel('Time (seconds)', fontsize=11)
    ax2.set_title('Experiment 1: Runtime Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, rotation=15, ha='right')
    ax2.legend(loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars3, base_time):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(base_time)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars4, enh_time):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(base_time)*0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, "experiment1_backtracks_time.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"[figures] Saved: {filepath}")
    
    plt.show()
    return fig


def generate_experiment2_chart(scaling_data, save=True):
    """
    Generate line chart for Experiment 2:
    Performance scaling across problem sizes
    
    Shows how runtime grows as number of courses increases.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[figures] Skipping Experiment 2 chart - matplotlib not installed")
        return None
    
    ensure_figures_dir()
    
    # Extract data in order of increasing courses
    sorted_data = sorted(scaling_data, key=lambda x: x["courses"])
    courses = [d["courses"] for d in sorted_data]
    base_times = [d["base_t"] for d in sorted_data]
    enh_times = [d["enh_t"] for d in sorted_data]
    base_bts = [d["base_bt"] for d in sorted_data]
    enh_bts = [d["enh_bt"] for d in sorted_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Subplot 1: Time scaling
    ax1.plot(courses, base_times, 'o-', label='Baseline CSP', 
             color='#2E86AB', linewidth=2, markersize=8)
    ax1.plot(courses, enh_times, 's-', label='Enhancement 1 (Restarts)', 
             color='#A23B72', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Courses', fontsize=11)
    ax1.set_ylabel('Time (seconds)', fontsize=11)
    ax1.set_title('Experiment 2: Runtime Scaling', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for x, y in zip(courses, base_times):
        ax1.annotate(f'{y:.3f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    for x, y in zip(courses, enh_times):
        ax1.annotate(f'{y:.3f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # Subplot 2: Backtrack scaling
    ax2.plot(courses, base_bts, 'o-', label='Baseline CSP', 
             color='#2E86AB', linewidth=2, markersize=8)
    ax2.plot(courses, enh_bts, 's-', label='Enhancement 1 (Restarts)', 
             color='#A23B72', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Courses', fontsize=11)
    ax2.set_ylabel('Number of Backtracks', fontsize=11)
    ax2.set_title('Experiment 2: Backtrack Scaling', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    for x, y in zip(courses, base_bts):
        ax2.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    for x, y in zip(courses, enh_bts):
        ax2.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, "experiment2_scaling.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"[figures] Saved: {filepath}")
    
    plt.show()
    return fig


def generate_experiment3_chart(diversity_data, save=True):
    """
    Generate bar chart for Experiment 3:
    Solution Diversity - Enhancement 1 vs Baseline
    
    Shows how many courses differ from baseline across different seeds.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[figures] Skipping Experiment 3 chart - matplotlib not installed")
        return None
    
    ensure_figures_dir()
    
    diffs = diversity_data["diffs"]
    seeds = diversity_data["seeds"]
    n_courses = diversity_data["n_courses"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar chart for diversity across seeds
    colors = plt.cm.viridis([i/len(seeds) for i in range(len(seeds))])
    bars = ax.bar([str(s) for s in seeds], diffs, color=colors, edgecolor='black')
    
    # Add baseline bar (always 0 differences from itself)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Baseline (deterministic)')
    
    ax.set_xlabel('Random Seed', fontsize=12)
    ax.set_ylabel('Number of Courses Differing from Baseline', fontsize=12)
    ax.set_title(f'Experiment 3: Solution Diversity Across Restarts\n(Total Courses: {n_courses})', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, diffs):
        pct = (val / n_courses) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(diffs)*0.02,
                f'{val} ({pct:.0f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, "experiment3_diversity.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"[figures] Saved: {filepath}")
    
    plt.show()
    return fig


def generate_combined_summary_chart(results, scaling_data, diversity_data, save=True):
    """
    Generate a combined summary figure with all three experiments.
    Useful for the report's main results section.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[figures] Skipping combined chart - matplotlib not installed")
        return None
    
    ensure_figures_dir()
    
    fig = plt.figure(figsize=(14, 10))
    
    # Experiment 1: Backtracks (top left)
    ax1 = fig.add_subplot(2, 2, 1)
    test_names = [r["name"].split("(")[0].strip() for r in results]
    base_bt = [r["b"]["backtracks"] for r in results]
    enh_bt = [r["e"]["avg_backtracks"] for r in results]
    x = range(len(test_names))
    width = 0.35
    ax1.bar([i - width/2 for i in x], base_bt, width, label='Baseline', color='#2E86AB')
    ax1.bar([i + width/2 for i in x], enh_bt, width, label='Enhancement 1', color='#A23B72')
    ax1.set_ylabel('Backtracks')
    ax1.set_title('(a) Backtrack Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Experiment 1: Time (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    base_time = [r["b"]["time_elapsed"] for r in results]
    enh_time = [r["e"]["avg_time"] for r in results]
    ax2.bar([i - width/2 for i in x], base_time, width, label='Baseline', color='#2E86AB')
    ax2.bar([i + width/2 for i in x], enh_time, width, label='Enhancement 1', color='#A23B72')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('(b) Runtime Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Experiment 2: Scaling (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    sorted_data = sorted(scaling_data, key=lambda x: x["courses"])
    courses = [d["courses"] for d in sorted_data]
    base_t = [d["base_t"] for d in sorted_data]
    enh_t = [d["enh_t"] for d in sorted_data]
    ax3.plot(courses, base_t, 'o-', label='Baseline', color='#2E86AB', linewidth=2)
    ax3.plot(courses, enh_t, 's-', label='Enhancement 1', color='#A23B72', linewidth=2)
    ax3.set_xlabel('Number of Courses')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('(c) Runtime Scaling')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Experiment 3: Diversity (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    diffs = diversity_data["diffs"]
    seeds = diversity_data["seeds"]
    n_courses = diversity_data["n_courses"]
    colors = plt.cm.viridis([i/len(seeds) for i in range(len(seeds))])
    ax4.bar([str(s) for s in seeds], diffs, color=colors, edgecolor='black')
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax4.set_xlabel('Random Seed')
    ax4.set_ylabel('Courses Differing from Baseline')
    ax4.set_title(f'(d) Solution Diversity (n={n_courses} courses)')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Course Scheduling CSP - Experimental Results Summary', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, "combined_experiments_summary.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"[figures] Saved: {filepath}")
    
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Runner helpers (updated with more runs for statistical significance)
# ---------------------------------------------------------------------------

def run_baseline(problem: dict) -> dict:
    """Run baseline once -- deterministic, same result every time."""
    solver   = CSPScheduler(problem)
    solution = solver.solve()
    s        = solver.stats
    return {
        "solution":      solution,
        "solved":        solution is not None,
        "backtracks":    s["backtracks"],
        "nodes_visited": s["nodes_visited"],
        "path_cost":     s["path_cost"],
        "time_elapsed":  s["time_elapsed"],
    }


def run_enhancement1_multi(problem: dict, max_restarts: int = 5,
                           cutoff: int = 30, outer_runs: int = 10) -> dict:
    """
    Run Enhancement 1 multiple times and average.
    Per M2 rubric: stochastic algorithms must be averaged over runs.
    Increased to 10 runs for statistical rigor.
    """
    all_results   = []
    all_solutions = []

    for _ in range(outer_runs):
        solver   = RestartSolver(problem, max_restarts=max_restarts,
                                 cutoff=cutoff, verbose=False)
        solution = solver.solve()
        s        = solver.stats
        all_results.append({
            "solved":           solution is not None,
            "total_backtracks": s["total_backtracks"],
            "total_nodes":      s["total_nodes"],
            "runs_cutoff":      s["runs_cutoff"],
            "solution_run":     s["solution_found_run"],
            "time_elapsed":     s["time_elapsed"],
        })
        if solution is not None:
            all_solutions.append(solution)

    solved = [r for r in all_results if r["solved"]]
    
    # Calculate statistics
    backtracks = [r["total_backtracks"] for r in all_results]
    times = [r["time_elapsed"] for r in all_results]
    cutoffs = [r["runs_cutoff"] for r in all_results]
    
    return {
        "solutions":       all_solutions,
        "solved_rate":     f"{len(solved)}/{outer_runs}",
        "avg_backtracks":  round(statistics.mean(backtracks), 2),
        "std_backtracks":  round(statistics.stdev(backtracks), 2) if len(backtracks) > 1 else 0,
        "avg_nodes":       round(statistics.mean([r["total_nodes"] for r in all_results]), 2),
        "avg_time":        round(statistics.mean(times), 4),
        "std_time":        round(statistics.stdev(times), 4) if len(times) > 1 else 0,
        "avg_runs_cutoff": round(statistics.mean(cutoffs), 2),
        "avg_solution_run": round(
            statistics.mean([r["solution_run"] for r in all_results if r["solution_run"]]), 2
        ) if solved else "N/A",
    }


def sep(char="=", w=74):
    print(char * w)


def row(vals, widths):
    print("  " + "".join(f"{str(v):<{w}}" for v, w in zip(vals, widths)))


def header(cols, widths):
    row(cols, widths)
    print("  " + "-" * sum(widths))


# ---------------------------------------------------------------------------
# Experiment 1 -- Baseline vs Enhancement 1
# ---------------------------------------------------------------------------

def experiment1(test_cases, generate_figures=True):
    sep()
    print("  EXPERIMENT 1 -- Baseline CSP vs Enhancement 1")
    print("  Research Question: On a near-infeasible hard problem, does")
    print("  Enhancement 1 find solutions with fewer wasted backtracks?")
    sep()

    cols   = ["Test Case",           "Algorithm",                        "Solved", "Backtracks", "Nodes", "Time(s)"]
    widths = [34,                    38,                                  10,       14,            10,      10]
    header(cols, widths)

    results = []
    for tc in test_cases:
        name = tc["name"]
        b    = run_baseline(tc)
        e    = run_enhancement1_multi(tc, max_restarts=5, cutoff=30, outer_runs=10)

        row([name,
             "Baseline CSP (MRV+LCV+FC)",
             "Yes" if b["solved"] else "No",
             b["backtracks"], b["nodes_visited"], b["time_elapsed"]], widths)

        row(["",
             f"Enhancement 1 (restarts=5, cutoff=30, avg 10 runs)",
             e["solved_rate"],
             f"{e['avg_backtracks']} ±{e['std_backtracks']}", 
             e["avg_nodes"], 
             f"{e['avg_time']} ±{e['std_time']}"], widths)

        print("  " + "-" * sum(widths))
        results.append({"name": name, "b": b, "e": e})

    print("\n  Key Observations:")
    for r in results:
        b = r["b"]; e = r["e"]
        ratio = round(e["avg_time"] / b["time_elapsed"], 2) if b["time_elapsed"] > 0 else "N/A"
        print(f"    {r['name']}:")
        print(f"      Baseline backtracks        : {b['backtracks']}")
        print(f"      Enhancement 1 avg BT total : {e['avg_backtracks']} ±{e['std_backtracks']} across {e['avg_runs_cutoff']} avg cutoff runs")
        print(f"      Time ratio Enh1/Baseline   : {ratio}x")
    
    # Generate figure
    if generate_figures and MATPLOTLIB_AVAILABLE:
        generate_experiment1_chart(results)
    
    return results


# ---------------------------------------------------------------------------
# Experiment 2 -- Performance Scaling
# ---------------------------------------------------------------------------

def experiment2(test_cases, generate_figures=True):
    sep()
    print("  EXPERIMENT 2 -- Performance Scaling Across Problem Sizes")
    print("  Research Question: How does runtime grow as problem size")
    print("  increases from easy to moderate to hard?")
    sep()

    cols   = ["Test Case",           "Courses", "Algorithm",              "Time(s)", "Backtracks"]
    widths = [34,                    10,        36,                        12,         12]
    header(cols, widths)

    scaling = []
    for tc in test_cases:
        name = tc["name"]
        n    = len(tc["courses"])
        b    = run_baseline(tc)
        e    = run_enhancement1_multi(tc, max_restarts=5, cutoff=30, outer_runs=10)

        row([name, n, "Baseline CSP",          f"{b['time_elapsed']:.4f}",  b["backtracks"]],   widths)
        row(["",   "", "Enhancement 1 (avg)", f"{e['avg_time']:.4f} ±{e['std_time']}", f"{e['avg_backtracks']} ±{e['std_backtracks']}"], widths)
        print("  " + "-" * sum(widths))

        scaling.append({
            "name": name, "courses": n,
            "base_t": b["time_elapsed"],  "enh_t": e["avg_time"],
            "base_bt": b["backtracks"],   "enh_bt": e["avg_backtracks"],
        })

    print("\n  Scaling summary (use for line chart in report):")
    print(f"  {'Test Case':<34} {'Courses':<10} {'Base Time':<14} {'Enh1 Time':<14} {'Base BT':<12} {'Enh1 BT'}")
    print("  " + "-" * 86)
    for r in scaling:
        print(f"  {r['name']:<34} {r['courses']:<10} {r['base_t']:<14} {r['enh_t']:<14} {r['base_bt']:<12} {r['enh_bt']}")
    
    # Generate figure
    if generate_figures and MATPLOTLIB_AVAILABLE:
        generate_experiment2_chart(scaling)
    
    return scaling


# ---------------------------------------------------------------------------
# Experiment 3 -- Solution Diversity
# ---------------------------------------------------------------------------

def experiment3(test_cases, generate_figures=True):
    sep()
    print("  EXPERIMENT 3 -- Solution Diversity")
    print("  Research Question: Does Enhancement 1 produce meaningfully")
    print("  different schedules across restarts vs the deterministic baseline?")
    print("  Diversity is the primary advantage: Enhancement 2 (GLS) needs")
    print("  diverse starting points to explore better soft-constraint solutions.")
    sep()

    tc   = test_cases[1]   # use moderate case -- 20 courses shows diversity clearly
    name = tc["name"]
    n    = len(tc["courses"])
    print(f"  Problem: {name} ({n} courses)\n")

    b     = run_baseline(tc)
    sol_b = b["solution"]

    print(f"  Baseline (deterministic -- always identical):")
    print(f"    Backtracks : {b['backtracks']} | Time: {b['time_elapsed']}s")
    print(f"    Always produces the exact same schedule every run.\n")

    seeds   = [42, 84, 126, 168, 210, 252, 294, 336]
    diffs   = []
    bt_list = []
    checker = ConstraintChecker(tc)

    cols   = ["Seed", "Backtracks", "Courses differing from baseline", "Valid?"]
    widths = [8,      14,           36,                                 10]
    header(cols, widths)

    for seed in seeds:
        csp   = SingleRunCSP(tc, cutoff=100, seed=seed)
        sol_e = csp.solve()
        bt    = csp.backtracks
        bt_list.append(bt)

        if sol_e is not None:
            diff   = count_solution_differences(sol_b, sol_e, tc["courses"])
            report = checker.validate_full(sol_e)
            valid  = "Yes" if report.is_feasible() else f"No({report.total})"
            diffs.append(diff)
        else:
            diff  = "N/A"
            valid = "None"

        row([seed, bt, diff, valid], widths)

    print()
    if diffs:
        avg_diff = round(statistics.mean(diffs), 1)
        pct      = round(avg_diff / n * 100, 1)
        print(f"  Summary:")
        print(f"    Baseline always  : 0 courses differ (identical every run)")
        print(f"    Enhancement 1    : avg {avg_diff} / {n} courses differ per seed")
        print(f"    Min differences  : {min(diffs)} courses")
        print(f"    Max differences  : {max(diffs)} courses")
        print(f"    Diversity rate   : {pct}% of assignments differ on average")
        print(f"\n  Conclusion:")
        print(f"    Enhancement 1 explores ~{pct}% different regions of the solution")
        print(f"    space per restart. This gives Enhancement 2 (GLS) diverse")
        print(f"    starting points to find better soft-constraint schedules.")
        print(f"    A deterministic baseline can only ever give GLS one start point.")

    diversity_data = {"diffs": diffs, "seeds": seeds, "n_courses": n}
    
    # Generate figure
    if generate_figures and MATPLOTLIB_AVAILABLE and diffs:
        generate_experiment3_chart(diversity_data)

    return diversity_data

# ---------------------------------------------------------------------------
# Experiment 4 — Hard Problem Deep Dive (Demonstrates Enhancement Advantage)
# ---------------------------------------------------------------------------

"""def experiment4_hard_problem_demo():
    
    Specialized experiment to clearly show Enhancement 1 beating Baseline.
    Runs multiple trials on the hardest problems and compares success rates.
    
    sep()
    print("  EXPERIMENT 4 -- Hard Problem Deep Dive")
    print("  Purpose: Demonstrate Enhancement 1's advantage on difficult")
    print("  problems where deterministic baseline gets stuck.")
    sep()
    
    # Load the hardest test cases
    from data_loader import generate_test_case_very_hard, generate_test_case_extreme
    
    hard_cases = [
        ("Very Hard (16 courses)", generate_test_case_very_hard()),
        ("Extreme (12 courses, 3 slots)", generate_test_case_extreme()),
    ]
    
    results = []
    
    for case_name, problem in hard_cases:
        print(f"\n  {'='*60}")
        print(f"  PROBLEM: {case_name}")
        print(f"  {'='*60}")
        print(f"  Courses: {len(problem['courses'])}")
        print(f"  Timeslots: {len(problem['timeslots'])}")
        print(f"  Rooms: {len(problem['rooms'])}")
        print(f"  Total room-slot combos: {len(problem['rooms']) * len(problem['timeslots'])}")
        print(f"  Curricula groups: {len(problem['curricula'])}")
        
        # Run Baseline 5 times (though deterministic, shows consistency)
        baseline_results = []
        print(f"\n  BASELINE (deterministic MRV):")
        for trial in range(5):
            solver = CSPScheduler(problem)
            solution = solver.solve()
            baseline_results.append({
                "solved": solution is not None,
                "backtracks": solver.stats["backtracks"],
                "time": solver.stats["time_elapsed"],
            })
            status = "✓ SOLVED" if solution else "✗ FAILED"
            print(f"    Trial {trial+1}: {status} | Backtracks: {solver.stats['backtracks']:4d} | Time: {solver.stats['time_elapsed']:.3f}s")
        
        baseline_success = sum(1 for r in baseline_results if r["solved"])
        
        # Run Enhancement 1 with restarts
        print(f"\n  ENHANCEMENT 1 (randomized restarts, max_restarts=10, cutoff=50):")
        enhancement_results = []
        for trial in range(5):
            solver = RestartSolver(problem, max_restarts=10, cutoff=50, verbose=False)
            solution = solver.solve()
            enhancement_results.append({
                "solved": solution is not None,
                "total_backtracks": solver.stats["total_backtracks"],
                "time": solver.stats["time_elapsed"],
                "runs_used": solver.stats["solution_found_run"] or solver.stats["runs_completed"],
            })
            status = "✓ SOLVED" if solution else "✗ FAILED"
            print(f"    Trial {trial+1}: {status} | Total Backtracks: {solver.stats['total_backtracks']:4d} | Time: {solver.stats['time_elapsed']:.3f}s | Runs: {enhancement_results[-1]['runs_used']}")
        
        enhancement_success = sum(1 for r in enhancement_results if r["solved"])
        
        results.append({
            "name": case_name,
            "baseline_success": f"{baseline_success}/5",
            "enhancement_success": f"{enhancement_success}/5",
            "baseline_avg_bt": statistics.mean([r["backtracks"] for r in baseline_results if r["solved"]]) if baseline_success > 0 else "N/A",
            "enhancement_avg_bt": statistics.mean([r["total_backtracks"] for r in enhancement_results if r["solved"]]) if enhancement_success > 0 else "N/A",
        })
    
    # Summary table
    print(f"\n\n  {'='*60}")
    print("  SUMMARY: Baseline vs Enhancement 1 on Hard Problems")
    print(f"  {'='*60}")
    print(f"  {'Problem':<30} {'Baseline Success':<20} {'Enhancement 1 Success':<22}")
    print(f"  {'-'*72}")
    for r in results:
        print(f"  {r['name']:<30} {r['baseline_success']:<20} {r['enhancement_success']:<22}")
    
    print(f"\n  CONCLUSION:")
    print(f"    On easy problems, both algorithms succeed.")
    print(f"    On HARD problems, Enhancement 1's randomized restarts")
    print(f"    escape dead ends that trap the deterministic baseline.")
    print(f"    This demonstrates the value of the restart enhancement.")
    
    return results
"""
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
    
    if not MATPLOTLIB_AVAILABLE:
        print("  NOTE: matplotlib not installed. Figures will not be generated.")
        print("  To install: pip install matplotlib")
        print()

    test_cases = get_all_test_cases()
    print("  Test cases loaded:")
    for tc in test_cases:
        print(f"    {tc['name']}: {len(tc['courses'])} courses, "
              f"{len(tc['rooms'])} rooms, {len(tc['timeslots'])} timeslots, "
              f"{len(tc['curricula'])} curricula groups")
    print()

    # Run experiments
    r1 = experiment1(test_cases[:3], generate_figures=True)  # First 3 cases
    print()
    r2 = experiment2(test_cases[:3], generate_figures=True)
    print()
    r3 = experiment3(test_cases, generate_figures=True)
    print()
    
    # Generate combined summary figure
    if MATPLOTLIB_AVAILABLE and r1 and r2 and r3:
        print()
        generate_combined_summary_chart(r1, r2, r3)

    print()
    sep()
    print("  All experiments complete. Figures saved to ./figures/ directory")
    print("  For your M2 report:")
    print("  Exp 1 -> Bar chart : Backtracks & time -- Baseline vs Enhancement 1")
    print("  Exp 2 -> Line chart: Problem size vs runtime (3 points per algorithm)")
    print("  Exp 3 -> Bar chart : Diversity % per seed vs 0% for baseline")
    print(f"  Figures saved to: {os.path.abspath(FIGURES_DIR)}/")
    sep()