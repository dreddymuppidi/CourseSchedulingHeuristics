"""
========================================================
ENHANCEMENT 2 -- Constraint Weighting / Guided Local Search
========================================================

What this enhancement adds over Enhancement 1:
    Enhancement 1 finds a FEASIBLE schedule -- one that satisfies
    all hard constraints (H1-H5). But all feasible solutions are
    treated equally -- no attempt is made to make the schedule
    GOOD in terms of soft preferences.

    Enhancement 2 starts from the feasible solution produced by
    Enhancement 1 and OPTIMIZES it by minimizing a weighted
    soft constraint penalty score F(S).

    Soft Constraints Optimized:
        S1 -- Instructor back-to-back: penalize instructors teaching
              consecutive timeslots on the same day
        S2 -- Room over-capacity: penalize assigning rooms much
              larger than needed (wasted space)
        S3 -- Student spread: penalize courses in same curriculum
              being scheduled on the same day (prefer spread across week)
        S4 -- Instructor daily load: penalize instructors teaching
              more than 2 courses in a single day

    Guided Local Search (GLS) Mechanism:
        Plain min-conflicts assigns equal weight to all soft constraints
        throughout. GLS adapts: when a local minimum is reached
        (no single reassignment improves F(S)), the weights of the
        most persistently violated constraints are INCREASED.
        This shifts the cost landscape and allows GLS to escape
        local minima that plain min-conflicts cannot.

Path Cost (Enhancement 2):
    F(S) = sum(wi * vi) for all soft constraints i
    where vi = number of violations of soft constraint i
          wi = adaptive weight (starts at 1, increases at local minima)
    Lower F(S) = better quality schedule.

How it differs from Enhancement 1:
    Enhancement 1: finds ANY feasible schedule (hard constraints only)
    Enhancement 2: improves that schedule by minimizing F(S) (soft)

How it differs from plain Min-Conflicts:
    Min-Conflicts: fixed weights wi = 1 throughout
    GLS:           wi increases when constraint i is repeatedly violated
                   at local minima -- adaptive cost landscape

References:
    [1] Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern
        Approach (4th ed.). Pearson. Chapter 4 (Local Search).
    [2] Schaerf, A. (1999). A survey of automated timetabling.
        Artificial Intelligence Review, 13(2), 87-127.
    [3] Bonutti et al. (2012). Benchmarking curriculum-based course timetabling.
        Annals of Operations Research, 194(1), 59-70.
    [5] Abramson, D. (1991). Constructing school timetables using simulated
        annealing. Management Science, 37(1), 98-113.
"""

import sys
import time
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

from data_loader import load_problem, generate_test_case_small
from scheduler import Assignment, Triple, CSPScheduler
from enhancement1_restarts import RestartSolver
from constraint_checker import ConstraintChecker


# ---------------------------------------------------------------------------
# Soft Constraint Configuration
# ---------------------------------------------------------------------------

SOFT_CONSTRAINTS = ["S1_back_to_back", "S2_room_waste", "S3_same_day", "S4_daily_load"]

INITIAL_WEIGHTS = {
    "S1_back_to_back": 1.0,
    "S2_room_waste":   1.0,
    "S3_same_day":     1.0,
    "S4_daily_load":   1.0,
}

WEIGHT_INCREMENT = 1.0
WEIGHT_CEILING   = 10.0
NORMALIZE_EVERY  = 50


# ---------------------------------------------------------------------------
# Soft Constraint Evaluator
# ---------------------------------------------------------------------------

def extract_day(timeslot: str) -> str:
    """Extract day prefix from timeslot. 'MWF_9am' -> 'MWF', 'T3' -> 'T3'."""
    if "_" in timeslot:
        return timeslot.split("_")[0]
    return timeslot


def compute_violations(assignment: Assignment, problem: dict) -> dict:
    """
    Compute raw violation count (vi) for each soft constraint.

    S1 -- Instructor back-to-back in same day
    S2 -- Room capacity > 2x enrollment (wasted room)
    S3 -- Two courses from same curriculum on same day
    S4 -- Instructor teaches > 2 courses in one day
    """
    mapping     = assignment.mapping
    rooms       = problem["rooms"]
    enrollments = problem["enrollments"]
    curricula   = problem["curricula"]

    violations = {sc: 0 for sc in SOFT_CONSTRAINTS}

    # S1: Instructor back-to-back
    instr_day_slots = defaultdict(list)
    for course, triple in mapping.items():
        day = extract_day(triple.timeslot)
        instr_day_slots[(triple.instructor, day)].append(triple.timeslot)
    for (instr, day), slots in instr_day_slots.items():
        unique_slots = sorted(set(slots))
        for i in range(len(unique_slots) - 1):
            violations["S1_back_to_back"] += 1

    # S2: Room waste
    for course, triple in mapping.items():
        capacity   = rooms.get(triple.room, 0)
        enrollment = enrollments.get(course, 0)
        if enrollment > 0 and capacity > 2 * enrollment:
            violations["S2_room_waste"] += 1

    # S3: Curriculum same-day
    for curriculum in curricula:
        clist = list(curriculum)
        for i in range(len(clist)):
            for j in range(i + 1, len(clist)):
                c1, c2 = clist[i], clist[j]
                if c1 not in mapping or c2 not in mapping:
                    continue
                if extract_day(mapping[c1].timeslot) == extract_day(mapping[c2].timeslot):
                    violations["S3_same_day"] += 1

    # S4: Instructor daily overload
    instr_day_count = defaultdict(int)
    for course, triple in mapping.items():
        day = extract_day(triple.timeslot)
        instr_day_count[(triple.instructor, day)] += 1
    for (instr, day), count in instr_day_count.items():
        if count > 2:
            violations["S4_daily_load"] += count - 2

    return violations


def compute_penalty(assignment: Assignment, problem: dict, weights: dict) -> float:
    """
    F(S) = sum(wi * vi) for all soft constraints.
    Lower F(S) = better quality schedule.
    This is the Enhancement 2 path cost / objective function.
    """
    violations = compute_violations(assignment, problem)
    return sum(weights[sc] * violations[sc] for sc in SOFT_CONSTRAINTS)


# ---------------------------------------------------------------------------
# Guided Local Search
# ---------------------------------------------------------------------------

class GuidedLocalSearch:
    """
    ENHANCEMENT 2 -- Guided Local Search (GLS).

    Starts from a feasible Enhancement 1 solution and minimizes
    F(S) = sum(wi * vi) by iteratively reassigning courses.

    GLS vs Min-Conflicts:
        Min-Conflicts uses fixed wi = 1 forever.
        GLS increases wi at local minima -- adaptive cost landscape
        that allows escaping stuck regions.

    Hard constraints (H1-H5) are checked before every reassignment.
    The schedule remains feasible throughout optimization.
    """

    def __init__(self, problem: dict, starting_solution: Assignment,
                 max_iterations: int = 200, verbose: bool = True):
        self.problem        = problem
        self.courses        = problem["courses"]
        self.rooms          = problem["rooms"]
        self.timeslots      = problem["timeslots"]
        self.instructors    = problem["instructors"]
        self.enrollments    = problem["enrollments"]
        self.curricula      = problem["curricula"]
        self.checker        = ConstraintChecker(problem)
        self.max_iterations = max_iterations
        self.verbose        = verbose
        self.current        = Assignment(mapping=dict(starting_solution.mapping))
        self.weights        = dict(INITIAL_WEIGHTS)
        self.all_triples    = self._build_all_triples()
        self.stats = {
            "iterations":       0,
            "local_minima_hit": 0,
            "weight_updates":   {},
            "initial_penalty":  0.0,
            "final_penalty":    0.0,
            "improvement_pct":  0.0,
            "time_elapsed":     0.0,
            "penalty_history":  [],
        }

    # ------------------------------------------------------------------
    # Public Entry Point
    # ------------------------------------------------------------------

    def optimize(self) -> Assignment:
        """
        Run GLS. Returns best Assignment found (lowest unweighted F(S)).
        Hard constraints satisfied at every iteration.
        """
        start = time.time()

        initial_penalty = compute_penalty(self.current, self.problem, INITIAL_WEIGHTS)
        self.stats["initial_penalty"] = round(initial_penalty, 3)

        if self.verbose:
            sep = "-" * 60
            print(f"\n  [Enhancement 2] Guided Local Search")
            print(f"  Max iterations : {self.max_iterations}")
            print(f"  Initial F(S)   : {initial_penalty:.3f}")
            print(f"  {sep}")
            print(f"  {'Iter':<8} {'Weighted F(S)':<16} {'Best F(S)':<14} {'Event'}")
            print(f"  {sep}")

        best_solution     = Assignment(mapping=dict(self.current.mapping))
        best_penalty      = initial_penalty
        no_improve_streak = 0

        for iteration in range(1, self.max_iterations + 1):
            self.stats["iterations"] = iteration

            # One local search step using WEIGHTED F(S) to guide moves
            improved, weighted_penalty = self._local_search_step()

            # Track best using UNWEIGHTED F(S) -- true quality measure
            true_penalty = compute_penalty(self.current, self.problem, INITIAL_WEIGHTS)
            if true_penalty < best_penalty:
                best_penalty      = true_penalty
                best_solution     = Assignment(mapping=dict(self.current.mapping))
                no_improve_streak = 0
            else:
                no_improve_streak += 1

            self.stats["penalty_history"].append(round(true_penalty, 3))

            event = ""
            if not improved:
                # GLS: increase weight of most-penalized constraint
                event = "LOCAL MIN -> weight update"
                self._update_weights()
                self.stats["local_minima_hit"] += 1
                if self.stats["local_minima_hit"] % NORMALIZE_EVERY == 0:
                    self._normalize_weights()
                    event = "LOCAL MIN -> normalize + weight update"

            if self.verbose and (
                iteration <= 5
                or iteration % 25 == 0
                or (not improved and iteration <= 15)
            ):
                print(f"  {iteration:<8} {weighted_penalty:<16.3f} {best_penalty:<14.3f} {event}")

            if best_penalty == 0.0:
                if self.verbose:
                    print(f"  {'-'*60}")
                    print(f"  Converged at iteration {iteration} -- F(S) = 0.0")
                break

            if no_improve_streak >= 40:
                if self.verbose:
                    print(f"  {'-'*60}")
                    print(f"  Stopping: no improvement for 40 iterations")
                break

        self.stats["final_penalty"]   = round(best_penalty, 3)
        self.stats["improvement_pct"] = round(
            (1 - best_penalty / initial_penalty) * 100, 1
        ) if initial_penalty > 0 else 0.0
        self.stats["time_elapsed"]    = round(time.time() - start, 3)
        self.current = best_solution

        if self.verbose:
            print(f"  {'-'*60}")

        return best_solution

    # ------------------------------------------------------------------
    # Local Search Step
    # ------------------------------------------------------------------

    def _local_search_step(self) -> tuple:
        """
        Try all single-course reassignments.
        Accept the one that most reduces WEIGHTED F(S).
        Hard constraints verified before acceptance.

        Returns: (improved: bool, new_weighted_penalty: float)
        """
        current_penalty = compute_penalty(self.current, self.problem, self.weights)
        best_delta  = 0.0
        best_course = None
        best_triple = None

        for course in self.courses:
            original = self.current.mapping[course]

            for candidate in self.all_triples[course]:
                if candidate == original:
                    continue

                self.current.mapping[course] = candidate

                if not self._is_hard_feasible():
                    self.current.mapping[course] = original
                    continue

                new_penalty = compute_penalty(self.current, self.problem, self.weights)
                delta = new_penalty - current_penalty

                if delta < best_delta:
                    best_delta  = delta
                    best_course = course
                    best_triple = candidate

                self.current.mapping[course] = original

        if best_course is not None:
            self.current.mapping[best_course] = best_triple
            new_penalty = compute_penalty(self.current, self.problem, self.weights)
            return True, new_penalty

        return False, current_penalty

    def _is_hard_feasible(self) -> bool:
        """Fast H1, H2, H5 check on current mapping."""
        seen_instr  = {}
        seen_room   = {}
        seen_curric = {}

        for course, triple in self.current.mapping.items():
            key_i = (triple.instructor, triple.timeslot)
            if key_i in seen_instr:
                return False
            seen_instr[key_i] = course

            key_r = (triple.room, triple.timeslot)
            if key_r in seen_room:
                return False
            seen_room[key_r] = course

            for idx, curriculum in enumerate(self.curricula):
                if course in curriculum:
                    key_c = (idx, triple.timeslot)
                    if key_c in seen_curric:
                        return False
                    seen_curric[key_c] = course

        return True

    # ------------------------------------------------------------------
    # GLS Weight Update
    # ------------------------------------------------------------------

    def _update_weights(self):
        """
        GLS utility rule: increase weight of constraint with highest
        utility = vi / (1 + wi).
        Balances violation frequency vs existing penalty level.
        Reference: AIMA 4th Ed., Chapter 4 -- GLS sidebar.
        """
        violations = compute_violations(self.current, self.problem)
        utility = {
            sc: violations[sc] / (1.0 + self.weights[sc])
            for sc in SOFT_CONSTRAINTS
        }
        best_sc = max(utility, key=lambda sc: utility[sc])
        if utility[best_sc] > 0:
            self.weights[best_sc] = min(
                self.weights[best_sc] + WEIGHT_INCREMENT,
                WEIGHT_CEILING
            )
            self.stats["weight_updates"][best_sc] = (
                self.stats["weight_updates"].get(best_sc, 0) + 1
            )

    def _normalize_weights(self):
        """Scale down weights to prevent single constraint domination."""
        max_w = max(self.weights.values())
        if max_w > WEIGHT_CEILING / 2:
            scale = (WEIGHT_CEILING / 2) / max_w
            for sc in self.weights:
                self.weights[sc] = max(1.0, self.weights[sc] * scale)

    # ------------------------------------------------------------------
    # Domain Builder
    # ------------------------------------------------------------------

    def _build_all_triples(self) -> dict:
        """Build all H3/H4-valid triples per course for reassignment."""
        all_triples = {}
        for course in self.courses:
            enrollment = self.enrollments.get(course, 0)
            triples    = []
            for room, capacity in self.rooms.items():
                if capacity < enrollment:
                    continue
                for timeslot in self.timeslots:
                    for instr_id, instr_info in self.instructors.items():
                        if course not in instr_info.get("qualified_courses", []):
                            continue
                        triples.append(Triple(
                            room=room,
                            timeslot=timeslot,
                            instructor=instr_id
                        ))
            all_triples[course] = triples
        return all_triples

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def print_summary(self):
        """Print Enhancement 2 results summary."""
        s = self.stats
        print(f"\n  ENHANCEMENT 2 -- GLS SUMMARY")
        print(f"  {'-' * 46}")
        print(f"  Iterations run         : {s['iterations']}")
        print(f"  Local minima escaped   : {s['local_minima_hit']}")
        print(f"  Initial F(S)           : {s['initial_penalty']}")
        print(f"  Final F(S)             : {s['final_penalty']}")
        print(f"  Improvement            : {s['improvement_pct']}%")
        print(f"  Time elapsed           : {s['time_elapsed']}s")

        if s["weight_updates"]:
            print(f"\n  GLS weight updates (most problematic constraints):")
            for sc, count in sorted(s["weight_updates"].items(), key=lambda x: -x[1]):
                print(f"    {sc:<28}: updated {count}x, final w={self.weights[sc]:.2f}")

        print(f"\n  Final soft constraint violations:")
        viol = compute_violations(self.current, self.problem)
        for sc in SOFT_CONSTRAINTS:
            print(f"    {sc:<28}: {viol[sc]} violations")

        checker = ConstraintChecker(self.problem)
        report  = checker.validate_full(self.current)
        print(f"\n  HARD CONSTRAINT CHECK (must all be 0):")
        print(report)
        if report.is_feasible():
            print(f"\n  ✓ All hard constraints satisfied throughout optimization.")
        else:
            print(f"\n  ✗ Hard violation detected!")


# ---------------------------------------------------------------------------
# Full Pipeline: Baseline -> Enhancement 1 -> Enhancement 2
# ---------------------------------------------------------------------------

class FullPipeline:
    """
    Runs the complete 3-stage optimization pipeline:
        Stage 1 -- Baseline CSP:    Find any feasible schedule
        Stage 2 -- Enhancement 1:   Find better feasible schedule via restarts
        Stage 3 -- Enhancement 2:   Optimize soft constraint quality via GLS
    """

    def __init__(self, problem: dict,
                 max_restarts: int = 5,
                 cutoff: int = 50,
                 max_gls_iterations: int = 150,
                 verbose: bool = True):
        self.problem            = problem
        self.max_restarts       = max_restarts
        self.cutoff             = cutoff
        self.max_gls_iterations = max_gls_iterations
        self.verbose            = verbose
        self.results            = {}

    def run(self):
        """Run all 3 stages and return the final optimized solution."""
        print("=" * 62)
        print("  Full Pipeline: Baseline -> Enhancement 1 -> Enhancement 2")
        print("=" * 62)

        # Stage 1: Baseline
        print("\n  STAGE 1 -- Baseline CSP (MRV + LCV + FC)")
        solver_b          = CSPScheduler(self.problem)
        baseline_solution = solver_b.solve()
        sb                = solver_b.stats

        if baseline_solution is None:
            print("  ✗ Baseline failed -- problem is infeasible.")
            return None

        base_penalty = compute_penalty(baseline_solution, self.problem, INITIAL_WEIGHTS)
        print(f"  ✓ Solution found | bt={sb['backtracks']} | "
              f"t={sb['time_elapsed']}s | F(S)={base_penalty:.3f}")
        self.results["baseline"] = {
            "backtracks": sb["backtracks"],
            "time":       sb["time_elapsed"],
            "penalty":    round(base_penalty, 3),
        }

        # Stage 2: Enhancement 1
        print(f"\n  STAGE 2 -- Enhancement 1 "
              f"(restarts={self.max_restarts}, cutoff={self.cutoff})")
        solver_e1 = RestartSolver(
            self.problem, max_restarts=self.max_restarts,
            cutoff=self.cutoff, verbose=self.verbose
        )
        enh1_solution = solver_e1.solve()
        se1           = solver_e1.stats

        if enh1_solution is None:
            print("  ✗ Enhancement 1 failed -- using baseline for Stage 3.")
            enh1_solution = baseline_solution
        else:
            enh1_penalty = compute_penalty(enh1_solution, self.problem, INITIAL_WEIGHTS)
            print(f"  ✓ Found on run {se1['solution_found_run']} | "
                  f"total_bt={se1['total_backtracks']} | "
                  f"cutoffs={se1['runs_cutoff']} | "
                  f"t={se1['time_elapsed']}s | F(S)={enh1_penalty:.3f}")
            self.results["enhancement1"] = {
                "backtracks":  se1["total_backtracks"],
                "runs_cutoff": se1["runs_cutoff"],
                "time":        se1["time_elapsed"],
                "penalty":     round(enh1_penalty, 3),
            }

        # Stage 3: Enhancement 2 (GLS)
        print(f"\n  STAGE 3 -- Enhancement 2 (GLS, max_iter={self.max_gls_iterations})")
        gls = GuidedLocalSearch(
            problem=self.problem,
            starting_solution=enh1_solution,
            max_iterations=self.max_gls_iterations,
            verbose=self.verbose
        )
        final_solution = gls.optimize()
        gls.print_summary()

        self.results["enhancement2"] = {
            "initial_penalty":  gls.stats["initial_penalty"],
            "final_penalty":    gls.stats["final_penalty"],
            "improvement_pct":  gls.stats["improvement_pct"],
            "local_minima_hit": gls.stats["local_minima_hit"],
            "time":             gls.stats["time_elapsed"],
        }

        self._print_final_comparison()
        return final_solution

    def _print_final_comparison(self):
        print(f"\n{'=' * 62}")
        print(f"  PIPELINE SUMMARY -- F(S) Across All Stages")
        print(f"{'=' * 62}")
        print(f"  {'Stage':<32} {'F(S)':<12} {'Notes'}")
        print(f"  {'-' * 58}")

        if "baseline" in self.results:
            r = self.results["baseline"]
            print(f"  {'Stage 1 -- Baseline CSP':<32} {r['penalty']:<12.3f} "
                  f"bt={r['backtracks']}, t={r['time']}s")

        if "enhancement1" in self.results:
            r = self.results["enhancement1"]
            print(f"  {'Stage 2 -- Enhancement 1':<32} {r['penalty']:<12.3f} "
                  f"total_bt={r['backtracks']}, cutoffs={r['runs_cutoff']}, t={r['time']}s")

        if "enhancement2" in self.results:
            r = self.results["enhancement2"]
            print(f"  {'Stage 3 -- Enhancement 2 (GLS)':<32} {r['final_penalty']:<12.3f} "
                  f"improved {r['improvement_pct']}%, t={r['time']}s")
            base_p = self.results.get("baseline", {}).get("penalty", 0)
            if base_p > 0:
                total = round((1 - r["final_penalty"] / base_p) * 100, 1)
                print(f"  {'-' * 58}")
                print(f"  Total improvement (Stage 1 -> Stage 3): {total}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("  Course Scheduling -- Enhancement 2: Guided Local Search")
    print("  CS 57200: Heuristic Problem Solving")
    print("  Deekshitha Reddy Muppidi | Track B: Optimization/Planning")
    print("=" * 62)

    problem = generate_test_case_small()

    print(f"\nProblem: {problem.get('name')}")
    print(f"  Courses     : {len(problem['courses'])}")
    print(f"  Rooms       : {len(problem['rooms'])}")
    print(f"  Timeslots   : {len(problem['timeslots'])}")
    print(f"  Instructors : {len(problem['instructors'])}")
    print(f"  Curricula   : {len(problem['curricula'])}")

    print(f"\nPath cost (Enhancement 2):")
    print(f"  F(S) = sum(wi * vi)")
    print(f"  S1=back-to-back | S2=room-waste | S3=same-day | S4=daily-load")
    print(f"  wi starts at 1.0, increases adaptively at local minima (GLS)")

    pipeline = FullPipeline(
        problem=problem,
        max_restarts=5,
        cutoff=50,
        max_gls_iterations=150,
        verbose=True
    )
    final = pipeline.run()

    if final:
        print(f"\n  Final schedule:")
        print(f"  {'-' * 58}")
        print(f"  {'Course':<12} {'Room':<12} {'Timeslot':<16} {'Instructor'}")
        print(f"  {'-' * 58}")
        for course in sorted(final.mapping.keys()):
            t = final.mapping[course]
            print(f"  {course:<12} {t.room:<12} {t.timeslot:<16} {t.instructor}")

    print("\nDone.")
