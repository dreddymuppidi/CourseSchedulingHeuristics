"""
enhancement1_restarts.py
========================
CS 57200: Heuristic Problem Solving -- Project Milestone 1
Track B: Optimization / Planning
Student: Deekshitha Reddy Muppidi

========================================================
ENHANCEMENT 1 -- Iterative Deepening with Restarts
========================================================

What this enhancement adds over the Baseline:
    The baseline CSP solver (scheduler.py) is deterministic --
    MRV and LCV always make the same choices, so it always
    explores the same search path. If that path is unproductive,
    the solver gets stuck repeating the same dead ends.

    Enhancement 1 fixes this by:
        1. Running the CSP solver multiple times (restarts)
        2. Each run uses RANDOMIZED tie-breaking in MRV
           so each run explores a DIFFERENT search path
        3. Each run has a BACKTRACK CUTOFF -- if the solver
           hits too many dead ends, it is abandoned and
           restarted fresh with a new random seed
        4. The BEST complete solution across all runs is kept

Path Cost (Enhancement 1):
    Path cost = total backtracks accumulated across all runs.
    Each run contributes its backtrack count to the total.
    Runs abandoned at the cutoff are counted in full.
    There is still NO soft constraint optimization at this
    stage -- the goal remains feasibility only, achieved
    more reliably through randomized search diversity.

How it differs from the Baseline:
    Baseline    : 1 deterministic run, no cutoff, stops at
                  first feasible solution found
    Enhancement : N randomized runs with cutoff, keeps best
                  solution across all runs

References:
    [1] Russell, S., & Norvig, P. (2020). Artificial Intelligence:
        A Modern Approach (4th ed.). Pearson. Chapters 4 & 6.
        Section 4.3 -- Random restarts for local search.
    [2] Schaerf, A. (1999). A survey of automated timetabling.
        Artificial Intelligence Review, 13(2), 87-127.
    [3] Bonutti et al. (2012). Benchmarking curriculum-based
        course timetabling. Annals of Operations Research,
        194(1), 59-70.
    [4] ITC-2007 Dataset: http://www.cs.qub.ac.uk/itc2007
"""

import sys
import time
import copy
import random
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

# Fix for Windows terminals that default to cp1252 encoding
sys.stdout.reconfigure(encoding="utf-8")

from data_loader import load_problem
from constraint_checker import ConstraintChecker
from display import print_schedule

# -----------------------------------------------------------------------
# Reuse core data structures from baseline
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class Triple:
    """
    A candidate assignment for one course: (room, timeslot, instructor).
    Identical to baseline -- Enhancement 1 reuses this structure unchanged.
    """
    room: str
    timeslot: str
    instructor: str

    def __repr__(self):
        return f"({self.room}, {self.timeslot}, {self.instructor})"


@dataclass
class Assignment:
    """
    Partial or complete schedule: maps course_id -> Triple.

    Enhancement 1 path cost:
        path_cost = len(mapping) per run.
        Total path cost across all runs is tracked in
        RestartSolver.stats["total_backtracks"].
    """
    mapping: dict = field(default_factory=dict)

    def assign(self, course: str, triple: Triple):
        self.mapping[course] = triple

    def unassign(self, course: str):
        self.mapping.pop(course, None)

    def is_complete(self, courses: list) -> bool:
        return all(c in self.mapping for c in courses)

    @property
    def path_cost(self) -> int:
        return len(self.mapping)

    def copy(self):
        return Assignment(mapping=dict(self.mapping))


# -----------------------------------------------------------------------
# Enhancement 1 -- Single Run CSP with Randomized MRV + Cutoff
# -----------------------------------------------------------------------

class SingleRunCSP:
    """
    One run of the CSP solver with:
        - Randomized MRV tie-breaking (controlled by random seed)
        - Backtrack cutoff limit

    This is the inner engine called repeatedly by RestartSolver.
    It is identical to the baseline EXCEPT:
        1. MRV tie-breaking is randomized (not degree-based)
        2. A backtrack cutoff stops the run early if triggered

    Reference: AIMA 4th Ed., Fig 6.5 -- BACKTRACKING-SEARCH(csp)
    """

    def __init__(self, problem: dict, cutoff: int, seed: int):
        self.courses     = problem["courses"]
        self.rooms       = problem["rooms"]
        self.timeslots   = problem["timeslots"]
        self.instructors = problem["instructors"]
        self.enrollments = problem["enrollments"]
        self.curricula   = problem["curricula"]
        self.checker     = ConstraintChecker(problem)
        self.cutoff      = cutoff   # max backtracks before this run is abandoned
        self.seed        = seed     # controls randomized MRV tie-breaking
        self.rng         = random.Random(seed)

        # Per-run stats
        self.backtracks    = 0
        self.nodes_visited = 0
        self.cutoff_hit    = False

        # Build domains with unary constraint filtering (same as baseline)
        self.domains = self._build_initial_domains()

    def solve(self) -> Optional[Assignment]:
        """
        Run one backtracking search attempt with randomized MRV.
        Returns a complete Assignment or None (cutoff hit / no solution).
        """
        assignment = Assignment()
        domains    = copy.deepcopy(self.domains)
        return self._backtrack(assignment, domains)

    # -------------------------------------------------------------------
    # Core Backtracking (AIMA Fig 6.5)
    # -------------------------------------------------------------------

    def _backtrack(self, assignment: Assignment, domains: dict) -> Optional[Assignment]:
        """Recursive backtracking -- same structure as baseline."""

        # Goal test
        if assignment.is_complete(self.courses):
            return assignment

        # Cutoff check -- abandon this run if too many backtracks
        # This is the key addition over the baseline
        if self.backtracks >= self.cutoff:
            self.cutoff_hit = True
            return None

        self.nodes_visited += 1

        # MRV with RANDOMIZED tie-breaking (Enhancement 1 change)
        course = self._select_unassigned_variable_random(assignment, domains)

        # LCV value ordering (same as baseline)
        ordered_triples = self._order_domain_values(course, assignment, domains)

        for triple in ordered_triples:
            if self.checker.is_consistent(course, triple, assignment):
                assignment.assign(course, triple)

                pruned = self._forward_check(course, triple, assignment, domains)

                if pruned is not None:
                    result = self._backtrack(assignment, domains)
                    if result is not None:
                        return result

                self._restore_pruned(pruned, domains)
                assignment.unassign(course)

        self.backtracks += 1
        return None

    # -------------------------------------------------------------------
    # ENHANCEMENT 1 CHANGE -- Randomized MRV Tie-Breaking
    # -------------------------------------------------------------------

    def _select_unassigned_variable_random(self, assignment: Assignment,
                                           domains: dict) -> str:
        """
        ENHANCEMENT 1 -- MRV with RANDOMIZED tie-breaking.

        Baseline:  ties broken by degree heuristic (deterministic)
        Enhancement 1: ties broken by random shuffle (non-deterministic)

        Why this matters:
            Each restart uses a different random seed, so tied variables
            are broken differently each run. This causes the solver to
            explore a different region of the search space every time,
            dramatically increasing the chance of finding a solution
            across multiple restarts.

        Reference: AIMA 4th Ed., Section 6.3.1 (MRV) +
                   Section 4.3 (random restarts).
        """
        unassigned = [c for c in self.courses if c not in assignment.mapping]

        # Find the minimum domain size (MRV primary criterion)
        min_domain = min(len(domains[c]) for c in unassigned)

        # Collect all courses tied at minimum domain size
        tied = [c for c in unassigned if len(domains[c]) == min_domain]

        # RANDOMIZED tie-break -- shuffle the tied group and pick first
        # This is the key difference from the baseline degree tie-break
        self.rng.shuffle(tied)
        return tied[0]

    # -------------------------------------------------------------------
    # LCV, Forward Checking, Conflicts -- same as baseline
    # -------------------------------------------------------------------

    def _order_domain_values(self, course: str, assignment: Assignment,
                             domains: dict) -> list:
        """LCV heuristic -- identical to baseline."""
        unassigned = [c for c in self.courses
                      if c not in assignment.mapping and c != course]

        def lcv_score(triple):
            return sum(
                1 for other in unassigned
                for t in domains[other]
                if self._conflicts(course, triple, other, t)
            )

        return sorted(domains[course], key=lcv_score)

    def _conflicts(self, c1: str, t1: Triple, c2: str, t2: Triple) -> bool:
        """Conflict check -- identical to baseline."""
        if t1.timeslot != t2.timeslot:
            return False
        if t1.room == t2.room:
            return True
        if t1.instructor == t2.instructor:
            return True
        for curriculum in self.curricula:
            if c1 in curriculum and c2 in curriculum:
                return True
        return False

    def _forward_check(self, course: str, triple: Triple,
                       assignment: Assignment, domains: dict) -> Optional[dict]:
        """Forward Checking -- identical to baseline."""
        pruned    = defaultdict(list)
        unassigned = [c for c in self.courses if c not in assignment.mapping]

        for other_course in unassigned:
            to_remove = [
                t for t in domains[other_course]
                if self._conflicts(course, triple, other_course, t)
            ]
            for t in to_remove:
                domains[other_course].remove(t)
                pruned[other_course].append(t)

            if len(domains[other_course]) == 0:
                self._restore_pruned(pruned, domains)
                return None

        return pruned

    def _restore_pruned(self, pruned: Optional[dict], domains: dict):
        """Restore pruned domains on backtrack -- identical to baseline."""
        if pruned is None:
            return
        for course, triples in pruned.items():
            domains[course].extend(triples)

    def _build_initial_domains(self) -> dict:
        """Build domains with H3/H4 unary filtering -- identical to baseline."""
        domains = {}
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
            domains[course] = triples
        return domains


# -----------------------------------------------------------------------
# Enhancement 1 -- Restart Solver (Outer Loop)
# -----------------------------------------------------------------------

class RestartSolver:
    """
    ENHANCEMENT 1 -- Iterative Deepening with Restarts.

    Runs SingleRunCSP repeatedly with:
        - A new random seed each run (different search path)
        - A backtrack cutoff per run (abandon unproductive runs early)
        - Tracks the best solution found across all runs

    Path Cost (Enhancement 1):
        total_backtracks = sum of backtracks across all runs
        This is the Enhancement 1 path cost metric -- it measures
        total search effort spent to find the best feasible solution.
        Runs abandoned at cutoff contribute their full backtrack count.

    Experiment variables (for Experiment 1 & 2 in proposal):
        max_restarts : how many runs to attempt
        cutoff       : backtrack limit per run
    """

    def __init__(self, problem: dict, max_restarts: int = 10, cutoff: int = 100,
                 verbose: bool = True):
        self.problem      = problem
        self.max_restarts = max_restarts
        self.cutoff       = cutoff
        self.verbose      = verbose
        self.courses      = problem["courses"]

        # Enhancement 1 stats
        self.stats = {
            "total_backtracks":   0,   # path cost = total across all runs
            "total_nodes":        0,
            "runs_completed":     0,
            "runs_cutoff":        0,   # how many runs hit the cutoff
            "solution_found_run": None,# which run produced the solution
            "time_elapsed":       0.0,
            "best_path_cost":     0,   # assignments in winning solution
        }

        self.best_solution: Optional[Assignment] = None

    def solve(self) -> Optional[Assignment]:
        """
        Main loop -- run CSP up to max_restarts times.

        Each run gets a fresh random seed so MRV tie-breaking
        is different every time, exploring a new search path.

        Returns the best complete feasible solution found,
        or None if no run succeeded.
        """
        if self.verbose:
            print(f"\n  [Enhancement 1] Starting Iterative Deepening with Restarts")
            print(f"  Max restarts   : {self.max_restarts}")
            print(f"  Cutoff/run     : {self.cutoff} backtracks")
            print(f"  {'─' * 55}")
            print(f"  {'Run':<6} {'Seed':<8} {'Backtracks':<14} {'Nodes':<10} {'Result'}")
            print(f"  {'─' * 55}")

        start = time.time()

        for run in range(1, self.max_restarts + 1):
            seed = run * 42  # deterministic seeds for reproducibility

            # Create a fresh single run with this seed and cutoff
            csp = SingleRunCSP(self.problem, cutoff=self.cutoff, seed=seed)
            solution = csp.solve()

            # Accumulate path cost (total backtracks across all runs)
            self.stats["total_backtracks"] += csp.backtracks
            self.stats["total_nodes"]      += csp.nodes_visited
            self.stats["runs_completed"]    = run

            if csp.cutoff_hit:
                self.stats["runs_cutoff"] += 1
                status = "CUTOFF"
            elif solution is not None:
                status = "SOLUTION FOUND"
                if self.best_solution is None:
                    self.best_solution = solution
                    self.stats["solution_found_run"] = run
                    self.stats["best_path_cost"]     = solution.path_cost
            else:
                status = "no solution"

            if self.verbose:
                print(f"  {run:<6} {seed:<8} {csp.backtracks:<14} "
                      f"{csp.nodes_visited:<10} {status}")

            # Stop early if solution found and no need for more runs
            if self.best_solution is not None and run >= self.max_restarts:
                break

        self.stats["time_elapsed"] = round(time.time() - start, 3)

        if self.verbose:
            print(f"  {'─' * 55}")
        return self.best_solution

    def print_summary(self):
        """Print Enhancement 1 summary stats after solve() completes."""
        s = self.stats
        print(f"\n  ENHANCEMENT 1 -- SOLVER SUMMARY")
        print(f"  {'─' * 42}")
        print(f"  Runs attempted          : {s['runs_completed']}")
        print(f"  Runs hit cutoff         : {s['runs_cutoff']}")
        print(f"  Solution found on run   : {s['solution_found_run']}")
        print(f"  Total nodes visited     : {s['total_nodes']}")
        print(f"  Total backtracks        : {s['total_backtracks']}")
        print(f"  Path cost (backtracks)  : {s['total_backtracks']} across all runs")
        print(f"  Best solution cost      : {s['best_path_cost']} assignments")
        print(f"  Time elapsed            : {s['time_elapsed']}s")

        if self.best_solution:
            checker = ConstraintChecker(self.problem)
            report  = checker.validate_full(self.best_solution)
            print(f"\n  HARD CONSTRAINT VALIDATION (H1-H5)")
            print(f"  {'─' * 42}")
            print(report)
            if report.is_feasible():
                print(f"\n  ✓ All hard constraints satisfied.")
                print(f"  ✓ Schedule is FEASIBLE -- Enhancement 1 goal achieved.")
            else:
                print(f"\n  ✗ Violations found -- schedule INFEASIBLE.")
        else:
            print(f"\n  ✗ No feasible solution found across {s['runs_completed']} runs.")
            print(f"    Consider increasing max_restarts or raising the cutoff.")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("  Course Scheduling -- Enhancement 1: Iterative Deepening")
    print("  with Restarts")
    print("  CS 57200: Heuristic Problem Solving")
    print("  Deekshitha Reddy Muppidi | Track B: Optimization/Planning")
    print("=" * 62)

    problem = load_problem("data/sample_problem.json")

    print(f"\nProblem loaded:")
    print(f"  Courses     : {len(problem['courses'])}")
    print(f"  Rooms       : {len(problem['rooms'])}")
    print(f"  Timeslots   : {len(problem['timeslots'])}")
    print(f"  Instructors : {len(problem['instructors'])}")
    print(f"  Curricula   : {len(problem['curricula'])}")

    print(f"\nPath cost model (Enhancement 1):")
    print(f"  Total backtracks accumulated across all restart runs.")
    print(f"  No soft constraint optimization -- feasibility only.")

    # -------------------------------------------------------
    # Run Enhancement 1
    # Tune max_restarts and cutoff for your experiments:
    #   Experiment 1: vary max_restarts (5, 10, 20)
    #   Experiment 2: vary cutoff (50, 100, 200)
    # -------------------------------------------------------
    solver = RestartSolver(
        problem      = problem,
        max_restarts = 10,    # how many runs to attempt
        cutoff       = 100    # backtrack limit per run
    )

    solution = solver.solve()

    if solution:
        print(f"\n✓ Best solution found!\n")
        print_schedule(solution, problem)

    solver.print_summary()

    print("\nDone.")
