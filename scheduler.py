"""
Course Scheduling with Constraint Satisfaction Problem (CSP)
============================================================
CS 57200: Heuristic Problem Solving — Project Milestone 2
Track B: Optimization / Planning
Student: Deekshitha Reddy Muppidi

Baseline Implementation: Backtracking CSP Solver
-------------------------------------------------
Path Cost (Baseline):
    The path cost is the number of assignments made to reach the
    current state — one unit per course assigned. The goal is to
    find the first complete, consistent assignment where all hard
    constraints are satisfied. All feasible solutions are treated
    as equally valid; the solver stops as soon as one is found.

Heuristics Used (Russell & Norvig, AIMA 4th Ed., Chapter 6):
    MRV : Minimum Remaining Values — assign most constrained course first
    LCV : Least Constraining Value  — pick triple that prunes fewest options
    FC  : Forward Checking          — prune domains after each assignment

References:
    [1] Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern
        Approach (4th ed.). Pearson. Chapters 4 & 6.
    [2] Schaerf, A. (1999). A survey of automated timetabling.
        Artificial Intelligence Review, 13(2), 87-127.
    [3] Bonutti et al. (2012). Benchmarking curriculum-based course timetabling.
        Annals of Operations Research, 194(1), 59-70.
    [4] ITC-2007 Benchmark Dataset.
        http://www.cs.qub.ac.uk/itc2007
"""

import sys
import time
import copy
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

# Fix for Windows terminals that default to cp1252 encoding
# This ensures checkmarks, dashes and other characters print correctly
sys.stdout.reconfigure(encoding="utf-8")

from data_loader import load_problem
from constraint_checker import ConstraintChecker
from display import print_schedule, print_stats


# ---------------------------------------------------------------------------
# Core Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Triple:
    """
    A candidate assignment for one course: (room, timeslot, instructor).

    Frozen so it can be used in sets and as dict keys.
    Represents one unit of path cost when assigned to a course.
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

    Path cost = len(mapping) — the number of courses successfully
    assigned so far. Increases by 1 with each assign() call.
    """
    mapping: dict = field(default_factory=dict)

    def assign(self, course: str, triple: Triple):
        """Assign a triple to a course — increments path cost by 1."""
        self.mapping[course] = triple

    def unassign(self, course: str):
        """Remove an assignment — decrements path cost by 1 (backtrack)."""
        self.mapping.pop(course, None)

    def is_complete(self, courses: list) -> bool:
        """
        Goal test: returns True when every course has been assigned.
        Reference: AIMA 4th Ed., Section 6.1 — CSP goal test.
        """
        return all(c in self.mapping for c in courses)

    @property
    def path_cost(self) -> int:
        """
        Baseline path cost = number of courses assigned so far.
        One unit per course assigned. No optimization — purely
        counts depth of the current search path.
        """
        return len(self.mapping)

    def copy(self):
        return Assignment(mapping=dict(self.mapping))


# ---------------------------------------------------------------------------
# CSP Solver
# ---------------------------------------------------------------------------

class CSPScheduler:
    """
    Backtracking CSP solver for course timetabling.

    Implements BACKTRACKING-SEARCH from AIMA 4th Ed., Fig 6.5,
    augmented with three heuristics:

        MRV (Section 6.3.1) — variable ordering
        LCV (Section 6.3.1) — value ordering
        FC  (Section 6.3.2) — inference / domain pruning

    Path Cost Model (Baseline):
        - Cost = number of assignments made (not optimized)
        - Goal = first complete hard-constraint-satisfying assignment
        - All feasible solutions treated as equally valid
        - Solver halts immediately on finding one solution
    """

    def __init__(self, problem: dict):
        self.courses     = problem["courses"]
        self.rooms       = problem["rooms"]
        self.timeslots   = problem["timeslots"]
        self.instructors = problem["instructors"]
        self.enrollments = problem["enrollments"]
        self.curricula   = problem["curricula"]
        self.checker     = ConstraintChecker(problem)

        # Build initial domains (unary constraint filtering applied here)
        self.initial_domains = self._build_initial_domains()

        # Stats — all tracked per solve() call
        self.stats = {
            "backtracks":    0,   # times a dead end was hit and search reversed
            "nodes_visited": 0,   # total course-assignment attempts
            "path_cost":     0,   # final path cost = courses placed (baseline)
            "time_elapsed":  0.0,
        }

    # ------------------------------------------------------------------
    # Public Entry Point
    # ------------------------------------------------------------------

    def solve(self) -> Optional[Assignment]:
        """
        BACKTRACKING-SEARCH(csp) — AIMA Fig 6.5.

        Returns a complete Assignment if a feasible solution exists,
        or None if the problem is over-constrained.

        Baseline path cost is recorded in self.stats["path_cost"]
        after the search completes.
        """
        # Reset stats for clean run
        self.stats = {
            "backtracks":    0,
            "nodes_visited": 0,
            "path_cost":     0,
            "time_elapsed":  0.0,
        }

        start      = time.time()
        assignment = Assignment()
        domains    = copy.deepcopy(self.initial_domains)

        result = self._backtrack(assignment, domains)

        self.stats["time_elapsed"] = round(time.time() - start, 3)
        if result is not None:
            # Baseline path cost = total assignments made in solution
            self.stats["path_cost"] = result.path_cost

        return result

    # ------------------------------------------------------------------
    # Core Backtracking — AIMA Fig 6.5
    # ------------------------------------------------------------------

    def _backtrack(self, assignment: Assignment, domains: dict) -> Optional[Assignment]:
        """
        BACKTRACK(assignment, csp) — recursive search with backtracking.

        At each step:
          1. Check if assignment is complete (goal test)
          2. Select next variable using MRV
          3. Order its values using LCV
          4. For each value: check consistency, apply FC, recurse
          5. On failure: restore domains and unassign (backtrack)
        """
        # Goal test — all courses assigned, solver stops immediately
        if assignment.is_complete(self.courses):
            return assignment

        self.stats["nodes_visited"] += 1

        # Step 2: SELECT-UNASSIGNED-VARIABLE via MRV + degree tie-break
        course = self._select_unassigned_variable(assignment, domains)

        # Step 3: ORDER-DOMAIN-VALUES via LCV
        ordered_triples = self._order_domain_values(course, assignment, domains)

        for triple in ordered_triples:

            # Step 4a: Consistency check against all existing assignments
            if self.checker.is_consistent(course, triple, assignment):

                # Assign — path cost increases by 1
                assignment.assign(course, triple)

                # Step 4b: Forward Checking — prune conflicting values
                pruned = self._forward_check(course, triple, assignment, domains)

                if pruned is not None:
                    # No domain wipe-out — recurse deeper
                    result = self._backtrack(assignment, domains)
                    if result is not None:
                        return result  # Solution found — propagate up

                # Step 5: Backtrack — restore domains and unassign
                self._restore_pruned(pruned, domains)
                assignment.unassign(course)  # path cost decreases by 1

        # All values exhausted — signal failure to caller
        self.stats["backtracks"] += 1
        return None

    # ------------------------------------------------------------------
    # MRV: Minimum Remaining Values (AIMA Section 6.3.1)
    # ------------------------------------------------------------------

    def _select_unassigned_variable(self, assignment: Assignment, domains: dict) -> str:
        """
        MRV heuristic: select the unassigned course with the smallest
        remaining domain — the "most constrained" variable.

        Rationale: courses with fewer valid options are more likely to
        cause failures. Selecting them first detects dead ends earlier
        and reduces overall backtracking (AIMA p.214).

        Tie-breaking: degree heuristic — prefer the course with the
        most constraints on other unassigned courses (AIMA p.215).

        Reference: Russell & Norvig (2020), Section 6.3.1.
        """
        unassigned = [c for c in self.courses if c not in assignment.mapping]

        def mrv_key(course):
            domain_size = len(domains[course])
            degree      = self._degree(course, assignment)
            # Primary: smallest domain | Tie-break: highest degree
            return (domain_size, -degree)

        return min(unassigned, key=mrv_key)

    def _degree(self, course: str, assignment: Assignment) -> int:
        """
        Degree heuristic: count how many unassigned courses share a
        curriculum (constraint) with this course.

        Higher degree = more influential variable = preferred tie-break.
        Reference: AIMA 4th Ed., p.215.
        """
        unassigned = set(self.courses) - set(assignment.mapping.keys())
        count = 0
        for curriculum in self.curricula:
            if course in curriculum:
                # Subtract 1 to exclude the course itself
                count += len(curriculum & unassigned) - 1
        return count

    # ------------------------------------------------------------------
    # LCV: Least Constraining Value (AIMA Section 6.3.1)
    # ------------------------------------------------------------------

    def _order_domain_values(self, course: str, assignment: Assignment, domains: dict) -> list:
        """
        LCV heuristic: order domain values by how few options they
        eliminate from neighboring unassigned courses.

        Rationale: trying the least constraining value first maximizes
        the chance of finding a solution without backtracking (AIMA p.215).

        LCV score = number of values pruned from other domains if this
        triple is assigned. Lower score = less constraining = preferred.

        Reference: Russell & Norvig (2020), Section 6.3.1.
        """
        unassigned = [c for c in self.courses
                      if c not in assignment.mapping and c != course]

        def lcv_score(triple):
            pruned_count = 0
            for other_course in unassigned:
                for other_triple in domains[other_course]:
                    if self._conflicts(course, triple, other_course, other_triple):
                        pruned_count += 1
            return pruned_count

        return sorted(domains[course], key=lcv_score)

    def _conflicts(self, c1: str, t1: Triple, c2: str, t2: Triple) -> bool:
        """
        Returns True if assigning t1 to c1 conflicts with t2 assigned to c2.

        Used by LCV to estimate constraint impact, and by FC to prune domains.
        Checks three hard constraints at the same timeslot:
            H1 — same instructor
            H2 — same room
            H5 — same curriculum group (student conflict)
        """
        if t1.timeslot != t2.timeslot:
            return False  # Different timeslots — no conflict possible

        if t1.room == t2.room:
            return True   # H2: Room conflict

        if t1.instructor == t2.instructor:
            return True   # H1: Instructor conflict

        # H5: Curriculum conflict — same student group at same time
        for curriculum in self.curricula:
            if c1 in curriculum and c2 in curriculum:
                return True

        return False

    # ------------------------------------------------------------------
    # Forward Checking (AIMA Section 6.3.2)
    # ------------------------------------------------------------------

    def _forward_check(self, course: str, triple: Triple,
                       assignment: Assignment, domains: dict) -> Optional[dict]:
        """
        Forward Checking: after assigning triple to course, scan all
        unassigned courses and remove domain values that now conflict.

        Returns:
            dict  — {course: [pruned triples]} for undo on backtrack
            None  — a domain was wiped out (failure, backtrack immediately)

        Rationale: catching dead ends one step ahead avoids wasting time
        exploring branches that cannot lead to a solution (AIMA p.216-217).

        Reference: Russell & Norvig (2020), Section 6.3.2.
        """
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

            # Domain wipe-out detected — restore and signal failure
            if len(domains[other_course]) == 0:
                self._restore_pruned(pruned, domains)
                return None

        return pruned

    def _restore_pruned(self, pruned: Optional[dict], domains: dict):
        """
        Undo all domain pruning done during Forward Checking.
        Called on every backtrack to restore domains to pre-assignment state.
        """
        if pruned is None:
            return
        for course, triples in pruned.items():
            domains[course].extend(triples)

    # ------------------------------------------------------------------
    # Initial Domain Builder
    # ------------------------------------------------------------------

    def _build_initial_domains(self) -> dict:
        """
        Build the initial domain for each course by filtering out
        triples that violate unary constraints:

            H3 — Room capacity must be >= course enrollment
            H4 — Instructor must be qualified to teach the course

        These are checked once upfront rather than during search,
        reducing domain sizes before backtracking begins.
        """
        domains = {}
        for course in self.courses:
            enrollment = self.enrollments.get(course, 0)
            triples    = []

            for room, capacity in self.rooms.items():
                if capacity < enrollment:
                    continue  # H3: Room too small — skip entirely

                for timeslot in self.timeslots:
                    for instr_id, instr_info in self.instructors.items():
                        qualified = instr_info.get("qualified_courses", [])
                        if course not in qualified:
                            continue  # H4: Not qualified — skip

                        triples.append(Triple(
                            room=room,
                            timeslot=timeslot,
                            instructor=instr_id
                        ))

            domains[course] = triples

        return domains


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 62)
    print("  Course Scheduling — CSP Baseline Solver")
    print("  CS 57200: Heuristic Problem Solving")
    print("  Deekshitha Reddy Muppidi | Track B: Optimization/Planning")
    print("=" * 62)

    # Load problem — uses sample if no file found
    problem = load_problem("data/sample_problem.json")

    print(f"\nProblem loaded:")
    print(f"  Courses     : {len(problem['courses'])}")
    print(f"  Rooms       : {len(problem['rooms'])}")
    print(f"  Timeslots   : {len(problem['timeslots'])}")
    print(f"  Instructors : {len(problem['instructors'])}")
    print(f"  Curricula   : {len(problem['curricula'])}")

    # Show instructor details
    print(f"\nInstructor Details:")
    for instr_id, instr_info in problem['instructors'].items():
        qualified = instr_info.get('qualified_courses', [])
        max_slots = instr_info.get('max_slots', '∞')
        print(f"  👨‍🏫 {instr_id}:")
        print(f"       Qualified for: {', '.join(qualified[:5])}{'...' if len(qualified) > 5 else ''}")
        print(f"       Max courses: {max_slots}")

    # Show room capacities
    print(f"\nRoom Details:")
    for room_id, capacity in problem['rooms'].items():
        print(f"  🏫 {room_id}: Capacity {capacity}")

    # Show domain sizes before search
    solver = CSPScheduler(problem)
    total_triples = sum(len(v) for v in solver.initial_domains.values())
    print(f"\nInitial domain sizes (after unary filtering):")
    for course, triples in list(solver.initial_domains.items())[:5]:  # Show first 5
        print(f"  {course:<10} : {len(triples)} valid triples")
    if len(solver.initial_domains) > 5:
        print(f"  ... and {len(solver.initial_domains) - 5} more courses")
    print(f"  {'TOTAL':<10} : {total_triples} triples across all courses")

    print(f"\nRunning Backtracking CSP...")
    print(f"  Heuristics: MRV + Degree tie-break + LCV + Forward Checking")
    print(f"  Path cost model: 1 unit per course assigned (no optimization)\n")

    solution = solver.solve()

    if solution:
        print(f"✓ Solution found!\n")
        # Use the detailed report function
        from display import print_complete_schedule_report
        print_complete_schedule_report(solution, problem, solver.stats)
        
        # Also save to file
        from display import save_schedule_to_file
        save_schedule_to_file(solution, problem, "baseline_schedule_report.txt")
    else:
        print(f"✗ No solution found — problem is over-constrained.")
        print(f"  Backtracks made : {solver.stats['backtracks']}")
        print(f"  Time elapsed    : {solver.stats['time_elapsed']}s")

    print("\nDone.")