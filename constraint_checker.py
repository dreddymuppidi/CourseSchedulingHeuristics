"""
constraint_checker.py
=====================
Hard constraint validation for the Course Scheduling CSP.

Hard Constraints (Schaerf, 1999; Bonutti et al., 2012):
    H1 — Instructor Conflict  : No instructor teaches two courses at the same timeslot
    H2 — Room Conflict        : No room hosts two courses at the same timeslot
    H3 — Room Capacity        : Room capacity >= course enrollment
    H4 — Instructor Qualified : Instructor must be qualified to teach the course
    H5 — Curriculum Conflict  : No student group has two courses at the same timeslot

References:
    [2] Schaerf, A. (1999). A survey of automated timetabling.
        Artificial Intelligence Review, 13(2), 87-127.
    [3] Bonutti et al. (2012). Benchmarking curriculum-based course timetabling.
        Annals of Operations Research, 194(1), 59-70.
"""

from dataclasses import dataclass, field


@dataclass
class ViolationReport:
    """
    Full audit report of hard constraint violations in a complete schedule.
    All counts should be 0 for a valid baseline solution.
    """
    h1_instructor_conflicts:  int = 0
    h2_room_conflicts:        int = 0
    h3_capacity_violations:   int = 0
    h4_qualification_errors:  int = 0
    h5_curriculum_conflicts:  int = 0

    @property
    def total(self) -> int:
        return (
            self.h1_instructor_conflicts +
            self.h2_room_conflicts +
            self.h3_capacity_violations +
            self.h4_qualification_errors +
            self.h5_curriculum_conflicts
        )

    def is_feasible(self) -> bool:
        """Returns True only if ALL hard constraints are satisfied."""
        return self.total == 0

    def __str__(self):
        return (
            f"  H1 Instructor Conflicts  : {self.h1_instructor_conflicts}\n"
            f"  H2 Room Conflicts        : {self.h2_room_conflicts}\n"
            f"  H3 Capacity Violations   : {self.h3_capacity_violations}\n"
            f"  H4 Qualification Errors  : {self.h4_qualification_errors}\n"
            f"  H5 Curriculum Conflicts  : {self.h5_curriculum_conflicts}\n"
            f"  {'─' * 38}\n"
            f"  Total Hard Violations    : {self.total}"
        )


class ConstraintChecker:
    """
    Validates hard constraints for the course scheduling CSP.

    Two modes of operation:

        Incremental (during search):
            is_consistent() — called inside BACKTRACK() for each candidate
            triple before it is assigned. Fast check against existing
            assignments only.

        Full validation (after search):
            validate_full() — complete audit of a finished schedule.
            Checks all constraint pairs. Used to verify the solution
            and report the ViolationReport.
    """

    def __init__(self, problem: dict):
        self.rooms       = problem["rooms"]        # dict: room_id -> capacity
        self.enrollments = problem["enrollments"]  # dict: course_id -> int
        self.instructors = problem["instructors"]  # dict: instr_id -> info
        self.curricula   = problem["curricula"]    # list of sets of course_ids

    # ------------------------------------------------------------------
    # Incremental Consistency Check (used during backtracking)
    # ------------------------------------------------------------------

    def is_consistent(self, course: str, triple, assignment) -> bool:
        """
        Check if assigning triple=(room, timeslot, instructor) to course
        is consistent with the current partial assignment.

        Only checks binary constraints (H1, H2, H5) against already-
        assigned courses. Unary constraints (H3, H4) are pre-filtered
        in _build_initial_domains() so are not repeated here.

        Returns True if the assignment is consistent, False otherwise.

        Reference: AIMA 4th Ed., Section 6.2.1 — CONSISTENT(assignment, csp).
        """
        for assigned_course, assigned_triple in assignment.mapping.items():

            # Only check courses at the same timeslot — different timeslots
            # cannot conflict on H1, H2, or H5
            if assigned_triple.timeslot != triple.timeslot:
                continue

            # H1: Instructor conflict — same instructor, same timeslot
            if assigned_triple.instructor == triple.instructor:
                return False

            # H2: Room conflict — same room, same timeslot
            if assigned_triple.room == triple.room:
                return False

            # H5: Curriculum conflict — same student group, same timeslot
            for curriculum in self.curricula:
                if course in curriculum and assigned_course in curriculum:
                    return False

        return True

    # ------------------------------------------------------------------
    # Full Solution Validation (used after search completes)
    # ------------------------------------------------------------------

    def validate_full(self, assignment) -> ViolationReport:
        """
        Complete hard constraint audit of a finished schedule.

        Checks all (course_i, course_j) pairs for binary constraint
        violations and each course individually for unary violations.

        Returns a ViolationReport with per-constraint violation counts.
        A valid baseline solution must have report.total == 0.

        Reference: ITC-2007 validator specification (Bonutti et al., 2012).
        """
        report = ViolationReport()
        items  = list(assignment.mapping.items())

        for i, (course_i, triple_i) in enumerate(items):

            # H3: Room capacity check
            capacity   = self.rooms.get(triple_i.room, 0)
            enrollment = self.enrollments.get(course_i, 0)
            if capacity < enrollment:
                report.h3_capacity_violations += 1

            # H4: Instructor qualification check
            instr_info = self.instructors.get(triple_i.instructor, {})
            if course_i not in instr_info.get("qualified_courses", []):
                report.h4_qualification_errors += 1

            # Pairwise checks — only against courses not yet checked (j > i)
            for j, (course_j, triple_j) in enumerate(items):
                if i >= j:
                    continue  # Avoid double-counting pairs

                if triple_i.timeslot != triple_j.timeslot:
                    continue  # Different timeslots — no binary conflict possible

                # H1: Instructor conflict
                if triple_i.instructor == triple_j.instructor:
                    report.h1_instructor_conflicts += 1

                # H2: Room conflict
                if triple_i.room == triple_j.room:
                    report.h2_room_conflicts += 1

                # H5: Curriculum conflict
                for curriculum in self.curricula:
                    if course_i in curriculum and course_j in curriculum:
                        report.h5_curriculum_conflicts += 1
                        break  # Count once per pair per timeslot

        return report
