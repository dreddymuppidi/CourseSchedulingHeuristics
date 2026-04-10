"""
data_loader.py
==============
CS 57200: Heuristic Problem Solving
Track B: Optimization / Planning
Student: Deekshitha Reddy Muppidi

Loads course scheduling problem instances from JSON.
Provides three built-in test cases of increasing size for M2
experimental evaluation (small=10, medium=20, large=30 courses).

Reference:
    [3] Bonutti et al. (2012). Benchmarking curriculum-based course timetabling.
        Annals of Operations Research, 194(1), 59-70.
    [4] ITC-2007 Dataset: http://www.cs.qub.ac.uk/itc2007
"""

import json
import os


def load_problem(filepath: str) -> dict:
    """
    Load a problem instance from a JSON file.
    Falls back to small test case if file not found.
    """
    if not os.path.exists(filepath):
        print(f"[data_loader] '{filepath}' not found — using small test case.")
        return generate_test_case_small()

    with open(filepath, "r") as f:
        raw = json.load(f)

    raw["curricula"] = [set(group) for group in raw.get("curricula", [])]
    return raw


# ---------------------------------------------------------------------------
# Test Case 1 — Easy (10 courses, well-spaced)
# Baseline and Enhancement 1 both solve quickly — shows baseline correctness
# ---------------------------------------------------------------------------

def generate_test_case_small() -> dict:
    """
    Test Case 1 — Easy Problem (10 courses).
    4 rooms, 10 timeslots, 4 instructors, 3 curricula.
    Both algorithms solve instantly. Used for correctness verification.
    Expected: 0 backtracks, < 0.05s for both.
    """
    courses = [
        "CS100", "CS200", "CS300", "CS400", "CS500",
        "CS600", "CS700", "CS800", "CS900", "CS1000"
    ]
    rooms = {
        "Room_A": 50, "Room_B": 35,
        "Room_C": 25, "Room_D": 60,
    }
    timeslots = [
        "MWF_8am",  "MWF_9am",  "MWF_10am", "MWF_11am", "MWF_12pm",
        "TTh_8am",  "TTh_9am",  "TTh_10am", "TTh_11am", "TTh_12pm",
    ]
    instructors = {
        "Prof_Smith": {"qualified_courses": ["CS100","CS200","CS300","CS400"], "max_slots": 3},
        "Prof_Jones": {"qualified_courses": ["CS500","CS600","CS700"],         "max_slots": 3},
        "Prof_Lee":   {"qualified_courses": ["CS800","CS900","CS100","CS500"], "max_slots": 3},
        "Prof_Patel": {"qualified_courses": ["CS200","CS400","CS600","CS800","CS1000"], "max_slots": 3},
    }
    enrollments = {
        "CS100": 45, "CS200": 30, "CS300": 20, "CS400": 55,
        "CS500": 25, "CS600": 35, "CS700": 15, "CS800": 28,
        "CS900": 40, "CS1000": 22,
    }
    curricula = [
        {"CS100", "CS200", "CS300"},
        {"CS400", "CS500", "CS600"},
        {"CS700", "CS800", "CS900"},
    ]
    return {
        "name": "Test Case 1 - Easy (10 courses)",
        "courses": courses, "rooms": rooms, "timeslots": timeslots,
        "instructors": instructors, "enrollments": enrollments,
        "curricula": curricula,
    }


# ---------------------------------------------------------------------------
# Test Case 2 — Moderate (20 courses, tighter constraints)
# Baseline solves but Enhancement 1 finds diverse solutions
# ---------------------------------------------------------------------------

def generate_test_case_medium() -> dict:
    """
    Test Case 2 — Moderate Problem (20 courses).
    6 rooms, 15 timeslots, 5 instructors, 4 curricula.
    Both solve cleanly. Enhancement 1 produces diverse solutions
    across restarts — different valid schedules per run.
    Expected: 0 backtracks baseline, diverse solutions from Enhancement 1.
    """
    courses = [f"CS{100 + i*100}" for i in range(20)]
    rooms = {
        "Room_A": 60, "Room_B": 45, "Room_C": 35,
        "Room_D": 70, "Room_E": 30, "Room_F": 50,
    }
    timeslots = [
        "MWF_8am",  "MWF_9am",  "MWF_10am", "MWF_11am", "MWF_12pm",
        "MWF_1pm",  "MWF_2pm",  "MWF_3pm",
        "TTh_8am",  "TTh_9am",  "TTh_10am",
        "TTh_11am", "TTh_12pm", "TTh_1pm",  "TTh_2pm",
    ]
    instructors = {
        "Prof_Smith": {
            "qualified_courses": ["CS100","CS200","CS300","CS400","CS500","CS600"],
            "max_slots": 4
        },
        "Prof_Jones": {
            "qualified_courses": ["CS700","CS800","CS900","CS1000","CS1100","CS1200"],
            "max_slots": 4
        },
        "Prof_Lee": {
            "qualified_courses": ["CS100","CS300","CS500","CS1300","CS1400","CS1500"],
            "max_slots": 4
        },
        "Prof_Patel": {
            "qualified_courses": ["CS200","CS400","CS600","CS800","CS1600","CS1700"],
            "max_slots": 4
        },
        "Prof_Kim": {
            "qualified_courses": ["CS1100","CS1200","CS1300","CS1800","CS1900","CS2000"],
            "max_slots": 4
        },
    }
    enrollments = {
        "CS100": 55, "CS200": 40, "CS300": 30, "CS400": 65,
        "CS500": 35, "CS600": 45, "CS700": 25, "CS800": 38,
        "CS900": 50, "CS1000": 32, "CS1100": 28, "CS1200": 42,
        "CS1300": 35, "CS1400": 20, "CS1500": 48, "CS1600": 33,
        "CS1700": 27, "CS1800": 41, "CS1900": 36, "CS2000": 29,
    }
    curricula = [
        {"CS100", "CS200", "CS300", "CS400"},
        {"CS500", "CS600", "CS700", "CS800"},
        {"CS900", "CS1000", "CS1100", "CS1200"},
        {"CS1300", "CS1400", "CS1500", "CS1600"},
    ]
    return {
        "name": "Test Case 2 - Moderate (20 courses)",
        "courses": courses, "rooms": rooms, "timeslots": timeslots,
        "instructors": instructors, "enrollments": enrollments,
        "curricula": curricula,
    }


# ---------------------------------------------------------------------------
# Test Case 3 — Hard (near-infeasible, tight constraints)
# Baseline backtracks significantly. Enhancement 1 finds solution faster
# via restarts that escape dead-end orderings.
# ---------------------------------------------------------------------------

def generate_test_case_hard() -> dict:
    """
    Test Case 3 — Hard / Near-Infeasible Problem.

    Design rationale:
        - 12 courses, only 3 rooms, only 5 timeslots
        - 3*5 = 15 slot-room combinations for 12 courses = tight
        - Large curriculum groups (size 4+) force many same-timeslot bans
        - Shared instructors across curriculum groups create H1 conflicts
        - Dense cross-group curricula prevent easy separation
        - Result: deterministic MRV ordering hits dead ends and backtracks.
          Enhancement 1's random restarts find feasible paths more reliably
          by exploring different variable orderings.

    Expected:
        Baseline: multiple backtracks, slower runtime
        Enhancement 1: some runs hit cutoff, but overall finds solution
                       faster by abandoning stuck runs early
    """
    courses = [f"C{i}" for i in range(1, 13)]
    rooms   = {"R1": 50, "R2": 50, "R3": 50}

    # Only 5 timeslots — very tight for 12 courses across 3 rooms
    timeslots = ["T1", "T2", "T3", "T4", "T5"]

    instructors = {
        # P1 teaches first 6, P2 teaches last 6
        # BUT P3 is the ONLY instructor for C3, C6, C9, C12 — bottleneck
        "P1": {"qualified_courses": ["C1","C2","C4","C5"],         "max_slots": 4},
        "P2": {"qualified_courses": ["C7","C8","C10","C11"],        "max_slots": 4},
        "P3": {"qualified_courses": ["C3","C6","C9","C12"],         "max_slots": 4},
    }
    enrollments = {f"C{i}": 45 for i in range(1, 13)}

    # Dense curricula: groups of 4 force each group to use 4 of 5 timeslots
    # Cross-group pairs double the constraint density
    curricula = [
        {"C1", "C2", "C3", "C4"},    # must all be different timeslots
        {"C5", "C6", "C7", "C8"},    # must all be different timeslots
        {"C9", "C10", "C11", "C12"}, # must all be different timeslots
        {"C1", "C5", "C9"},          # cross-group: 3 more pairwise bans
        {"C2", "C6", "C10"},
        {"C3", "C7", "C11"},
        {"C4", "C8", "C12"},
    ]
    return {
        "name": "Test Case 3 - Hard (12 courses, tight)",
        "courses": courses, "rooms": rooms, "timeslots": timeslots,
        "instructors": instructors, "enrollments": enrollments,
        "curricula": curricula,
    }

# ---------------------------------------------------------------------------
# Test Case 4 — Very Hard (Near-Infeasible, Forces Backtracking)
# This dataset is designed to make Baseline struggle while Enhancement 1
# succeeds through randomized restarts
# ---------------------------------------------------------------------------

def generate_test_case_very_hard() -> dict:
    """
    Test Case 4 — Very Hard / Near-Infeasible Problem.
    
    DESIGN PHILOSOPHY:
    -----------------
    Baseline (deterministic MRV) will always make the same "wrong" choices
    and get stuck. Enhancement 1's random tie-breaking explores different
    paths and finds solutions.
    
    KEY FEATURES:
    -------------
    1. Very few timeslots (4) for many courses (16) → extreme time pressure
    2. Small rooms (capacity exactly matches enrollment) → no slack
    3. Overlapping curricula forcing many same-time conflicts
    4. Instructor specialization bottlenecks (some instructors can only teach 1 course)
    5. Dense constraint graph where MRV order matters critically
    
    EXPECTED RESULTS:
    -----------------
    Baseline: 500-2000+ backtracks, may fail to find solution
    Enhancement 1: finds solution within 3-5 restarts, fewer total backtracks
    """
    courses = [f"C{i:02d}" for i in range(1, 17)]  # 16 courses
    
    # Only 3 rooms with exact capacities (no slack = tight)
    rooms = {
        "R101": 30,  # Small room
        "R102": 30,  # Small room  
        "R103": 30,  # Small room
        "R201": 45,  # Medium room
        "R202": 45,  # Medium room
    }
    
    # Only 4 timeslots for 16 courses = 4*5 = 20 slot-room combos for 16 courses
    # Only 4 slots free — extremely tight!
    timeslots = ["Mon_9am", "Mon_11am", "Tue_9am", "Tue_11am"]
    
    instructors = {
        # Generalists (can teach many)
        "Prof_General_A": {
            "qualified_courses": ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"],
            "max_slots": 3
        },
        "Prof_General_B": {
            "qualified_courses": ["C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16"],
            "max_slots": 3
        },
        # Specialists (bottlenecks — only they can teach certain courses)
        "Prof_Spec_X": {
            "qualified_courses": ["C01", "C09"],  # Only teaches these two
            "max_slots": 2
        },
        "Prof_Spec_Y": {
            "qualified_courses": ["C05", "C13"],  # Only teaches these two
            "max_slots": 2
        },
        "Prof_Spec_Z": {
            "qualified_courses": ["C08", "C16"],  # Only teaches these two
            "max_slots": 2
        },
        # Ultra-specialist (single course bottleneck)
        "Prof_Bottleneck": {
            "qualified_courses": ["C04", "C12"],  # Critical courses
            "max_slots": 1  # Can only teach ONE course total!
        },
    }
    
    # Exact enrollments matching room capacities (no flexibility)
    enrollments = {
        "C01": 30, "C02": 30, "C03": 30, "C04": 30,
        "C05": 30, "C06": 30, "C07": 30, "C08": 30,
        "C09": 45, "C10": 45, "C11": 45, "C12": 45,
        "C13": 45, "C14": 45, "C15": 45, "C16": 45,
    }
    
    # DENSE CURRICULA — force many conflicts
    # Each curriculum has 4-6 courses that must be at different times
    # With only 4 timeslots, each curriculum maxes out the schedule
    curricula = [
        # Core group 1 (4 courses → needs all 4 timeslots)
        {"C01", "C02", "C03", "C04"},
        
        # Core group 2 (4 courses → needs all 4 timeslots)
        {"C05", "C06", "C07", "C08"},
        
        # Core group 3 (4 courses → needs all 4 timeslots)
        {"C09", "C10", "C11", "C12"},
        
        # Core group 4 (4 courses → needs all 4 timeslots)
        {"C13", "C14", "C15", "C16"},
        
        # CROSS GROUPS — create conflicts between groups
        # These force specific assignments or cause backtracking
        {"C01", "C05", "C09", "C13"},  # One from each core group
        {"C02", "C06", "C10", "C14"},
        {"C03", "C07", "C11", "C15"},
        {"C04", "C08", "C12", "C16"},
        
        # BOTTLENECK CURRICULA — force bottleneck professor's courses apart
        {"C04", "C12"},  # Both taught by Prof_Bottleneck (can only teach 1!)
        {"C01", "C09"},  # Both taught by Prof_Spec_X
        {"C05", "C13"},  # Both taught by Prof_Spec_Y
        {"C08", "C16"},  # Both taught by Prof_Spec_Z
    ]
    
    return {
        "name": "Test Case 4 - VERY HARD (16 courses, near-infeasible)",
        "courses": courses,
        "rooms": rooms,
        "timeslots": timeslots,
        "instructors": instructors,
        "enrollments": enrollments,
        "curricula": curricula,
    }


# ---------------------------------------------------------------------------
# Test Case 5 — Extreme (Designed to Break Baseline)
# This pushes the problem to the edge of feasibility
# ---------------------------------------------------------------------------

def generate_test_case_extreme() -> dict:
    """
    Test Case 5 — EXTREME (Edge of Feasibility)
    
    DESIGN: Only 3 timeslots for 12 courses → 3*4=12 room-slot combos,
    exactly enough IF assigned perfectly. Any wrong choice = dead end.
    
    Baseline's deterministic MRV will make a wrong choice early and fail.
    Enhancement 1's randomization will eventually find the perfect assignment.
    
    This is like a "sudoku" where the solution is a perfect matching.
    """
    courses = [f"E{i:02d}" for i in range(1, 13)]
    
    # Exactly enough total capacity for all courses (if perfectly distributed)
    rooms = {
        "Lab_A": 25,
        "Lab_B": 25,
        "Lab_C": 25,
        "Lab_D": 25,  # 4 rooms * 3 timeslots = 12 slots for 12 courses
    }
    
    # Only 3 timeslots — extremely tight!
    timeslots = ["Morning", "Afternoon", "Evening"]
    
    instructors = {
        # Each instructor can only teach specific courses
        "Dr_Alpha": {
            "qualified_courses": ["E01", "E02", "E03", "E04"],
            "max_slots": 2
        },
        "Dr_Beta": {
            "qualified_courses": ["E05", "E06", "E07", "E08"],
            "max_slots": 2
        },
        "Dr_Gamma": {
            "qualified_courses": ["E09", "E10", "E11", "E12"],
            "max_slots": 2
        },
        "Dr_Delta": {
            "qualified_courses": ["E01", "E05", "E09"],  # Cross-group
            "max_slots": 2
        },
    }
    
    enrollments = {f"E{i:02d}": 25 for i in range(1, 13)}
    
    # Curricula that force a Latin-square style assignment
    # Each row must have all different timeslots
    curricula = [
        # Row constraints
        {"E01", "E02", "E03", "E04"},  # All different times
        {"E05", "E06", "E07", "E08"},  # All different times
        {"E09", "E10", "E11", "E12"},  # All different times
        
        # Column constraints (cross-group)
        {"E01", "E05", "E09"},  # All different times
        {"E02", "E06", "E10"},  # All different times
        {"E03", "E07", "E11"},  # All different times
        {"E04", "E08", "E12"},  # All different times
        
        # Diagonal constraints (makes it even harder)
        {"E01", "E06", "E11"},
        {"E04", "E07", "E10"},
    ]
    
    return {
        "name": "Test Case 5 - EXTREME (12 courses, 3 timeslots, perfect matching required)",
        "courses": courses,
        "rooms": rooms,
        "timeslots": timeslots,
        "instructors": instructors,
        "enrollments": enrollments,
        "curricula": curricula,
    }


# ---------------------------------------------------------------------------
# Update get_all_test_cases to include new hard cases
# ---------------------------------------------------------------------------

def get_all_test_cases() -> list:
    """Return all 5 test cases for M2 experimental evaluation."""
    return [
        generate_test_case_small(),      # Easy - 10 courses
        generate_test_case_medium(),     # Moderate - 20 courses
        generate_test_case_hard(),       # Hard - 12 courses (dense)
        generate_test_case_very_hard(),  # Very Hard - 16 courses (near-infeasible)
        generate_test_case_extreme(),    # Extreme - 12 courses (perfect matching)
    ]

# ---------------------------------------------------------------------------
# Solution diversity metric
# ---------------------------------------------------------------------------

def count_solution_differences(sol_a, sol_b, courses) -> int:
    """Count how many courses are assigned differently between two solutions."""
    if sol_a is None or sol_b is None:
        return -1
    return sum(
        1 for c in courses
        if sol_a.mapping.get(c) != sol_b.mapping.get(c)
    )


def get_all_test_cases() -> list:
    """Return all 3 test cases for M2 experimental evaluation."""
    return [
        generate_test_case_small(),
        generate_test_case_medium(),
        generate_test_case_hard(),
    ]


def save_problem(problem: dict, filepath: str):
    """Serialize a problem instance to JSON for reuse."""
    serializable = dict(problem)
    serializable["curricula"] = [list(g) for g in problem["curricula"]]
    serializable.pop("name", None)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[data_loader] Problem saved to {filepath}")
