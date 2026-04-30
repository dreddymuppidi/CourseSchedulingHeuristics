"""
data_loader.py
==============
CS 57200: Heuristic Problem Solving -- Project Milestone 2
Track B: Optimization / Planning
Student: Deekshitha Reddy Muppidi

Three test cases for M2 experimental evaluation:
    TC1 -- Easy   (10 courses, 4 rooms, 10 slots) -- correctness check
    TC2 -- Medium (20 courses, 6 rooms, 15 slots) -- diversity demonstration
    TC3 -- Hard   (15 courses, 2 rooms,  8 slots) -- real backtracking

TC3 design: randomly structured curricula (seed=27) produce 1226 backtracks
on the baseline. Enhancement 1's random restarts find feasible paths faster
by trying different variable orderings. This is the key comparison problem.

References:
    [3] Bonutti et al. (2012). Benchmarking curriculum-based course timetabling.
        Annals of Operations Research, 194(1), 59-70.
    [4] ITC-2007 Dataset: http://www.cs.qub.ac.uk/itc2007
"""

import json
import os
import random


def load_problem(filepath: str) -> dict:
    """Load from JSON file. Falls back to TC1 if not found."""
    if not os.path.exists(filepath):
        print(f"[data_loader] '{filepath}' not found -- using TC1.")
        return generate_test_case_small()
    with open(filepath, "r") as f:
        raw = json.load(f)
    raw["curricula"] = [set(g) for g in raw.get("curricula", [])]
    return raw


# ---------------------------------------------------------------------------
# Test Case 1 -- Easy (10 courses)
# Both algorithms solve instantly. Used for correctness + Enhancement 2 demo.
# ---------------------------------------------------------------------------

def generate_test_case_small() -> dict:
    """
    TC1: 10 courses, 4 rooms, 10 timeslots, 4 instructors, 3 curricula.
    Expected baseline: 0 backtracks, <0.05s.
    Both algorithms solve instantly. Primary test case for Enhancement 2 demo
    because rich timeslot structure gives GLS room to improve soft constraints.
    """
    courses   = ["CS100","CS200","CS300","CS400","CS500",
                 "CS600","CS700","CS800","CS900","CS1000"]
    rooms     = {"Room_A":50, "Room_B":35, "Room_C":25, "Room_D":60}
    timeslots = ["MWF_8am","MWF_9am","MWF_10am","MWF_11am","MWF_12pm",
                 "TTh_8am","TTh_9am","TTh_10am","TTh_11am","TTh_12pm"]
    instructors = {
        "Prof_Smith": {"qualified_courses":["CS100","CS200","CS300","CS400"], "max_slots":3},
        "Prof_Jones": {"qualified_courses":["CS500","CS600","CS700"],         "max_slots":3},
        "Prof_Lee":   {"qualified_courses":["CS800","CS900","CS100","CS500"], "max_slots":3},
        "Prof_Patel": {"qualified_courses":["CS200","CS400","CS600","CS800","CS1000"],"max_slots":3},
    }
    enrollments = {
        "CS100":45,"CS200":30,"CS300":20,"CS400":55,"CS500":25,
        "CS600":35,"CS700":15,"CS800":28,"CS900":40,"CS1000":22,
    }
    curricula = [
        {"CS100","CS200","CS300"},
        {"CS400","CS500","CS600"},
        {"CS700","CS800","CS900"},
    ]
    return {
        "name":"Test Case 1 - Easy (10 courses)",
        "courses":courses,"rooms":rooms,"timeslots":timeslots,
        "instructors":instructors,"enrollments":enrollments,"curricula":curricula,
    }


# ---------------------------------------------------------------------------
# Test Case 2 -- Medium (20 courses)
# Both solve cleanly. Key use: solution diversity demonstration for Exp 3.
# Enhancement 1 produces 100% different schedules across random seeds.
# ---------------------------------------------------------------------------

def generate_test_case_medium() -> dict:
    """
    TC2: 20 courses, 6 rooms, 15 timeslots, 5 instructors, 4 curricula.
    Expected baseline: 0 backtracks.
    Primary use: Experiment 3 -- solution diversity. Enhancement 1
    finds completely different valid schedules per restart seed, giving
    Enhancement 2 (GLS) diverse starting points.
    """
    courses = [f"CS{100+i*100}" for i in range(20)]
    rooms   = {"Room_A":60,"Room_B":45,"Room_C":35,
               "Room_D":70,"Room_E":30,"Room_F":50}
    timeslots = ["MWF_8am","MWF_9am","MWF_10am","MWF_11am","MWF_12pm",
                 "MWF_1pm","MWF_2pm","MWF_3pm",
                 "TTh_8am","TTh_9am","TTh_10am",
                 "TTh_11am","TTh_12pm","TTh_1pm","TTh_2pm"]
    instructors = {
        "Prof_Smith": {"qualified_courses":["CS100","CS200","CS300","CS400","CS500","CS600"],"max_slots":4},
        "Prof_Jones": {"qualified_courses":["CS700","CS800","CS900","CS1000","CS1100","CS1200"],"max_slots":4},
        "Prof_Lee":   {"qualified_courses":["CS100","CS300","CS500","CS1300","CS1400","CS1500"],"max_slots":4},
        "Prof_Patel": {"qualified_courses":["CS200","CS400","CS600","CS800","CS1600","CS1700"],"max_slots":4},
        "Prof_Kim":   {"qualified_courses":["CS1100","CS1200","CS1300","CS1800","CS1900","CS2000"],"max_slots":4},
    }
    enrollments = {
        "CS100":55,"CS200":40,"CS300":30,"CS400":65,"CS500":35,
        "CS600":45,"CS700":25,"CS800":38,"CS900":50,"CS1000":32,
        "CS1100":28,"CS1200":42,"CS1300":35,"CS1400":20,"CS1500":48,
        "CS1600":33,"CS1700":27,"CS1800":41,"CS1900":36,"CS2000":29,
    }
    curricula = [
        {"CS100","CS200","CS300","CS400"},
        {"CS500","CS600","CS700","CS800"},
        {"CS900","CS1000","CS1100","CS1200"},
        {"CS1300","CS1400","CS1500","CS1600"},
    ]
    return {
        "name":"Test Case 2 - Medium (20 courses)",
        "courses":courses,"rooms":rooms,"timeslots":timeslots,
        "instructors":instructors,"enrollments":enrollments,"curricula":curricula,
    }


# ---------------------------------------------------------------------------
# Test Case 3 -- Hard (15 courses, produces real backtracking)
#
# Design: randomly generated with seed=27.
# Baseline produces 1226 backtracks -- measurably harder than TC1/TC2.
# Enhancement 1 reduces backtracks via random restarts that escape
# the deterministic MRV ordering that leads to dead ends.
# Enhancement 2 runs on this case to show F(S) improvement on a harder problem.
# ---------------------------------------------------------------------------

def generate_test_case_hard() -> dict:
    """
    TC3: 15 courses, 2 rooms, 8 timeslots, 3 instructors.
    Randomly generated (seed=27) -- produces 1226 backtracks on baseline.

    Why this causes backtracking:
        - Only 2 rooms and 8 timeslots = 16 slots for 15 courses (very tight)
        - Instructors each qualified for ~5 courses (non-uniform coverage)
        - Random curriculum groupings create unpredictable constraint chains
        - The deterministic MRV+degree tie-break picks a suboptimal ordering
          that requires backtracking before finding a feasible solution
        - Enhancement 1's random MRV tie-breaking finds better orderings
          faster via restart diversity

    Expected:
        Baseline:       ~1226 backtracks, ~0.1s
        Enhancement 1:  variable per seed, some runs 0 BT, overall faster
                        when runs that hit dead ends are cut off early
    """
    rng     = random.Random(27)
    nc, nr, nt, ni = 15, 2, 8, 3
    courses = [f"C{i}" for i in range(nc)]
    rooms   = {f"R{i}": 60 for i in range(nr)}
    slots   = [f"T{i}" for i in range(nt)]

    instructors = {}
    per = max(2, nc // ni)
    for i in range(ni):
        start = i * per
        qual  = courses[start:min(start+per, nc)]
        instructors[f"P{i}"] = {"qualified_courses": list(qual), "max_slots": nt}
    # Ensure all courses covered
    for c in courses:
        if not any(c in v["qualified_courses"] for v in instructors.values()):
            instructors["P0"]["qualified_courses"].append(c)

    enrollments = {c: 40 for c in courses}

    # Random curricula
    curricula = []
    shuffled  = courses[:]
    rng.shuffle(shuffled)
    i = 0
    while i < len(shuffled) - 1:
        size = rng.randint(2, min(4, max(2, len(shuffled)-i)))
        curricula.append(set(shuffled[i:i+size]))
        i += size

    return {
        "name":"Test Case 3 - Hard (15 courses, 1226 BT baseline)",
        "courses":courses,"rooms":rooms,"timeslots":slots,
        "instructors":instructors,"enrollments":enrollments,"curricula":curricula,
    }



def generate_test_case_hard_100() -> dict:
    """
    TC4: Hard 100-course benchmark.
    Designed to be feasible but highly constrained so that baseline,
    restart-based search, and GLS show measurable differences.
    """
    rng = random.Random(100)
    nc, nr, nt, ni = 100, 8, 20, 10
    courses = [f"C{i:03d}" for i in range(1, nc+1)]
    rooms = {f"R{i}": cap for i, cap in enumerate([30,40,50,60,80,100,120,150], start=1)}
    slots = [f"D{d}_T{t}" for d in range(1, 6) for t in range(1, 5)]  # 20 slots

    instructors = {}
    chunk = nc // ni
    for i in range(ni):
        start = i * chunk
        end = nc if i == ni - 1 else (i + 1) * chunk
        primary = courses[start:end]
        overlap_start = max(0, start - 2)
        overlap_end = min(nc, end + 2)
        overlap = courses[overlap_start:overlap_end]
        instructors[f"P{i+1}"] = {
            "qualified_courses": sorted(set(primary + overlap)),
            "max_slots": 10
        }

    enrollments = {c: rng.choice([25, 30, 35, 40, 45, 60, 75, 90]) for c in courses}

    curricula = []
    for i in range(0, nc, 5):
        group = set(courses[i:i+5])
        if i + 7 < nc:
            group.update(courses[i+5:i+7])  # intentional overlap
        curricula.append(group)

    return {
        "name": "Test Case 4 - Hard 100",
        "courses": courses,
        "rooms": rooms,
        "timeslots": slots,
        "instructors": instructors,
        "enrollments": enrollments,
        "curricula": curricula,
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_solution_differences(sol_a, sol_b, courses) -> int:
    """Count courses assigned differently between two solutions."""
    if sol_a is None or sol_b is None:
        return -1
    return sum(1 for c in courses if sol_a.mapping.get(c) != sol_b.mapping.get(c))


def get_all_test_cases() -> list:
    """Return all 3 test cases for M2 experimental evaluation."""
    return [
        generate_test_case_small(),
        generate_test_case_medium(),
        generate_test_case_hard(),
        generate_test_case_hard_100(),
    ]


def save_problem(problem: dict, filepath: str):
    """Serialize a problem to JSON."""
    s = dict(problem)
    s["curricula"] = [list(g) for g in problem["curricula"]]
    s.pop("name", None)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(s, f, indent=2)
    print(f"[data_loader] Saved to {filepath}")
