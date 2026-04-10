"""
display.py
==========
Output utilities for schedule display and solver statistics.
"""

import sys
from constraint_checker import ConstraintChecker
from collections import defaultdict

# Fix for Windows terminals that default to cp1252 encoding
sys.stdout.reconfigure(encoding="utf-8")


def print_schedule(assignment, problem: dict):
    """Print the final schedule grouped by timeslot."""
    if not assignment or not assignment.mapping:
        print("  (No assignments to display)")
        return

    # Group assignments by timeslot
    by_timeslot = {}
    for course, triple in assignment.mapping.items():
        by_timeslot.setdefault(triple.timeslot, []).append((course, triple))

    print("─" * 80)
    print(f"  {'TIMESLOT':<18} {'COURSE':<12} {'ROOM':<10} {'INSTRUCTOR':<20}")
    print("─" * 80)

    for ts in sorted(by_timeslot.keys()):
        entries = sorted(by_timeslot[ts], key=lambda x: x[0])
        for i, (course, triple) in enumerate(entries):
            slot_label = ts if i == 0 else ""
            print(f"  {slot_label:<18} {course:<12} {triple.room:<10} {triple.instructor:<20}")
        print("─" * 80)


def print_stats(stats: dict, assignment, problem: dict):
    """Print solver performance statistics and full constraint audit."""
    checker = ConstraintChecker(problem)
    report = checker.validate_full(assignment)

    print("\n  SOLVER STATISTICS")
    print("  " + "─" * 42)
    print(f"  Nodes visited    : {stats.get('nodes_visited', 0)}")
    print(f"  Backtracks       : {stats.get('backtracks', 0)}")
    print(f"  Time elapsed     : {stats.get('time_elapsed', 0)}s")
    print(f"  Courses placed   : {stats.get('path_cost', len(assignment.mapping))} / {len(problem['courses'])}")
    print(f"  Path cost        : {stats.get('path_cost', len(assignment.mapping))} assignments made")

    print("\n  HARD CONSTRAINT VALIDATION (H1-H5)")
    print("  " + "─" * 42)
    print(report)

    if report.is_feasible():
        print("\n  ✓ All hard constraints satisfied.")
        print("  ✓ Schedule is FEASIBLE — baseline goal achieved.")
    else:
        print("\n  ✗ Hard constraint violations found — schedule INFEASIBLE.")


def print_schedule_by_instructor(assignment, problem: dict):
    """
    Print schedule grouped by instructor.
    Shows which courses each instructor teaches and at what times.
    """
    if not assignment or not assignment.mapping:
        print("  (No assignments to display)")
        return

    # Group by instructor
    by_instructor = defaultdict(list)
    for course, triple in assignment.mapping.items():
        by_instructor[triple.instructor].append((course, triple.timeslot, triple.room))

    print("\n" + "=" * 70)
    print("  SCHEDULE BY INSTRUCTOR")
    print("=" * 70)

    for instructor in sorted(by_instructor.keys()):
        print(f"\n  👨‍🏫 {instructor}")
        print(f"  {'─' * 55}")
        print(f"    {'COURSE':<12} {'TIMESLOT':<18} {'ROOM':<10}")
        print(f"    {'─' * 45}")
        for course, timeslot, room in sorted(by_instructor[instructor], key=lambda x: x[1]):
            print(f"    {course:<12} {timeslot:<18} {room:<10}")
        print(f"    {'─' * 45}")
        print(f"    Total courses: {len(by_instructor[instructor])}")


def print_schedule_by_timeslot_detailed(assignment, problem: dict):
    """
    Print detailed schedule showing all courses at each timeslot with full info.
    """
    if not assignment or not assignment.mapping:
        print("  (No assignments to display)")
        return

    # Group by timeslot
    by_timeslot = defaultdict(list)
    for course, triple in assignment.mapping.items():
        by_timeslot[triple.timeslot].append((course, triple))

    print("\n" + "=" * 85)
    print("  DETAILED SCHEDULE (by Timeslot)")
    print("=" * 85)

    for timeslot in sorted(by_timeslot.keys()):
        print(f"\n  📅 {timeslot.upper()}")
        print(f"  {'─' * 75}")
        print(f"    {'COURSE':<12} {'ROOM':<10} {'INSTRUCTOR':<20} {'ENROLLMENT':<12} {'CAPACITY'}")
        print(f"    {'─' * 75}")
        
        for course, triple in sorted(by_timeslot[timeslot], key=lambda x: x[0]):
            enrollment = problem["enrollments"].get(course, "N/A")
            capacity = problem["rooms"].get(triple.room, "N/A")
            print(f"    {course:<12} {triple.room:<10} {triple.instructor:<20} {enrollment:<12} {capacity}")
        
        print(f"    {'─' * 75}")
        print(f"    Total courses at this timeslot: {len(by_timeslot[timeslot])}")


def print_complete_schedule_report(assignment, problem: dict, solver_stats: dict = None):
    """
    Print a complete, well-formatted schedule report including:
    - Schedule by timeslot
    - Schedule by instructor
    - Constraint validation results
    - Solver statistics (if provided)
    """
    if not assignment or not assignment.mapping:
        print("\n  ✗ No valid schedule to display.")
        return

    print("\n" + "█" * 85)
    print(f"  COMPLETE COURSE SCHEDULE REPORT")
    print(f"  Problem: {problem.get('name', 'Course Scheduling Problem')}")
    print("█" * 85)

    # Section 1: Schedule by Timeslot
    print_schedule_by_timeslot_detailed(assignment, problem)

    # Section 2: Schedule by Instructor
    print_schedule_by_instructor(assignment, problem)

    # Section 3: Constraint Validation
    print("\n" + "=" * 85)
    print("  HARD CONSTRAINT VALIDATION (H1-H5)")
    print("=" * 85)
    checker = ConstraintChecker(problem)
    report = checker.validate_full(assignment)
    print(report)

    if report.is_feasible():
        print("\n  ✓ All hard constraints satisfied. Schedule is FEASIBLE.")
    else:
        print("\n  ✗ Hard constraint violations found. Schedule is INFEASIBLE.")

    # Section 4: Solver Statistics (if provided)
    if solver_stats:
        print("\n" + "=" * 85)
        print("  SOLVER STATISTICS")
        print("=" * 85)
        print(f"  Nodes visited    : {solver_stats.get('nodes_visited', 'N/A')}")
        print(f"  Backtracks       : {solver_stats.get('backtracks', 'N/A')}")
        print(f"  Time elapsed     : {solver_stats.get('time_elapsed', 'N/A')}s")
        print(f"  Courses placed   : {solver_stats.get('path_cost', len(assignment.mapping))} / {len(problem['courses'])}")

    print("\n" + "█" * 85)


def print_instructor_timetable(assignment, problem: dict):
    """
    Print a timetable-style view per instructor (like a weekly schedule).
    """
    if not assignment or not assignment.mapping:
        print("  (No assignments to display)")
        return

    # Get unique timeslots and instructors
    all_timeslots = sorted(set(t.timeslot for t in assignment.mapping.values()))
    all_instructors = sorted(set(t.instructor for t in assignment.mapping.values()))
    
    # Build grid: instructor -> timeslot -> course
    grid = defaultdict(lambda: defaultdict(list))
    for course, triple in assignment.mapping.items():
        grid[triple.instructor][triple.timeslot].append(course)
    
    print("\n" + "=" * 100)
    print("  INSTRUCTOR TIMETABLE (Who teaches what and when)")
    print("=" * 100)
    
    for instructor in all_instructors:
        print(f"\n  👨‍🏫 {instructor}")
        print(f"  {'─' * 90}")
        print(f"    {'TIMESLOT':<20} {'COURSES TAUGHT'}")
        print(f"    {'─' * 90}")
        
        for timeslot in all_timeslots:
            courses = grid[instructor].get(timeslot, [])
            if courses:
                courses_str = ", ".join(sorted(courses))
                print(f"    {timeslot:<20} {courses_str}")
            else:
                print(f"    {timeslot:<20} (free period)")
        
        print(f"    {'─' * 90}")
        total_courses = sum(len(c) for c in grid[instructor].values())
        print(f"    Total courses taught: {total_courses}")


def print_room_schedule(assignment, problem: dict):
    """
    Print schedule grouped by room (which courses use each room and when).
    """
    if not assignment or not assignment.mapping:
        print("  (No assignments to display)")
        return

    # Group by room
    by_room = defaultdict(list)
    for course, triple in assignment.mapping.items():
        by_room[triple.room].append((course, triple.timeslot, triple.instructor))
    
    print("\n" + "=" * 90)
    print("  ROOM SCHEDULE")
    print("=" * 90)
    
    for room in sorted(by_room.keys()):
        print(f"\n  🏫 {room} (Capacity: {problem['rooms'].get(room, 'N/A')})")
        print(f"  {'─' * 80}")
        print(f"    {'TIMESLOT':<18} {'COURSE':<12} {'INSTRUCTOR':<20}")
        print(f"    {'─' * 80}")
        
        for course, timeslot, instructor in sorted(by_room[room], key=lambda x: x[1]):
            print(f"    {timeslot:<18} {course:<12} {instructor:<20}")
        
        print(f"    {'─' * 80}")
        print(f"    Total courses in this room: {len(by_room[room])}")


def print_curriculum_check(assignment, problem: dict):
    """
    Print curriculum conflict check results.
    Shows which student groups have which courses at which times.
    """
    if not assignment or not assignment.mapping:
        print("  (No assignments to display)")
        return

    print("\n" + "=" * 90)
    print("  CURRICULUM CONFLICT CHECK")
    print("=" * 90)
    
    # Build mapping: curriculum index -> course -> timeslot
    curriculum_courses = defaultdict(dict)
    
    for idx, curriculum in enumerate(problem["curricula"]):
        for course in curriculum:
            if course in assignment.mapping:
                curriculum_courses[idx][course] = assignment.mapping[course].timeslot
    
    for idx, courses in curriculum_courses.items():
        print(f"\n  📚 Curriculum Group {idx + 1}:")
        print(f"  {'─' * 75}")
        
        # Group by timeslot to show potential conflicts
        by_timeslot = defaultdict(list)
        for course, timeslot in courses.items():
            by_timeslot[timeslot].append(course)
        
        has_conflict = False
        for timeslot, course_list in by_timeslot.items():
            if len(course_list) > 1:
                print(f"    ⚠️ CONFLICT at {timeslot}: {', '.join(course_list)}")
                has_conflict = True
            else:
                print(f"    ✓ {timeslot}: {course_list[0]}")
        
        if not has_conflict:
            print(f"    ✓ No conflicts - all courses at different times")
        
        print(f"    Total courses in curriculum: {len(courses)}")


def save_schedule_to_file(assignment, problem: dict, filename: str = "schedule_report.txt"):
    """
    Save complete schedule report to a text file.
    """
    import io
    
    # Capture print output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    print_complete_schedule_report(assignment, problem)
    output = sys.stdout.getvalue()
    
    sys.stdout = old_stdout
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"\n  📄 Schedule report saved to: {filename}")