"""
display_all_schedules.py
========================
Display complete schedules for all test cases showing:
- Which instructor teaches which course
- At what timeslot
- In which room
- Curriculum conflict checks
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from data_loader import get_all_test_cases
from scheduler import CSPScheduler
from enhancement1_restarts import RestartSolver
from display import print_complete_schedule_report, save_schedule_to_file


def display_all_schedules():
    """Generate and display schedules for all test cases."""
    
    print("\n" + "█" * 85)
    print("  COURSE SCHEDULING CSP - COMPLETE SCHEDULE REPORT FOR ALL DATASETS")
    print("  CS 57200: Heuristic Problem Solving | Track B: Optimization/Planning")
    print("  Deekshitha Reddy Muppidi")
    print("█" * 85)

    test_cases = get_all_test_cases()
    
    for i, problem in enumerate(test_cases, 1):
        print(f"\n\n{'█' * 85}")
        print(f"  TEST CASE {i}: {problem['name']}")
        print(f"{'█' * 85}")
        
        # Problem statistics
        print(f"\n  📊 PROBLEM STATISTICS")
        print(f"  {'─' * 50}")
        print(f"  Courses           : {len(problem['courses'])}")
        print(f"  Rooms             : {len(problem['rooms'])}")
        print(f"  Timeslots         : {len(problem['timeslots'])}")
        print(f"  Instructors       : {len(problem['instructors'])}")
        print(f"  Curricula Groups  : {len(problem['curricula'])}")
        print(f"  Room-Slot Combos  : {len(problem['rooms']) * len(problem['timeslots'])}")
        
        # Instructor details
        print(f"\n  👨‍🏫 INSTRUCTOR ASSIGNMENTS")
        print(f"  {'─' * 50}")
        for instr_id, instr_info in problem['instructors'].items():
            qualified = instr_info.get('qualified_courses', [])
            max_slots = instr_info.get('max_slots', '∞')
            print(f"  {instr_id}:")
            print(f"      Qualified for {len(qualified)} courses: {', '.join(qualified[:3])}{'...' if len(qualified) > 3 else ''}")
            print(f"      Max courses per term: {max_slots}")
        
        # Room details
        print(f"\n  🏫 ROOM CAPACITIES")
        print(f"  {'─' * 50}")
        for room_id, capacity in problem['rooms'].items():
            print(f"  {room_id}: {capacity} seats")
        
        # Solve with Baseline
        print(f"\n  {'─' * 50}")
        print(f"  🔍 BASELINE CSP SOLUTION")
        print(f"  {'─' * 50}")
        
        scheduler = CSPScheduler(problem)
        solution = scheduler.solve()
        
        if solution:
            print_complete_schedule_report(solution, problem, scheduler.stats)
            # Save to file
            filename = f"schedule_testcase_{i}_baseline.txt"
            save_schedule_to_file(solution, problem, filename)
        else:
            print(f"\n  ❌ BASELINE CSP FAILED")
            print(f"     Backtracks: {scheduler.stats['backtracks']}")
            print(f"     Time: {scheduler.stats['time_elapsed']}s")
            print(f"     Reason: Deterministic MRV leads to dead ends")
            
            # Try Enhancement 1
            print(f"\n  {'─' * 50}")
            print(f"  🚀 ENHANCEMENT 1 SOLUTION (with Restarts)")
            print(f"  {'─' * 50}")
            
            enh_solver = RestartSolver(problem, max_restarts=10, cutoff=100, verbose=False)
            enh_solution = enh_solver.solve()
            
            if enh_solution:
                print_complete_schedule_report(enh_solution, problem, enh_solver.stats)
                filename = f"schedule_testcase_{i}_enhancement1.txt"
                save_schedule_to_file(enh_solution, problem, filename)
            else:
                print(f"\n  ❌ ENHANCEMENT 1 ALSO FAILED")
                print(f"     Total backtracks across runs: {enh_solver.stats['total_backtracks']}")
                print(f"     Runs attempted: {enh_solver.stats['runs_completed']}")
                print(f"     Consider increasing max_restarts or cutoff")
        
        print("\n" + "─" * 85)
        input("\n  Press Enter to continue to next test case...")


def display_single_test_case(test_case_number: int):
    """Display schedule for a specific test case (1-5)."""
    test_cases = get_all_test_cases()
    
    if test_case_number < 1 or test_case_number > len(test_cases):
        print(f"Invalid test case number. Choose 1-{len(test_cases)}")
        return
    
    problem = test_cases[test_case_number - 1]
    
    print(f"\n{'█' * 85}")
    print(f"  TEST CASE {test_case_number}: {problem['name']}")
    print(f"{'█' * 85}")
    
    # Show instructor-course assignments
    print(f"\n  📋 COURSE REQUIREMENTS")
    print(f"  {'─' * 50}")
    for course in problem['courses'][:10]:  # Show first 10
        enrollment = problem['enrollments'].get(course, 'N/A')
        print(f"  {course}: {enrollment} students")
    
    # Find which instructors can teach each course
    print(f"\n  👨‍🏫 INSTRUCTOR QUALIFICATIONS BY COURSE")
    print(f"  {'─' * 55}")
    for course in problem['courses'][:10]:
        qualified_instructors = []
        for instr_id, instr_info in problem['instructors'].items():
            if course in instr_info.get('qualified_courses', []):
                qualified_instructors.append(instr_id)
        print(f"  {course}: {', '.join(qualified_instructors) if qualified_instructors else 'NO QUALIFIED INSTRUCTOR!'}")
    
    solver = CSPScheduler(problem)
    solution = solver.solve()
    
    if solution:
        print_complete_schedule_report(solution, problem, solver.stats)
    else:
        print(f"\n  No solution found with baseline. Trying Enhancement 1...")
        enh_solver = RestartSolver(problem, max_restarts=10, cutoff=100, verbose=True)
        enh_solution = enh_solver.solve()
        if enh_solution:
            print_complete_schedule_report(enh_solution, problem, enh_solver.stats)


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 85)
    print("  COURSE SCHEDULING - SCHEDULE DISPLAY UTILITY")
    print("  ============================================")
    print("\n  Options:")
    print("    1. Display all test cases (full report)")
    print("    2. Display specific test case")
    print("    3. Run experiments and generate figures")
    print("    4. Quick summary (just instructor assignments)")
    print()
    
    choice = input("  Enter choice (1-4): ").strip()
    
    if choice == '1':
        display_all_schedules()
    elif choice == '2':
        tc = int(input("  Enter test case number (1-5): ").strip())
        display_single_test_case(tc)
    elif choice == '3':
        print("\n  Running experiments.py...")
        exec(open("experiments.py").read())
    elif choice == '4':
        from data_loader import get_all_test_cases
        from scheduler import CSPScheduler
        
        test_cases = get_all_test_cases()
        for i, problem in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"TEST CASE {i}: {problem['name']}")
            print(f"{'='*60}")
            
            solver = CSPScheduler(problem)
            solution = solver.solve()
            
            if solution:
                print(f"\n  INSTRUCTOR ASSIGNMENTS:")
                print(f"  {'─' * 40}")
                assignments_by_instructor = {}
                for course, triple in solution.mapping.items():
                    assignments_by_instructor.setdefault(triple.instructor, []).append((course, triple.timeslot))
                
                for instructor, courses in sorted(assignments_by_instructor.items()):
                    print(f"\n  {instructor}:")
                    for course, timeslot in sorted(courses, key=lambda x: x[1]):
                        print(f"      {course} at {timeslot}")
            else:
                print(f"\n  No solution found.")
    else:
        print("  Invalid choice.")