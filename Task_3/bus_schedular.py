# Part 3: CSP Bus Scheduler

import time
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import copy
from collections import deque

print("=" * 65)
print("     PakTravel AI System - Part 3: Bus Scheduler (CSP)")
print("=" * 65)

# ============================================================
# TASK 1: CSP Modeling (formal definition)
# ============================================================
print("""
--- TASK 1: CSP Formal Model ---

VARIABLES:
  X = { R1, R2, R3, R4, R5, R6, R7, R8 }
  Each variable represents a bus route that needs a bus assigned.

DOMAINS:
  D(Ri) = { Bus1, Bus2, Bus3, Bus4, Bus5, Bus6, Bus7, Bus8, Bus9, Bus10 }
  Every route can be assigned any of the 10 available buses.

HARD CONSTRAINTS (must be satisfied):
  HC1 - Time Conflict: R1 and R7 are both at 8am  => R1 != R7
  HC2 - Time Conflict: R3 and R8 are both at 12pm => R3 != R8
  HC3 - Relay Guarantee: R3 MUST always be assigned (Ali's connecting bus!)
  HC4 - Driver Rest: No bus can operate more than 2 routes per day

SOFT CONSTRAINTS (maximize score):
  SC1 - Large coaches (Bus1, Bus2, Bus3) on highway routes (R1, R4, R5): +2 pts
  SC2 - Any bus operating <= 2 routes: +1 pt per compliant bus

ROUTES AND TIMES:
  R1: Karachi -> Hyderabad   | 8am
  R2: Hyderabad -> Sukkur    | 10am
  R3: Sukkur -> Multan       | 12pm  <-- ALI'S CONNECTING BUS!
  R4: Multan -> Lahore       | 2pm
  R5: Lahore -> Islamabad    | 4pm
  R6: Islamabad -> Peshawar  | 6pm
  R7: Karachi -> Quetta      | 8am
  R8: Quetta -> Gwadar       | 12pm
""")

# ============================================================
# Setup: Variables, Domains, Constraints
# ============================================================

routes = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8']
buses  = ['Bus1', 'Bus2', 'Bus3', 'Bus4', 'Bus5', 'Bus6',
          'Bus7', 'Bus8', 'Bus9', 'Bus10']

route_info = {
    'R1': {'name': 'Karachi -> Hyderabad',   'time': '8am'},
    'R2': {'name': 'Hyderabad -> Sukkur',    'time': '10am'},
    'R3': {'name': 'Sukkur -> Multan',       'time': '12pm'},
    'R4': {'name': 'Multan -> Lahore',       'time': '2pm'},
    'R5': {'name': 'Lahore -> Islamabad',    'time': '4pm'},
    'R6': {'name': 'Islamabad -> Peshawar',  'time': '6pm'},
    'R7': {'name': 'Karachi -> Quetta',      'time': '8am'},
    'R8': {'name': 'Quetta -> Gwadar',       'time': '12pm'},
}

# routes that conflict with each other (same time slot)
time_conflicts = {
    'R1': ['R7'],
    'R7': ['R1'],
    'R3': ['R8'],
    'R8': ['R3'],
}

# highway routes - large coaches preferred on these
highway_routes = ['R1', 'R4', 'R5']

# large coach buses
large_coaches = ['Bus1', 'Bus2', 'Bus3']


def get_initial_domains():
    """Returns a fresh copy of domains for all routes."""
    return {route: list(buses) for route in routes}


def is_consistent(assignment, route, bus):
    """
    Checks if assigning 'bus' to 'route' is consistent with current assignment.
    Enforces all hard constraints.
    """
    # HC1 / HC2 - time conflict check
    for conflict in time_conflicts.get(route, []):
        if conflict in assignment and assignment[conflict] == bus:
            return False  # same bus at same time - not allowed

    # HC4 - max 2 routes per bus (driver rest rule)
    bus_usage = sum(1 for b in assignment.values() if b == bus)
    if bus_usage >= 2:
        return False  # bus already doing 2 routes today

    return True


def get_valid_buses(assignment, route, domains):
    """Returns list of buses that can be validly assigned to this route."""
    return [bus for bus in domains[route] if is_consistent(assignment, route, bus)]


# ============================================================
# TASK 2: Backtracking + MRV Heuristic
# ============================================================
print("=" * 65)
print("--- TASK 2: Backtracking Search ---\n")

# ---- Plain Backtracking ----
bt_calls = 0  # global counter - tracking how many times we backtrack

def plain_backtrack(assignment, domains):
    """
    Plain backtracking search - assigns buses to routes one by one.
    If a conflict is found, backtracks and tries another bus.
    Routes are assigned in fixed order R1, R2, ... R8.
    """
    global bt_calls

    if len(assignment) == len(routes):
        return assignment  # all routes assigned, we're done!

    # pick next unassigned route (fixed order)
    route = next(r for r in routes if r not in assignment)

    for bus in domains[route]:
        if is_consistent(assignment, route, bus):
            assignment[route] = bus
            result = plain_backtrack(assignment, domains)
            if result is not None:
                return result
            # backtrack - this bus didn't work out
            del assignment[route]
            bt_calls += 1  # count each backtrack

    return None  # no valid assignment found from here


print("Running Plain Backtracking...")
bt_calls = 0
t0 = time.perf_counter()
plain_assignment = plain_backtrack({}, get_initial_domains())
plain_time = (time.perf_counter() - t0) * 1000
plain_bt_count = bt_calls
print(f"  Backtrack calls: {plain_bt_count}")
print(f"  Time: {plain_time:.4f} ms")
print(f"  Schedule found: {'YES' if plain_assignment else 'NO'}")

# ---- MRV Heuristic ----
# MRV = Minimum Remaining Values - assign the route with FEWEST valid bus options first
# this reduces the branching factor early and catches conflicts sooner

bt_calls = 0

def pick_mrv_route(assignment, domains):
    """
    MRV heuristic: selects the unassigned route with the smallest domain.
    Route with fewest valid bus options gets assigned first.
    This is smart because if a route only has 1-2 options, we should handle it early.
    HC3 enforcement: R3 (Ali's bus) gets priority when tied with others.
    """
    unassigned = [r for r in routes if r not in assignment]
    # count valid buses for each unassigned route
    # pick the one with minimum count
    # HC3: tie-break by giving R3 priority (0) over others (1)
    # this ensures Ali's connecting bus R3 is assigned as early as possible
    return min(unassigned, key=lambda r: (
        len([b for b in domains[r] if is_consistent(assignment, r, b)]),
        0 if r == 'R3' else 1  # HC3: R3 gets priority on ties
    ))


def mrv_backtrack(assignment, domains):
    """
    Backtracking with MRV heuristic.
    Same as plain backtracking but picks which route to assign next more cleverly.
    """
    global bt_calls

    if len(assignment) == len(routes):
        return assignment

    # use MRV to pick which route to assign next
    route = pick_mrv_route(assignment, domains)

    for bus in domains[route]:
        if is_consistent(assignment, route, bus):
            assignment[route] = bus
            result = mrv_backtrack(assignment, domains)
            if result is not None:
                return result
            del assignment[route]
            bt_calls += 1

    return None


bt_calls = 0
t0 = time.perf_counter()
mrv_assignment = mrv_backtrack({}, get_initial_domains())
mrv_time = (time.perf_counter() - t0) * 1000
mrv_bt_count = bt_calls
print(f"\nRunning MRV Backtracking...")
print(f"  Backtrack calls: {mrv_bt_count}")
print(f"  Time: {mrv_time:.4f} ms")
print(f"  Schedule found: {'YES' if mrv_assignment else 'NO'}")

improvement = plain_bt_count - mrv_bt_count
print(f"\nMRV improvement: {improvement} fewer backtrack calls than plain BT")
if improvement > 0:
    print(f"  MRV is smarter! Assigns constrained routes first.")
else:
    print(f"  Similar performance (small problem size).")

# print schedule from MRV result
def print_schedule(assignment, label="Schedule"):
    print(f"\n  {label}:")
    print(f"  {'Route':<6} {'Bus':<10} {'Time':<8} {'Route Name':<30} {'Status'}")
    print("  " + "-" * 70)
    for r in routes:
        bus = assignment.get(r, 'UNASSIGNED')
        info = route_info[r]
        status = ''
        if r == 'R3':
            status = '<-- ALI\'S BUS!'
        elif bus in large_coaches and r in highway_routes:
            status = '(large coach on highway)'
        print(f"  {r:<6} {bus:<10} {info['time']:<8} {info['name']:<30} {status}")


if mrv_assignment:
    print_schedule(mrv_assignment, "MRV Backtracking Schedule")
    ali_bus = mrv_assignment.get('R3', 'UNASSIGNED')
    print(f"\n  >>> Ali's connecting bus is {ali_bus} at 12pm ✓")


# ============================================================
# TASK 3: AC3 Arc Consistency
# ============================================================
print("\n" + "=" * 65)
print("--- TASK 3: AC3 Arc Consistency ---\n")

def ac3(domains):
    """
    AC3 Algorithm - Arc Consistency Algorithm 3.
    For each pair of conflicting routes (same time slot), ensure
    every value in one domain has a compatible value in the other domain.
    Returns True if domains are consistent, False if any domain becomes empty.
    """
    # arcs are the time-conflict pairs (both directions)
    # an arc (Xi, Xj) means "Xi's domain must be consistent with Xj"
    queue = deque()
    for r1, conflicts in time_conflicts.items():
        for r2 in conflicts:
            queue.append((r1, r2))

    iterations = 0
    while queue:
        xi, xj = queue.popleft()
        iterations += 1

        if revise(domains, xi, xj):
            if len(domains[xi]) == 0:
                print(f"  Domain of {xi} became empty! No solution possible.")
                return False
            # if we removed values from xi, re-add all arcs pointing to xi
            for xk, conflicts in time_conflicts.items():
                if xi in conflicts and xk != xj:
                    queue.append((xk, xi))

    print(f"  AC3 completed in {iterations} iterations.")
    return True


def revise(domains, xi, xj):
    """
    Remove values from domain of xi that have NO compatible value in xj.
    For our != constraint: value v is consistent if there exists w in Dj where w != v.
    So v is removed only if ALL values in Dj equal v (i.e., Dj = {v}).
    """
    removed = False
    for val in list(domains[xi]):
        # check if there is ANY value in xj's domain != val
        compatible = any(other != val for other in domains[xj])
        if not compatible:
            domains[xi].remove(val)
            removed = True
    return removed


# show domain sizes before AC3
print("Domain sizes BEFORE AC3:")
fresh_domains = get_initial_domains()
for r in routes:
    conflicts = time_conflicts.get(r, [])
    conflict_str = f" [conflicts with {', '.join(conflicts)}]" if conflicts else ""
    print(f"  {r}: {len(fresh_domains[r])} buses{conflict_str}")

# run AC3
print("\nRunning AC3...")
ac3_domains = get_initial_domains()
ac3_result = ac3(ac3_domains)

print("\nDomain sizes AFTER AC3:")
reduced_total = 0
for r in routes:
    before = len(buses)
    after  = len(ac3_domains[r])
    diff   = before - after
    reduced_total += diff
    change = f"  (-{diff})" if diff > 0 else "  (unchanged)"
    print(f"  {r}: {after} buses{change}")

print(f"\nTotal domain reductions by AC3: {reduced_total}")
if reduced_total == 0:
    print("  Note: AC3 made no reductions here because all domains start with")
    print("  10 buses and our != constraints can always be satisfied (9 other buses).")
    print("  AC3 is most useful when domains are already partially restricted.")

# run backtracking on AC3-reduced domains
bt_calls = 0
t0 = time.perf_counter()
ac3_assignment = mrv_backtrack({}, ac3_domains)
ac3_time = (time.perf_counter() - t0) * 1000
ac3_bt_count = bt_calls

# full comparison table
print("\nComparison Table:")
print(f"  {'Method':<15} {'BT Calls':>10} {'Time(ms)':>12} {'Schedule?':>10}")
print("  " + "-" * 50)
print(f"  {'Plain BT':<15} {plain_bt_count:>10} {plain_time:>12.4f} {'YES' if plain_assignment else 'NO':>10}")
print(f"  {'MRV':<15} {mrv_bt_count:>10} {mrv_time:>12.4f} {'YES' if mrv_assignment else 'NO':>10}")
print(f"  {'MRV + AC3':<15} {ac3_bt_count:>10} {ac3_time:>12.4f} {'YES' if ac3_assignment else 'NO':>10}")


# ============================================================
# TASK 4: Soft Constraints + Best Schedule
# ============================================================
print("\n" + "=" * 65)
print("--- TASK 4: Soft Constraints + Optimal Schedule ---\n")

def compute_soft_score(assignment):
    """
    Calculates soft constraint score for a given bus assignment.
    SC1: large bus (Bus1/2/3) on highway route (R1/R4/R5) = +2 pts each
    SC2: each bus operating <= 2 routes = +1 pt per compliant bus
    """
    score = 0
    details = []

    # SC1 - large coaches on highway routes
    for route in highway_routes:
        if route in assignment and assignment[route] in large_coaches:
            score += 2
            details.append(f"+2 pts: {assignment[route]} on highway {route}")

    # SC2 - driver rest compliance
    bus_usage = {}
    for r, b in assignment.items():
        bus_usage[b] = bus_usage.get(b, 0) + 1
    for bus, count in bus_usage.items():
        if count <= 2:
            score += 1
            details.append(f"+1 pt:  {bus} operating {count} route(s) (rest compliant)")

    return score, details


def find_best_schedule():
    """
    Finds the schedule with the highest soft constraint score.
    Phase 1: Greedy - try assigning large coaches (Bus1/2/3) to highway routes
             (R1/R4/R5) and fill remaining routes with distinct unused buses.
    Phase 2: Random - also check random alternatives via backtracking.
    Returns the best schedule found across both phases.
    """
    from itertools import permutations
    import random
    random.seed(42)

    best_score = -1
    best_assignment = None
    candidates_checked = 0

    # Phase 1: Try all permutations of large coaches on highway routes
    # Then fill remaining routes with distinct unused buses (maximizes SC2)
    for perm in permutations(large_coaches):
        # assign large coaches to highway routes
        partial = {highway_routes[i]: perm[i] for i in range(len(highway_routes))}
        used = set(partial.values())

        # fill remaining routes with distinct unused buses
        remaining = [r for r in routes if r not in partial]
        available = [b for b in buses if b not in used]
        assignment = dict(partial)
        valid = True

        for route in remaining:
            placed = False
            for bus in available:
                if bus not in assignment.values() and is_consistent(assignment, route, bus):
                    assignment[route] = bus
                    placed = True
                    break
            if not placed:
                valid = False
                break

        if valid and len(assignment) == len(routes):
            score, _ = compute_soft_score(assignment)
            candidates_checked += 1
            if score > best_score:
                best_score = score
                best_assignment = dict(assignment)

    # Phase 2: Also try random backtracking alternatives for comparison
    def bt_collect(assignment, domains, solutions, max_solutions=30):
        """Collects up to max_solutions valid assignments."""
        if len(solutions) >= max_solutions:
            return
        if len(assignment) == len(routes):
            solutions.append(dict(assignment))
            return

        route = pick_mrv_route(assignment, domains)
        shuffled = list(domains[route])
        random.shuffle(shuffled)

        for bus in shuffled:
            if is_consistent(assignment, route, bus):
                assignment[route] = bus
                bt_collect(assignment, domains, solutions, max_solutions)
                del assignment[route]

    solutions = []
    bt_collect({}, get_initial_domains(), solutions, max_solutions=30)
    candidates_checked += len(solutions)

    for sol in solutions:
        score, _ = compute_soft_score(sol)
        if score > best_score:
            best_score = score
            best_assignment = sol

    print(f"  Evaluated {candidates_checked} candidate schedules.")
    return best_assignment, best_score


best_sched, best_score = find_best_schedule()
score_detail, detail_list = compute_soft_score(best_sched)

print(f"\n  Best Soft Constraint Score: {best_score} points")
print(f"\n  Score Breakdown:")
for d in detail_list:
    print(f"    {d}")

print()
print("  Optimal Schedule:")
print(f"  {'Route':<6} {'Bus Assigned':<14} {'Time':<8} {'Route Name':<30} {'Type'}")
print("  " + "-" * 80)
for r in routes:
    bus  = best_sched.get(r, 'UNASSIGNED')
    info = route_info[r]
    btype = ''
    if bus in large_coaches and r in highway_routes:
        btype = '[Large Coach on Highway ✓]'
    elif r == 'R3':
        btype = '[ALI\'S RELAY BUS ✓]'
    print(f"  {r:<6} {bus:<14} {info['time']:<8} {info['name']:<30} {btype}")

ali_bus = best_sched.get('R3', 'UNASSIGNED')
print(f"\n  >>> Ali's connecting bus is {ali_bus} at 12pm ✓")

print("""
  Analysis: Could Ali have missed his connecting bus?
  ====================================================
  Hard Constraint HC3 states that R3 (Sukkur->Multan) MUST always
  have a bus assigned. The backtracking algorithm enforces this by
  ensuring R3 is never left without a valid bus during the search.

  Also, Hard Constraint HC2 ensures R3 and R8 (both at 12pm) never
  share the same bus, so the bus assigned to R3 is exclusively
  available for Ali's route.

  Without HC3, the scheduler could theoretically skip R3 if all
  buses were already assigned to 2 routes each. HC3 prevents this
  by treating R3 as a priority constraint during the search.

  Result: Ali's connecting bus is ALWAYS guaranteed by the CSP!
""")