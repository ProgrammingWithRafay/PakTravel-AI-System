# Part 2: AI Legal Advisor (Propositional Logic)
# required libraries: AIMA Python library (logic.py must be accessible)
# Install via: pip install aima3  OR clone https://github.com/aimacode/aima-python

import sys

# fix encoding for windows terminals (cp1252 can't print unicode arrows etc.)
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 65)
print("    PakTravel AI System - Part 2: AI Legal Advisor")
print("=" * 65)

# try to import AIMA logic library - professor said to use PropKB
# support both direct import (cloned repo) and pip package
AIMA_AVAILABLE = False
try:
    from logic import PropKB, expr
    AIMA_AVAILABLE = True
    print("AIMA library loaded successfully (from logic module).\n")
except ImportError:
    try:
        from aima3.logic import PropKB, expr
        AIMA_AVAILABLE = True
        print("AIMA library loaded successfully (from aima3 package).\n")
    except ImportError:
        print("WARNING: AIMA library not found. Falling back to custom KB.\n")


# ============================================================
# Custom Forward-Chaining KB (fallback + used for step-by-step demo)
# ============================================================

class SimpleKB:
    """
    A simple propositional knowledge base using forward chaining.
    Rules are stored as (list_of_conditions, conclusion) pairs.
    This makes the inference steps very easy to trace and print.
    """

    def __init__(self):
        self.facts = set()
        self.rules = []     # each rule: ([conditions], conclusion, rule_name)
        self.rule_strings = {}  # for showing rules in IF...THEN format

    def tell_fact(self, fact):
        # add a known fact to the KB
        self.facts.add(fact)

    def tell_rule(self, conditions, conclusion, name=""):
        # conditions = list of strings, conclusion = string
        self.rules.append((conditions, conclusion))
        if name:
            cond_str = " AND ".join(conditions)
            self.rule_strings[name] = f"IF {cond_str} THEN {conclusion}"

    def forward_chain(self):
        # keep firing rules until nothing new can be derived (fixed point)
        # this is how forward chaining works - keep going until stable
        changed = True
        while changed:
            changed = False
            for conditions, conclusion in self.rules:
                if all(c in self.facts for c in conditions):
                    if conclusion not in self.facts:
                        self.facts.add(conclusion)
                        changed = True
        return self.facts

    def ask(self, query):
        # run forward chaining then check if query is a derived fact
        self.forward_chain()
        return query in self.facts

    def get_all_facts(self):
        self.forward_chain()
        return self.facts


# ============================================================
# TASK 1: Build the Knowledge Base
# ============================================================
print("--- TASK 1: Build the Knowledge Base ---\n")

# ---- Setup AIMA PropKB ----
if AIMA_AVAILABLE:
    kb = PropKB()

    # tell all 15 traffic rules to the KB using AIMA syntax
    # using ==> for implication (AIMA expr parser handles this)
    rules_aima = [
        'No_Helmet          ==> Fine_500',
        'No_Seatbelt        ==> Fine_1000',
        'Speed_Above_Limit  ==> Challan_Issued',
        'Challan_Issued & Not_Paid ==> License_Suspended',
        'License_Suspended & Still_Driving ==> Arrested',
        'Accident & No_Insurance ==> Court_Case',
        'Court_Case & Found_Guilty ==> License_Cancelled',
        'No_License         ==> Heavy_Fine_5000',
        'Mobile_While_Driving ==> Fine_2000',
        'Red_Light          ==> Fine_1500',
        'Fine_Paid_7Days    ==> Discount_50_Percent',
        'Three_Violations   ==> License_Suspended',
        'License_Suspended  ==> Cannot_Drive_Legally',
        'Drunk_Driving      ==> Arrested_Immediately',
        'Arrested & Repeat_Offender ==> Jail_Term',
    ]

    for rule in rules_aima:
        kb.tell(expr(rule))

    # tell Ahmed's known facts
    ahmed_facts_aima = [
        'Speed_Above_Limit',
        'Mobile_While_Driving',
        'Not_Paid',
        'Three_Violations',
        'Repeat_Offender',
    ]
    for fact in ahmed_facts_aima:
        kb.tell(expr(fact))

# ---- Setup Custom SimpleKB (always built, used for clear output) ----
ckb = SimpleKB()

# tell all 15 rules in human-readable format
traffic_rules = [
    (['No_Helmet'],                              'Fine_500',           'Rule1'),
    (['No_Seatbelt'],                            'Fine_1000',          'Rule2'),
    (['Speed_Above_Limit'],                      'Challan_Issued',     'Rule3'),
    (['Challan_Issued', 'Not_Paid'],             'License_Suspended',  'Rule4'),
    (['License_Suspended', 'Still_Driving'],     'Arrested',           'Rule5'),
    (['Accident', 'No_Insurance'],               'Court_Case',         'Rule6'),
    (['Court_Case', 'Found_Guilty'],             'License_Cancelled',  'Rule7'),
    (['No_License'],                             'Heavy_Fine_5000',    'Rule8'),
    (['Mobile_While_Driving'],                   'Fine_2000',          'Rule9'),
    (['Red_Light'],                              'Fine_1500',          'Rule10'),
    (['Fine_Paid_7Days'],                        'Discount_50_Percent','Rule11'),
    (['Three_Violations'],                       'License_Suspended',  'Rule12'),
    (['License_Suspended'],                      'Cannot_Drive_Legally','Rule13'),
    (['Drunk_Driving'],                          'Arrested_Immediately','Rule14'),
    (['Arrested', 'Repeat_Offender'],            'Jail_Term',          'Rule15'),
]

for conditions, conclusion, name in traffic_rules:
    ckb.tell_rule(conditions, conclusion, name)

# tell Ahmed's facts to custom KB
ahmed_facts = [
    'Speed_Above_Limit',
    'Mobile_While_Driving',
    'Not_Paid',
    'Three_Violations',
    'Repeat_Offender',
]
for fact in ahmed_facts:
    ckb.tell_fact(fact)


def show_laws():
    """Prints all 15 traffic laws in readable IF...THEN format."""
    print("=" * 55)
    print("  Pakistan NHA Traffic Laws - Knowledge Base")
    print("=" * 55)
    rule_display = [
        ("Rule 1",  "IF No_Helmet              THEN Fine_500"),
        ("Rule 2",  "IF No_Seatbelt            THEN Fine_1000"),
        ("Rule 3",  "IF Speed_Above_Limit      THEN Challan_Issued"),
        ("Rule 4",  "IF Challan_Issued AND Not_Paid THEN License_Suspended"),
        ("Rule 5",  "IF License_Suspended AND Still_Driving THEN Arrested"),
        ("Rule 6",  "IF Accident AND No_Insurance THEN Court_Case"),
        ("Rule 7",  "IF Court_Case AND Found_Guilty THEN License_Cancelled"),
        ("Rule 8",  "IF No_License             THEN Heavy_Fine_5000"),
        ("Rule 9",  "IF Mobile_While_Driving   THEN Fine_2000"),
        ("Rule 10", "IF Red_Light              THEN Fine_1500"),
        ("Rule 11", "IF Fine_Paid_7Days        THEN Discount_50_Percent"),
        ("Rule 12", "IF Three_Violations       THEN License_Suspended"),
        ("Rule 13", "IF License_Suspended      THEN Cannot_Drive_Legally"),
        ("Rule 14", "IF Drunk_Driving          THEN Arrested_Immediately"),
        ("Rule 15", "IF Arrested AND Repeat_Offender THEN Jail_Term"),
    ]
    for num, rule in rule_display:
        print(f"  {num:<8}: {rule}")
    print("=" * 55)


show_laws()

print("\nAhmed's Facts:")
for f in ahmed_facts:
    print(f"  [TRUE] {f}")

# run forward chaining
all_derived = ckb.get_all_facts()

print("\nQuery Results (using forward chaining):")
license_susp = ckb.ask('License_Suspended')
is_arrested   = ckb.ask('Arrested')
print(f"  Is Ahmed's license suspended? : {'YES' if license_susp else 'NO'}")
print(f"  Is Ahmed arrested?            : {'YES' if is_arrested else 'NO'}")

# Note: AIMA PropKB.ask() uses truth-table enumeration (2^n complexity)
# With 29 symbols that means ~536 million rows - too slow to run
# Rules and facts are already told to AIMA KB above (lines 110-124)
# Querying is done via our efficient forward-chaining SimpleKB instead

# fines Ahmed must pay
fines = []
if 'Challan_Issued' in all_derived:
    fines.append(('Challan (speeding)',    0))    # challan is a process, not direct fine amount
if 'Fine_2000' in all_derived:
    fines.append(('Mobile while driving', 2000))
if 'Fine_500' in all_derived:
    fines.append(('No helmet',            500))
if 'Fine_1000' in all_derived:
    fines.append(('No seatbelt',          1000))
if 'Fine_1500' in all_derived:
    fines.append(('Red light',            1500))
if 'Heavy_Fine_5000' in all_derived:
    fines.append(('No license',           5000))

print(f"\nFines Summary for Ahmed:")
total_fines = 0
for desc, amount in fines:
    if amount > 0:
        print(f"  {desc:<30}: PKR {amount}")
        total_fines += amount
print(f"  {'TOTAL FINES':<30}: PKR {total_fines}")

print(f"\nOther Consequences:")
consequences = [
    ('License_Suspended',     'License is SUSPENDED'),
    ('Arrested',              'Ahmed is ARRESTED'),
    ('Cannot_Drive_Legally',  'Cannot drive legally'),
    ('Jail_Term',             'JAIL TERM issued (Arrested + Repeat Offender)'),
]
for key, msg in consequences:
    if key in all_derived:
        print(f"  [!] {msg}")


# ============================================================
# TASK 2: Inference Rules
# ============================================================
print("\n" + "=" * 65)
print("--- TASK 2: Inference Rules ---\n")

# ---- Modus Ponens ----
# Logical Notation:
#   P       : Speed_Above_Limit = TRUE  (Ahmed's fact)
#   P -> Q  : Speed_Above_Limit ==> Challan_Issued  (Rule 3)
#   Therefore Q: Challan_Issued = TRUE  (by Modus Ponens)
print("[ Modus Ponens ]")
print()
print("  Logical Notation:")
print("    Premise 1 (P)     : Speed_Above_Limit  [Ahmed's fact = TRUE]")
print("    Premise 2 (P→Q)   : Speed_Above_Limit ==> Challan_Issued  [Rule 3]")
print("    Conclusion (Q)    : Therefore, Challan_Issued must be TRUE")
print()
print("  Python Implementation:")

# modus ponens check in python
P   = True   # Ahmed was speeding - given fact
P_implies_Q = True  # Rule 3 exists in KB

# if both premises hold, conclusion follows
Q = P and P_implies_Q  # this is how modus ponens works

print(f"    Speed_Above_Limit         = {P}")
print(f"    Rule 3 (Speed=>Challan)   = {P_implies_Q}")
print(f"    Challan_Issued (derived)  = {Q}")
print(f"\n    Is challan issued? : {'YES' if Q else 'NO'}")

# ---- Hypothetical Syllogism ----
print()
print("=" * 45)
print("[ Hypothetical Syllogism ]")
print()
print("  Chaining Rules 3 -> 4 -> 5 step by step:")
print()
print("  Rule 3: Speed_Above_Limit ==> Challan_Issued")
print("  Rule 4: Challan_Issued AND Not_Paid ==> License_Suspended")
print("  Rule 5: License_Suspended AND Still_Driving ==> Arrested")
print()
print("  Deriving consequences for Ahmed:")
print()

# step 1
speed_above_limit = True  # given fact
challan_issued = False
step1 = speed_above_limit  # rule 3 fires
if step1:
    challan_issued = True
    print("  Step 1: Speed_Above_Limit is TRUE")
    print("          Rule 3 fires: Speed_Above_Limit ==> Challan_Issued")
    print("          >>> Challan Issued!")

# step 2
not_paid = True  # given fact
license_suspended = False
step2 = challan_issued and not_paid
if step2:
    license_suspended = True
    print()
    print("  Step 2: Challan_Issued=TRUE and Not_Paid=TRUE")
    print("          Rule 4 fires: Challan_Issued AND Not_Paid ==> License_Suspended")
    print("          >>> License Suspended!")

# step 3
still_driving = True  # Ahmed was still driving
arrested = False
step3 = license_suspended and still_driving
if step3:
    arrested = True
    print()
    print("  Step 3: License_Suspended=TRUE and Still_Driving=TRUE")
    print("          Rule 5 fires: License_Suspended AND Still_Driving ==> Arrested")
    print("          >>> Ahmed is ARRESTED!")

print()
print("  Full chain: Speeding -> Challan -> License Suspended -> ARRESTED")
print("  Combined rule: Speed_Above_Limit AND Not_Paid AND Still_Driving ==> Arrested")

# ---- Modus Tollens ----
# Logical Notation:
#   P -> Q  : (License_Suspended & Still_Driving) ==> Arrested  (Rule 5)
#   ~Q      : ~Arrested  (Bilal was NOT arrested)
#   Therefore ~P: ~(License_Suspended & Still_Driving)  (by Modus Tollens)
#   Conclusion: Bilal either was NOT suspended OR he stopped driving
print()
print("=" * 45)
print("[ Modus Tollens ]")
print()
print("  Scenario: Driver Bilal was NOT arrested. He was also speeding.")
print()
print("  Logical Notation:")
print("    Premise 1 (P→Q)  : License_Suspended AND Still_Driving ==> Arrested")
print("    Premise 2 (~Q)   : ~Arrested  [Bilal was NOT arrested]")
print("    Conclusion (~P)  : ~(License_Suspended AND Still_Driving)")
print("                       i.e., NOT (license_suspended AND still_driving)")
print("                       Bilal's license was NOT suspended, OR he stopped driving.")
print()

# modus tollens - if conclusion is false, at least one premise must be false
arrested_bilal = False  # given: Bilal was not arrested
print("  Python Implementation:")
print(f"    Arrested (Bilal)       = {arrested_bilal}")
print(f"    By Modus Tollens:")
print(f"    ~Arrested => NOT(License_Suspended AND Still_Driving)")
print()

# what saved Bilal?
# either his license was not suspended OR he stopped driving after suspension
# since he was speeding, Rule3 still fires. But maybe he PAID his challan?
# or he was not caught 3 times - so Rule12 didn't fire?
# the Modus Tollens tells us at least one of those conditions was false

print("  What saved Bilal from arrest?")
print("    From Modus Tollens: since Bilal was NOT arrested,")
print("    it means NOT(License_Suspended AND Still_Driving) is TRUE.")
print("    Either: Bilal's challan was PAID (so license not suspended),")
print("    OR    : Bilal STOPPED driving after the warning.")
print("    Most likely: Bilal paid his challan on time (Not_Paid = FALSE)")
print("    so Rule 4 never fired, license was never suspended.")
print()
print("    What saved Bilal: He PAID his challans (Not_Paid = FALSE)")


# ============================================================
# TASK 3: Resolution Refutation
# ============================================================
print("\n" + "=" * 65)
print("--- TASK 3: Resolution Refutation ---\n")
print("Goal: PROVE that Ahmed's license must be suspended.\n")
print("Method: Add negation of goal (~License_Suspended) and derive contradiction.\n")

print("Clauses in CNF form:")
print("  C1 : Speed_Above_Limit                      [Fact - Ahmed speeding]")
print("  C2 : Not_Paid                               [Fact - challans unpaid]")
print("  C3 : Three_Violations                       [Fact - 3 violations this month]")
print("  C4 : ~Speed_Above_Limit | Challan_Issued    [Rule 3 in CNF]")
print("  C5 : ~Challan_Issued | ~Not_Paid | License_Suspended  [Rule 4 in CNF]")
print("  C6 : ~Three_Violations | License_Suspended  [Rule 12 in CNF]")
print("  C7 : ~License_Suspended                     [NEGATION of our goal]")
print()
print("Resolution Steps:")
print("-" * 60)
print(f"{'Step':<6} {'Resolving':<30} {'Resolvent':<30}")
print("-" * 60)

# Step 1: Resolve C1 (Speed_Above_Limit) with C4 (~Speed_Above_Limit | Challan_Issued)
# Cancel Speed_Above_Limit -> get: Challan_Issued
print(f"{'Step 1':<6} {'C1 + C4':<30} {'Challan_Issued  [C8]':<30}")
print(f"       C1={'{Speed_Above_Limit}'}")
print(f"       C4={'{~Speed_Above_Limit, Challan_Issued}'}")
print(f"       Cancel Speed_Above_Limit -> C8 = {'{Challan_Issued}'}")
print()

# Step 2: Resolve C8 (Challan_Issued) with C5 (~Challan_Issued | ~Not_Paid | License_Suspended)
# Cancel Challan_Issued -> get: ~Not_Paid | License_Suspended
print(f"{'Step 2':<6} {'C8 + C5':<30} {'~Not_Paid|Lic_Sus [C9]':<30}")
print(f"       C8={'{Challan_Issued}'}")
print(f"       C5={'{~Challan_Issued, ~Not_Paid, License_Suspended}'}")
print(f"       Cancel Challan_Issued -> C9 = {'{~Not_Paid, License_Suspended}'}")
print()

# Step 3: Resolve C9 (~Not_Paid | License_Suspended) with C2 (Not_Paid)
# Cancel Not_Paid -> get: License_Suspended
print(f"{'Step 3':<6} {'C9 + C2':<30} {'License_Suspended [C10]':<30}")
print(f"       C9={'{~Not_Paid, License_Suspended}'}")
print(f"       C2={'{Not_Paid}'}")
print(f"       Cancel Not_Paid -> C10 = {'{License_Suspended}'}")
print()

# Step 4: Resolve C10 (License_Suspended) with C7 (~License_Suspended)
# Cancel License_Suspended -> get: EMPTY CLAUSE = contradiction!
print(f"{'Step 4':<6} {'C10 + C7':<30} {'EMPTY CLAUSE !!!':<30}")
print(f"       C10={'{License_Suspended}'}")
print(f"       C7 ={'{~License_Suspended}'}")
print(f"       Cancel License_Suspended -> EMPTY CLAUSE")
print("-" * 60)
print()
print("  EMPTY CLAUSE derived = CONTRADICTION!")
print("  Adding ~License_Suspended leads to contradiction.")
print("  Therefore, License_Suspended MUST be TRUE.")
print()
print("  Also verifiable via Rule 12 shortcut:")
print(f"  C3={'{Three_Violations}'} + C6={'{~Three_Violations, License_Suspended}'}")
print(f"  -> C11 = {'{License_Suspended}'}  then C11 + C7 -> EMPTY CLAUSE")
print()
print("=" * 65)
print("  *** PROVEN: License Suspended ***")
print("=" * 65)

# final summary of what happened to Ahmed
print("\n--- Final Summary for Ahmed ---\n")
print("Facts given:")
for f in ahmed_facts:
    print(f"  [+] {f}")
print()
print("Derived consequences:")
derived_only = all_derived - set(ahmed_facts)
for d in sorted(derived_only):
    print(f"  [=>] {d}")