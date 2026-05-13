import sys
sys.path.insert(0, './aima-python')
from utils import *
from logic import *

kb = PropKB()

# ---------- Ali: CGPA 3.8 (good), no disciplinary case ----------
kb.tell(expr('Good_CGPA_Ali'))
kb.tell(expr('No_Case_Ali'))
kb.tell(expr('Good_CGPA_Ali & No_Case_Ali ==> Deans_List_Ali'))

# ---------- Sara: CGPA 3.6 (good), HAS a case — do NOT tell No_Case_Sara ----------
kb.tell(expr('Good_CGPA_Sara'))
kb.tell(expr('Good_CGPA_Sara & No_Case_Sara ==> Deans_List_Sara'))

# ---------- Ahmed: CGPA 2.5 (bad), no case — do NOT tell Good_CGPA_Ahmed ----------
kb.tell(expr('No_Case_Ahmed'))
kb.tell(expr('Good_CGPA_Ahmed & No_Case_Ahmed ==> Deans_List_Ahmed'))

# ---------- Zara: CGPA 3.9 (good), no case ----------
kb.tell(expr('Good_CGPA_Zara'))
kb.tell(expr('No_Case_Zara'))
kb.tell(expr('Good_CGPA_Zara & No_Case_Zara ==> Deans_List_Zara'))

print("   FAST University \u2014 Dean's List Check")
print("   Rule: \u2200x Good_CGPA(x) \u2227 No_Case(x) \u2192 DL(x)")

students = ['Ali', 'Sara', 'Ahmed', 'Zara']

for name in students:
    result = kb.ask(expr('Deans_List_' + name)) is not False
    if result:
        print(f"{name:<6}: YES \u2014 On Dean's List \u2713")
    else:
        print(f"{name:<6}: NO  \u2014 Not on Dean's List \u2717")

