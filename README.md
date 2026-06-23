# PakTravel AI System

> An AI-powered intercity bus travel system for Pakistan, built as a semester project covering classical search, propositional logic, CSP scheduling, neural networks, and unsupervised learning.

---

## Project Overview

PakTravel AI System is a multi-part Python project that applies core Artificial Intelligence techniques to solve real-world problems in Pakistan's intercity bus travel domain — from finding the shortest route between cities to predicting delays and clustering passengers.

---

## Project Structure

```
Project/
├── Task_1/   route_finder.py              # Search Algorithms (UCS, A*, Bidirectional)
├── Task_2/   aima-python-master/
│             ├── legal_advisor.py         # AI Legal Advisor (Propositional Logic)
│             └── logic.py                 # AIMA logic engine (PropKB, expr, resolution)
├── Task_3/   bus_schedular.py             # CSP Bus Scheduler
├── Task_4/   delay_predictor.py           # ANN Delay Predictor
└── Task_5/   traveller_clustering.py      # K-Means Traveller Segmentation
```

> **Note:** `legal_advisor.py` lives inside `aima-python-master/` because it imports `logic.py` directly from that folder at runtime.

---

## Tasks Breakdown

### Task 1 — Route Finder (Search Algorithms)

Builds a weighted road network of **15 Pakistani cities** and implements three search algorithms to find optimal intercity routes.

| Algorithm | Description |
|---|---|
| **UCS** (Uniform Cost Search) | Guarantees the shortest path by expanding nodes in order of cost |
| **A\*** | Uses straight-line distance as an admissible heuristic to explore fewer nodes |
| **Bidirectional UCS** | Runs two simultaneous searches from both ends to meet in the middle |

**Key cities in the network:** Karachi, Lahore, Islamabad, Multan, Peshawar, Quetta, Faisalabad, and more.

A NetworkX visualization is generated showing the road network with the optimal route highlighted.

**Verdict:** A* is recommended for PakTravel — optimal, fast, and scalable as the network grows.

---

### Task 2 — AI Legal Advisor (Propositional Logic)

Models Pakistan's **NHA traffic laws** as a propositional knowledge base and uses logical inference to determine the legal consequences for a driver named Ahmed.

**Knowledge Base:** 15 traffic rules encoded in `IF ... THEN` form using AIMA's `PropKB` and `expr`, with a custom `SimpleKB` forward-chaining engine for clear step-by-step output.

**Ahmed's facts fed into the KB:**

| Fact | Meaning |
|---|---|
| `Speed_Above_Limit` | Ahmed was caught speeding |
| `Mobile_While_Driving` | Ahmed used his phone while driving |
| `Not_Paid` | His challan remains unpaid |
| `Three_Violations` | He has 3 violations this month |
| `Repeat_Offender` | He has a prior record |

**Derived consequences (forward chaining):**

| Derived Fact | Triggered By |
|---|---|
| `Challan_Issued` | Rule 3 — speeding |
| `Fine_2000` | Rule 9 — mobile while driving |
| `License_Suspended` | Rule 4 (challan + unpaid) & Rule 12 (3 violations) |
| `Cannot_Drive_Legally` | Rule 13 — suspended license |
| `Arrested` | Rule 5 — suspended + still driving |
| `Jail_Term` | Rule 15 — arrested + repeat offender |

**Inference rules demonstrated:**

- **Modus Ponens** — `Speed_Above_Limit` (P) + Rule 3 `P→Q` → `Challan_Issued` (Q)
- **Hypothetical Syllogism** — chains Rules 3→4→5 to trace Ahmed's path from speeding all the way to arrest
- **Modus Tollens** — applied to driver Bilal (not arrested) to reason backwards: since `~Arrested`, at least one of `{License_Suspended, Still_Driving}` must be false — concluding Bilal paid his challans

**Resolution Refutation (Task 3 inside legal_advisor):**  
Proves `License_Suspended` is necessarily true by negating it (`~License_Suspended`), converting all rules to CNF clauses, and resolving until the empty clause (contradiction) is derived in 4 steps.

**Dependency:** Must be run from inside `Task_2/aima-python-master/` so that `from logic import PropKB, expr` resolves correctly. Alternatively, install `pip install aima3` and run from anywhere.

---

### Task 3 — Bus Scheduler (CSP)

Models bus-to-route assignment as a **Constraint Satisfaction Problem** with 8 routes and 10 buses.

**Variables:** R1–R8 (bus routes)  
**Domain:** Bus1–Bus10 (available buses)

**Hard Constraints:**
- `HC1 / HC2` — Time conflict: no bus can run two routes at the same time slot (8am, 12pm)
- `HC3` — Ali's connecting bus (R3: Sukkur → Multan) must always be assigned
- `HC4` — Driver rest rule: no bus can operate more than 2 routes per day

**Soft Constraints:**
- `SC1` — Large coaches (Bus1–Bus3) preferred on highway routes
- `SC2` — Reward buses that stay within the 2-route limit

Solved using **Backtracking Search with AC-3 arc consistency** propagation.

---

### Task 4 — Delay Predictor (ANN)

Trains a **Multi-Layer Perceptron (MLP)** neural network to predict whether a bus journey will be delayed or on time.

**Features used:**

| Feature | Description |
|---|---|
| `Distance` | Journey distance in km (80–700 km) |
| `Weather_Score` | 1–5 scale (5 = severe weather) |
| `Traffic_Score` | 1–5 scale (5 = heavy traffic) |
| `Road_Quality` | 1–5 scale (1 = poor road) |
| `Bus_Age` | Age of the bus in years |

**Label Rule:**
- Delayed if `Weather ≥ 4` OR `Traffic ≥ 4` OR (`Distance > 400` AND `Road Quality ≤ 2`)

**Pipeline:** Data generation → MinMax scaling → 80/20 train-test split → MLP training → Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)

---

### Task 5 — Traveller Clustering (K-Means)

Segments **400 synthetic traveller records** into meaningful groups using **K-Means clustering** (k=3), validated with the Elbow Method and Silhouette Score.

**Features:** Trips per month, avg booking lead days, avg journey distance, preferred travel time, loyalty points, complaints filed.

**Discovered Clusters:**

| Cluster | Traveller Type | Characteristics |
|---|---|---|
| **A** | Business Travellers | High frequency, long distance, books in advance, high loyalty |
| **B** | Family Travellers | Low frequency, medium distance, last-minute booker |
| **C** | Short-distance Commuters | Very high frequency, short routes, price-sensitive |

Cluster profiles are visualized with scatter plots and bar charts.

---

## Installation & Usage

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install networkx matplotlib numpy pandas scikit-learn
```

### Run Each Task

```bash
# Task 1 – Route Finder
python Task_1/route_finder.py

# Task 2 – AI Legal Advisor (must run from inside the aima folder)
cd Task_2/aima-python-master
python legal_advisor.py

# Task 3 – Bus Scheduler
python Task_3/bus_schedular.py

# Task 4 – Delay Predictor
python Task_4/delay_predictor.py

# Task 5 – Traveller Clustering
python Task_5/traveller_clustering.py
```

---

## AI Techniques Used

| Technique | Task | Library |
|---|---|---|
| Uniform Cost Search | Route Finder | Pure Python |
| A\* Search | Route Finder | Pure Python |
| Bidirectional Search | Route Finder | Pure Python |
| Propositional Logic (Forward Chaining, Resolution) | AI Legal Advisor | AIMA `logic.py` / Custom KB |
| Constraint Satisfaction (Backtracking + AC-3) | Bus Scheduler | Pure Python |
| Artificial Neural Network (MLP) | Delay Predictor | scikit-learn |
| K-Means Clustering | Traveller Segmentation | scikit-learn |

---

## Dependencies

```
networkx
matplotlib
numpy
pandas
scikit-learn
```

---

## License

This project is for educational purposes as part of an AI semester project.

---

## 🙏 Acknowledgements

- [AIMA Python](https://github.com/aimacode/aima-python) — `logic.py` (PropKB, expr, resolution) used as the inference engine for Task 2; the full library ships in `Task_2/aima-python-master/` for reference
