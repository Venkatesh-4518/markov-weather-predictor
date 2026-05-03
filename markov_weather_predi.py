# ============================================================
#   Markov Chain Weather Predictor  --  Interactive CLI
#   Linear Algebra Group Project
#   States: Sunny, Rainy, Cloudy
# ============================================================
#


import numpy as np

# -------------------------------------------------------
# SECTION 1 -- TRANSITION MATRIX  
# -------------------------------------------------------
STATES = ["Sunny", "Rainy", "Cloudy"]

#            --> Sunny   Rainy  Cloudy
P = np.array([
    [0.60, 0.20, 0.20],   # from Sunny
    [0.30, 0.40, 0.30],   # from Rainy
    [0.30, 0.30, 0.40],   # from Cloudy
])

def validate_transition_matrix(matrix):
    assert np.all(matrix >= 0),                  "Probabilities must be >= 0"
    assert np.allclose(matrix.sum(axis=1), 1.0), "Each row must sum to 1"

validate_transition_matrix(P)

# -------------------------------------------------------
# SECTION 2 -- STATE ENCODING  
# -------------------------------------------------------
def encode_state(name):
    vec = np.zeros(len(STATES))
    vec[STATES.index(name)] = 1.0
    return vec

def decode_state(vec):
    return STATES[int(np.argmax(vec))]

# -------------------------------------------------------
# SECTION 3 -- PREDICTION  
# -------------------------------------------------------
def predict_n_days(state_name, n):
    """
    Predict weather probabilities n days ahead.
    Formula: state_vector @ P^n
    """
    v  = encode_state(state_name)
    Pn = np.linalg.matrix_power(P, n)
    return v @ Pn

def show_probabilities(label, probs):
    print()
    print(label)
    print("-" * 40)
    for name, p in zip(STATES, probs):
        bar = "#" * int(p * 30)
        print("  {:<8}  {:5.1f}%  {}".format(name, p * 100, bar))
    print()
    print("  Most likely: {}".format(decode_state(probs)))

# -------------------------------------------------------
# SECTION 4 -- STEADY STATE  
# -------------------------------------------------------
def steady_state_eigenvalue():
    """
    Find pi using eigenvalue decomposition of P.T
    Solves: pi @ P = pi  and  sum(pi) = 1
    The steady state is the left eigenvector for eigenvalue = 1.
    """
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi  = eigenvectors[:, idx].real
    return pi / pi.sum()

STEADY = steady_state_eigenvalue()

# -------------------------------------------------------
# SECTION 5 -- SIMULATION  
# -------------------------------------------------------
def simulate_weather(start, days, seed=42):
    """
    Monte Carlo simulation.
    Each day: randomly pick next state based on transition probabilities.
    """
    rng = np.random.default_rng(seed)
    seq, cur = [start], start
    for _ in range(days - 1):
        row = P[STATES.index(cur)]
        cur = rng.choice(STATES, p=row)
        seq.append(cur)
    return seq

# -------------------------------------------------------
# SECTION 6 -- DISPLAY HELPERS  
# -------------------------------------------------------
SEP = "=" * 50

def print_banner():
    print(SEP)
    print("  Markov Chain Weather Predictor")
    print("  Linear Algebra Group Project")
    print(SEP)

def print_matrix():
    print()
    print("Transition Matrix P:")
    print("(Row = current state, Column = next state)")
    print()
    header = "          " + "  ".join("{:>8}".format(s) for s in STATES)
    print(header)
    print("          " + "-" * 32)
    for i, rs in enumerate(STATES):
        row = "  ".join("{:>8.2f}".format(P[i, j]) for j in range(3))
        print("{:<8}| {}".format(rs, row))
    print()

def print_steady():
    print()
    print("Long-run Steady State Probabilities (pi):")
    print("(System converges here regardless of starting state)")
    print("-" * 40)
    for name, p in zip(STATES, STEADY):
        bar = "#" * int(p * 30)
        print("  {:<8}  {:5.1f}%  {}".format(name, p * 100, bar))
    print()
    ok = np.allclose(STEADY @ P, STEADY)
    print("  Verification pi @ P = pi : {}".format("PASSED" if ok else "FAILED"))

def show_sequence(seq):
    print()
    for i, s in enumerate(seq):
        print("  Day {:>3} :  {}".format(i + 1, s))
    print()
    print("  Frequency Summary:")
    print("  " + "-" * 36)

    total = len(seq)
    for s in STATES:
        count = seq.count(s)
        pct   = count / total * 100
        bar   = "#" * int(pct / 3)
        print("  {:<8}  {:>2} days  ({:5.1f}%)  {}".format(s, count, pct, bar))

def show_progression(start, n):
    limit = min(n, 10)
    print()
    print("  Day-by-day probabilities:")
    print("  " + "-" * 50)
    print("  {:<6}  {:<10}  {}".format("Day", "Top State", "[Sunny%  Rainy%  Cloudy%]"))
    print("  " + "-" * 50)
    for d in range(1, limit + 1):
        p   = predict_n_days(start, d)
        top = decode_state(p)
        print("  Day {:>3}  {:<10}  [S:{:4.1f}%  R:{:4.1f}%  C:{:4.1f}%]".format(
            d, top, p[0]*100, p[1]*100, p[2]*100))
    if n > 10:
        print("  ...")
        p   = predict_n_days(start, n)
        top = decode_state(p)
        print("  Day {:>3}  {:<10}  [S:{:4.1f}%  R:{:4.1f}%  C:{:4.1f}%]".format(
            n, top, p[0]*100, p[1]*100, p[2]*100))

# -------------------------------------------------------
# INPUT HELPERS
# -------------------------------------------------------
def get_state_input(prompt="Choose weather state:"):
    print()
    print(prompt)
    print("  1 --> Sunny")
    print("  2 --> Rainy")
    print("  3 --> Cloudy")
    mapping = {"1": "Sunny", "2": "Rainy", "3": "Cloudy",
               "s": "Sunny", "r": "Rainy",  "c": "Cloudy",
               "sunny": "Sunny", "rainy": "Rainy", "cloudy": "Cloudy"}
    while True:
        raw = input("  Your choice: ").strip().lower()
        if raw in mapping:
            return mapping[raw]
        print("  Invalid. Enter 1, 2, 3 (or s / r / c).")

def get_int(prompt, lo, hi):
    while True:
        try:
            val = int(input("  {} ({}-{}): ".format(prompt, lo, hi)).strip())
            if lo <= val <= hi:
                return val
            print("  Enter a number between {} and {}.".format(lo, hi))
        except ValueError:
            print("  Please enter a valid integer.")

# -------------------------------------------------------
# MAIN MENU
# -------------------------------------------------------
def menu():
    print()
    print("-" * 50)
    print("  MENU")
    print("    1.  Predict weather N days ahead")
    print("    2.  Simulate a day-by-day sequence")
    print("    3.  Show transition matrix")
    print("    4.  Show steady-state distribution")
    print("    5.  Exit")
    print("-" * 50)
    while True:
        c = input("  Choice: ").strip()
        if c in ("1", "2", "3", "4", "5"):
            return c
        print("  Enter 1 to 5.")

def main():
    print_banner()
    print_matrix()

    while True:
        choice = menu()

        # --- Option 1: Predict ---
        if choice == "1":
            state = get_state_input("What is TODAY's weather?")
            n     = get_int("How many days ahead to predict?", 1, 365)
            probs = predict_n_days(state, n)
            show_probabilities(
                "Forecast for Day +{} (starting from {}):".format(n, state),
                probs
            )
            ans = input("\n  Show day-by-day progression? (y/n): ").strip().lower()
            if ans == "y":
                show_progression(state, n)

        # --- Option 2: Simulate ---
        elif choice == "2":
            state = get_state_input("What is the starting weather?")
            days  = get_int("How many days to simulate?", 1, 60)
            seed  = get_int("Random seed (for reproducibility)?", 0, 9999)
            seq   = simulate_weather(state, days, seed=seed)
            print("\n  Simulated {}-day sequence (start={}, seed={}):".format(
                days, state, seed))
            show_sequence(seq)
            ans = input("  Compare with steady state? (y/n): ").strip().lower()
            if ans == "y":
                print_steady()

        # --- Option 3: Matrix ---
        elif choice == "3":
            print_matrix()

        # --- Option 4: Steady State ---
        elif choice == "4":
            print_steady()

        # --- Option 5: Exit ---
        elif choice == "5":
            print("\n  Goodbye!\n")
            break

if __name__ == "__main__":
    main()
