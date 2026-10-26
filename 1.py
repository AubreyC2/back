import random

# ======================================================
# ================= USER PARAMETERS ====================
# ======================================================

NUM_VARIABLES = 5       # number of variables (n)
DATASET_SIZE = 50      # how many ideals to generate
INVERTIBLE_STEPS = 22   # how many gi <- gi + gj*p(x) steps
COEFF_RANGE = (-5, 5)   # coefficients (excluding 0)
MAX_DEG = 4             # degree for the univariate polynomial
P_DEG = 5               # max degree for random multiplier p(x)
OUTPUT_FILE = "data_backg_1.txt"


# ======================================================
# =============== POLYNOMIAL UTILITIES =================
# ======================================================

def rand_coeff():
    c = 0
    while c == 0:
        c = random.randint(*COEFF_RANGE)
    return c


def random_monomial(n, max_deg):
    exps = [0] * n
    var = random.randrange(n)
    exps[var] = random.randint(0, max_deg)
    return exps


def poly_add(p, q):
    r = p.copy()
    for m, c in q.items():
        r[m] = r.get(m, 0) + c
        if r[m] == 0:
            del r[m]
    return r


def poly_mul(p, q):
    r = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            m = tuple(a + b for a, b in zip(m1, m2))
            r[m] = r.get(m, 0) + c1 * c2
    return r


def random_multiplier(n):
    p = {}
    terms = random.randint(1, 3)
    for _ in range(terms):
        exp = tuple(random_monomial(n, P_DEG))
        p[tuple(exp)] = rand_coeff()
    return p


# ======================================================
# ============== SHAPE POSITION GENERATOR ==============
# ======================================================

def shape_position_ideal(n):
    """
    Triangular/shape form:
        f1(x1)
        x2 - g2(x1)
        ...
        xn - gn(x1)
    """
    polys = []

    # f1 univariate in x1
    f1 = {}
    for d in range(MAX_DEG + 1):
        if random.random() < 0.7:
            exp = [0] * n
            exp[0] = d
            f1[tuple(exp)] = rand_coeff()
    polys.append(f1)

    # xi - gi(x1)
    for i in range(1, n):
        poly = {}

        # xi
        exp = [0] * n
        exp[i] = 1
        poly[tuple(exp)] = 1

        # subtract random polynomial in x1
        for d in range(random.randint(0, MAX_DEG)):
            exp = [0] * n
            exp[0] = d
            poly[tuple(exp)] = poly.get(tuple(exp), 0) + rand_coeff()

        polys.append(poly)

    return polys


# ======================================================
# ============== INVERTIBLE OPERATIONS =================
# ======================================================

def mess_up(polys, steps, n):
    polys = [p.copy() for p in polys]

    for _ in range(steps):
        i, j = random.sample(range(len(polys)), 2)
        p = random_multiplier(n)
        polys[i] = poly_add(polys[i], poly_mul(polys[j], p))

    return polys


# ======================================================
# ================== STRING FORMATTER ==================
# ======================================================

def format_poly(poly, n):
    terms = []
    for m, c in sorted(poly.items()):
        exps = ",".join(str(e) for e in m)
        terms.append(f"({c},({exps}))")
    return ",".join(terms)


def format_ideal(polys, n):
    return ";".join(format_poly(p, n) for p in polys)


# ======================================================
# ======================= MAIN =========================
# ======================================================

with open(OUTPUT_FILE, "w") as f:
    for _ in range(DATASET_SIZE):
        original = shape_position_ideal(NUM_VARIABLES)
        messed = mess_up(original, INVERTIBLE_STEPS, NUM_VARIABLES)

        left = format_ideal(original, NUM_VARIABLES)
        right = format_ideal(messed, NUM_VARIABLES)

        f.write(left + " | " + right + "\n")
