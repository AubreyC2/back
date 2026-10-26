import random as rand
import sympy
from sympy import groebner, symbols, Mul
    
def gen_ran_poly(variables, degree, terms, coeff_range):
    poly = 0
    for _ in range(terms):
        coeff = rand.randint(*coeff_range)
        if coeff == 0:
            continue
        exponents = [rand.randint(0, degree) for _ in variables]
        term = coeff * Mul(*[v**e for v, e in zip(variables, exponents)])
        poly += term
    return poly

# Step 1: Random number of variables
num_vars = rand.randint(1, 5)
var_symbols = symbols(f'x1:{num_vars+1}')  # sympy style, gives x1, x2, ..., xn

# Step 2: Generate 10 random polynomials
polys = [gen_ran_poly(var_symbols, degree=2, terms=3, coeff_range=(-10, 10)) for _ in range(10)]

# Step 3: Compute Groebner basis
for p in polys:
    G = groebner([p], *var_symbols, domain='QQ', order='lex')
    print(f'Poly: {p}')
    print(f'{G}\n')