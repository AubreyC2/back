import random as rand
import sympy
from sympy import groebner

def gen_ran_poly(variables, multidegree, degree, terms, coeff_range):
    poly = 0
    for _ in range(terms):
        coeff = rand.randint(*coeff_range)
        if coeff == 0:
            continue
        exponents = [rand.randint(0, degree) for _ in variables]
        term = coeff * sympy.Mul(*[v**e for v, e in zip(variables, exponents)])
        poly += term
    return poly

num_vars = rand.randint(1, 10)
var_symbols = sympy.symbols(f'x1:{num_vars + 1}')

polys = [gen_ran_poly(var_symbols, 30, degree=10, terms=3, coeff_range=(-10, 10)) for _ in range(1)]

for p in (polys):
    G = sympy.groebner([p], *var_symbols, domain='QQ', order='lex')
    print(f'Poly :{p} ')
    print(f'{G} \n')


