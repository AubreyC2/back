import random as rand
import sympy
def gen_ran_poly(var, degree, terms, coeff):
    poly = 0
    for _ in range(terms):
        coeff_range= rand.randint(*coeff)
        if coeff==0:
            continue
        exponent = [rand.randint(0, degree) for _ in var]
        term= coeff_range * sympy.Mul(*[v**e for v,e in zip(var, exponent)])
        poly+= term
    return poly
num_var = rand.randint(3,10)
variables= sympy.symbols(f'x_1:{num_var +1}')
poly= [gen_ran_poly(variables, 5, 5 , (-10,10)) for _ in range(10)]
G = sympy.groebner(poly, *variables, domain='QQ', order='lex')
for p in (poly):
    G = sympy.groebner(poly, *variables, domain='QQ', order='lex')
    print(f'poly:{p}')






