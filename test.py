from sympy import *
from sys import settrace

class LineCounter:
    def __init__(self):
        self.count = 0

    def trace(self, frame, event, arg):
        if event == 'line':
            self.count += 1
        return self.trace

# First create all needed symbols
variables = symbols('x1:11')  # Creates x1, x2, ..., x10
x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = variables

# Setup counter
counter = LineCounter()
settrace(counter.trace)

# Your Gröbner basis computation
polys = [
    -2*x1**7*x2**6*x4**6*x5**5 + 5*x1**6*x2**4*x3**2*x4**6*x5**6*x6**8 + 10*x1**4*x2**6*x3**5*x4*x5**3*x6 + 9*x1**4*x2**5*x3**6*x4**4*x5**8*x6**4 + 9*x1**2*x2**8*x4**3*x5**2*x6**8 + 11*x2**6*x4**8*x5**6*x6**4,
    -4*x1**2*x2**2*x3**2*x4*x5**2 - 2*x1**2*x2**2*x3*x4**2*x6**2 - 6*x1**2*x2**2*x4**2*x5**2*x6 - 2*x1*x2*x6 + 8*x2*x3*x5*x6**2 + 4*x4*x5*x6**2,
    -8*x1**3*x2**3*x3**2*x4*x5*x6**3 - 6*x1**3*x3**2*x4*x6**2 + x1**2*x2**3*x3**3*x4**2*x5**3*x6 - 3*x1*x2**3*x3**3*x4**2*x5**2*x6**2 - 10*x3**2*x4**2*x5**2 + 11*x3*x4*x6,
    -5*x1**10*x2**9*x3*x4**3*x5**2*x6**8 - 8*x1**10*x2**6*x4**6*x5**2*x6**6 + 4*x1**8*x2**6*x3**3*x4**7*x5*x6 + 6*x1**6*x2**6*x3*x4**7*x5**5*x6 - 10*x1**2*x2**9*x3**7*x4**8*x5**10*x6**3 - x2**3*x3**3*x4**6*x5**5*x6**4
]

gb = groebner(polys, variables, order='lex')

# Reset and print results
settrace(None)
print(f"Total lines executed: {counter.count}")
print(f"Gröbner basis: {gb}")