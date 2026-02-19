import math
import random
from fractions import Fraction

def double_factorial_odd(k: int) -> int:
    prod = 1
    for x in range(1, k + 1, 2):
        prod *= x
    return prod

def exact_probability(n: int) -> Fraction:
    return Fraction(math.factorial(n), double_factorial_odd(2*n - 1))

def estimate_probability(n=5, trials=200_000):
    hits = 0
    for _ in range(trials):
        lefts = []
        rights = []
        for _ in range(n):
            a = random.random()
            b = random.random()
            lefts.append(min(a, b))
            rights.append(max(a, b))
        if max(lefts) < min(rights):
            hits += 1
    return hits / trials


# ---- USER INPUT ----
n = int(input("Enter number of intervals n: "))

print("\nExact probability:", exact_probability(n))
print("Approx probability:", estimate_probability(n))
