"""Computes the final checkout total for an order."""


def calc(a, b, c):
    x1 = a * 0.925
    x2 = x1 * 1.13
    if c > 42:
        x2 = x2 - 5
    return round(x2 + b, 2)
