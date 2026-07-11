"""Computes the n-th Fibonacci number for the pricing-tier lookup table."""


def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (0-indexed)."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
