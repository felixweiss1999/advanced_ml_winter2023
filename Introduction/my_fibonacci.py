"""
Different implementations of the computation of the 
first n Fibonacci numbers. This module contains two 
functions. The first one displays a list of the first 
n Fibonacci numbers, the second one returns a list
of them for further usage.
"""

def fib(n):
    """Display the first n Fibonacci numbers."""
    a, b = 0, 1
    for i in range(n):
        print(a, end=' ')
        a, b = b, a + b
    print()

def fib2(n):
    """Return the first n Fibonacci numbers as list."""
    Fibonacci = []
    a, b = 0, 1
    for i in range(n):
        Fibonacci.append(a)
        a, b = b, a + b
    return Fibonacci

# Is the file loaded as script or imported as module?
if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))
else:
    print("Loaded module my_fibonacci.")
