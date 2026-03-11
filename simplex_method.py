#!/usr/bin/env python3
"""Simplex method — linear programming solver."""
import sys

def simplex(c, A, b):
    """Maximize c·x subject to Ax <= b, x >= 0"""
    m, n = len(A), len(c)
    # Build tableau with slack variables
    tableau = []
    for i in range(m):
        row = A[i][:] + [0]*m + [b[i]]
        row[n+i] = 1
        tableau.append(row)
    obj = [-ci for ci in c] + [0]*m + [0]
    tableau.append(obj)
    basis = list(range(n, n+m))
    
    while True:
        # Find pivot column (most negative in objective)
        obj_row = tableau[-1]
        pivot_col = min(range(n+m), key=lambda j: obj_row[j])
        if obj_row[pivot_col] >= -1e-10: break  # optimal
        
        # Find pivot row (minimum ratio test)
        ratios = []
        for i in range(m):
            if tableau[i][pivot_col] > 1e-10:
                ratios.append((tableau[i][-1] / tableau[i][pivot_col], i))
        if not ratios: return None, None  # unbounded
        _, pivot_row = min(ratios)
        
        # Pivot
        pivot_val = tableau[pivot_row][pivot_col]
        tableau[pivot_row] = [x/pivot_val for x in tableau[pivot_row]]
        for i in range(m+1):
            if i != pivot_row:
                factor = tableau[i][pivot_col]
                tableau[i] = [tableau[i][j] - factor*tableau[pivot_row][j] for j in range(n+m+1)]
        basis[pivot_row] = pivot_col
    
    x = [0]*(n+m)
    for i, bi in enumerate(basis):
        x[bi] = tableau[i][-1]
    return x[:n], tableau[-1][-1]

if __name__ == "__main__":
    # Maximize 5x + 4y subject to: 6x + 4y <= 24, x + 2y <= 6, x,y >= 0
    c = [5, 4]
    A = [[6, 4], [1, 2]]
    b = [24, 6]
    x, val = simplex(c, A, b)
    if x:
        print(f"Maximize {c[0]}x + {c[1]}y")
        print(f"  6x + 4y <= 24")
        print(f"  x + 2y <= 6")
        print(f"Solution: x={x[0]:.2f}, y={x[1]:.2f}")
        print(f"Optimal value: {val:.2f}")
    else:
        print("Unbounded or infeasible")
