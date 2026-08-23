import numpy as np

def cubic_spline_interpolate(x_eval, x, y):
    """Evaluates a Natural Cubic Spline at query points x_eval.

    Parameters:
        x : 1D array-like, strictly increasing knot x-coordinates
        y : 1D array-like, corresponding y-coordinates
        x_eval : 1D array-like, points where to interpolate
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)

    n = len(x) - 1
    h = np.diff(x)  # h_i = x_{i+1} - x_i

    # 1. Build the tridiagonal matrix system A * M = b
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    # Natural boundary condition: M[0] = M[n] = 0
    A[0, 0] = 1.0
    A[n, n] = 1.0

    # Fill interior rows for continuous 1st derivatives
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Solve for second derivatives M
    M = np.linalg.solve(A, b)

    # 2. Find which interval [x_i, x_{i+1}] each evaluation point falls into
    idx = np.searchsorted(x, x_eval) - 1
    idx = np.clip(idx, 0, n - 1)

    # 3. Evaluate the cubic polynomial on each interval
    dx_left = x_eval - x[idx]
    dx_right = x[idx + 1] - x_eval
    h_i = h[idx]

    # Spline piecewise formula
    term1 = M[idx + 1] * (dx_left**3) / (6.0 * h_i)
    term2 = M[idx] * (dx_right**3) / (6.0 * h_i)
    term3 = (y[idx + 1] / h_i - M[idx + 1] * h_i / 6.0) * dx_left
    term4 = (y[idx] / h_i - M[idx] * h_i / 6.0) * dx_right

    return term1 + term2 + term3 + term4