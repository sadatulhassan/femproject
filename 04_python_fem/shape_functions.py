import numpy as np


def shape_functions(xi, eta):
    """
    Q4 shape functions.

    Node numbering:

        4 -------- 3
        |          |
        |          |
        1 -------- 2

    Parameters
    ----------
    xi : float
        Natural coordinate (-1 <= xi <= 1)

    eta : float
        Natural coordinate (-1 <= eta <= 1)

    Returns
    -------
    N : numpy.ndarray
        Shape functions [N1, N2, N3, N4]
    """

    N1 = 0.25 * (1 - xi) * (1 - eta)
    N2 = 0.25 * (1 + xi) * (1 - eta)
    N3 = 0.25 * (1 + xi) * (1 + eta)
    N4 = 0.25 * (1 - xi) * (1 + eta)

    return np.array([N1, N2, N3, N4])


def shape_function_derivatives(xi, eta):
    """
    Derivatives of Q4 shape functions with respect
    to natural coordinates xi and eta.

    Returns
    -------
    dN_dxi : numpy.ndarray
        Derivatives with respect to xi.

    dN_deta : numpy.ndarray
        Derivatives with respect to eta.
    """

    dN_dxi = 0.25 * np.array([
        -(1 - eta),
         (1 - eta),
         (1 + eta),
        -(1 + eta)
    ])

    dN_deta = 0.25 * np.array([
        -(1 - xi),
        -(1 + xi),
         (1 + xi),
         (1 - xi)
    ])

    return dN_dxi, dN_deta


if __name__ == "__main__":

    # Test at the center of the Q4 element
    xi = 0.0
    eta = 0.0

    N = shape_functions(xi, eta)
    dN_dxi, dN_deta = shape_function_derivatives(xi, eta)

    print("Q4 Shape Function Test")
    print("----------------------")

    print("\nAt xi =", xi, "eta =", eta)

    print("\nShape functions:")
    print(N)

    print("\ndN/dxi:")
    print(dN_dxi)

    print("\ndN/deta:")
    print(dN_deta)

    print("\nSum of shape functions:")
    print(np.sum(N))