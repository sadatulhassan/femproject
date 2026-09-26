import numpy as np


def solve_system(K, F):
    """
    Solve the FEM system:

        K d = F

    Parameters
    ----------
    K : ndarray
        Global stiffness matrix.

    F : ndarray
        Global force vector.

    Returns
    -------
    d : ndarray
        Nodal displacement vector.
    """

    d = np.linalg.solve(K, F)

    return d


if __name__ == "__main__":

    print("FEM Solver Test")
    print("----------------")

    K = np.array([
        [10.0, 2.0],
        [2.0, 8.0]
    ])

    F = np.array([
        100.0,
        50.0
    ])

    d = solve_system(K, F)

    print("\nK:")
    print(K)

    print("\nF:")
    print(F)

    print("\nDisplacement vector d:")
    print(d)

    print("\nVerification Kd:")
    print(K @ d)