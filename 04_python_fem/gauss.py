import numpy as np


def gauss_quadrature_2x2():
    """
    Return the Gauss points and weights
    for 2x2 Gauss quadrature.

    Returns
    -------
    points : ndarray, shape (4, 2)
        Natural coordinates [xi, eta] of
        the four Gauss integration points.

    weights : ndarray, shape (4,)
        Corresponding Gauss weights.
    """

    # Location of the two 1D Gauss points
    a = 1.0 / np.sqrt(3.0)

    # Four points in the natural coordinate system
    points = np.array([
        [-a, -a],
        [ a, -a],
        [ a,  a],
        [-a,  a]
    ])

    # Weight of every point
    weights = np.ones(4)

    return points, weights


if __name__ == "__main__":

    points, weights = gauss_quadrature_2x2()

    print("2x2 Gauss Quadrature Test")
    print("-------------------------")

    print("\nGauss points [xi, eta]:")
    print(points)

    print("\nGauss weights:")
    print(weights)

    print("\nSum of weights:")
    print(np.sum(weights))