import numpy as np

from shape_functions import shape_function_derivatives


def calculate_jacobian(xi, eta, coordinates):
    """
    Calculate the Jacobian matrix for a Q4 element.

    Node ordering:

        4 -------- 3
        |          |
        |          |
        1 -------- 2

    Parameters
    ----------
    xi : float
        Natural coordinate.

    eta : float
        Natural coordinate.

    coordinates : ndarray, shape (4, 2)
        Physical coordinates of the four nodes:

        [[x1, y1],
         [x2, y2],
         [x3, y3],
         [x4, y4]]

    Returns
    -------
    J : ndarray, shape (2, 2)
        Jacobian matrix.

    det_J : float
        Determinant of the Jacobian.

    inv_J : ndarray, shape (2, 2)
        Inverse Jacobian matrix.
    """

    # Shape function derivatives in natural coordinates
    dN_dxi, dN_deta = shape_function_derivatives(xi, eta)

    # -------------------------------------------------
    # Calculate derivatives of physical coordinates
    # -------------------------------------------------

    dx_dxi = np.dot(dN_dxi, coordinates[:, 0])
    dy_dxi = np.dot(dN_dxi, coordinates[:, 1])

    dx_deta = np.dot(dN_deta, coordinates[:, 0])
    dy_deta = np.dot(dN_deta, coordinates[:, 1])

    # -------------------------------------------------
    # Construct Jacobian matrix
    # -------------------------------------------------

    J = np.array([
        [dx_dxi, dy_dxi],
        [dx_deta, dy_deta]
    ])

    # -------------------------------------------------
    # Determinant
    # -------------------------------------------------

    det_J = np.linalg.det(J)

    # Check for invalid/inverted element
    if det_J <= 0:
        raise ValueError(
            f"Invalid Q4 element: det(J) = {det_J:.6e}"
        )

    # -------------------------------------------------
    # Inverse Jacobian
    # -------------------------------------------------

    inv_J = np.linalg.inv(J)

    return J, det_J, inv_J


def calculate_physical_derivatives(xi, eta, coordinates):
    """
    Convert derivatives of shape functions from
    natural coordinates (xi, eta) to physical
    coordinates (x, y).

    Returns
    -------
    dN_dx : ndarray, shape (4,)
        Derivatives of shape functions with respect to x.

    dN_dy : ndarray, shape (4,)
        Derivatives of shape functions with respect to y.
    """

    # Shape function derivatives in natural coordinates
    dN_dxi, dN_deta = shape_function_derivatives(xi, eta)

    # Calculate Jacobian
    J, det_J, inv_J = calculate_jacobian(
        xi, eta, coordinates
    )

    dN_dx = np.zeros(4)
    dN_dy = np.zeros(4)

    # -------------------------------------------------
    # Convert natural derivatives to physical
    # derivatives
    #
    # [dN/dx]       [J]^-T [dN/dxi]
    # [dN/dy]   =         [dN/deta]
    #
    # Therefore:
    #
    # physical_derivative = inv_J.T @ natural_derivative
    # -------------------------------------------------

    for i in range(4):

        natural_derivative = np.array([
            dN_dxi[i],
            dN_deta[i]
        ])

        physical_derivative = inv_J.T @ natural_derivative

        dN_dx[i] = physical_derivative[0]
        dN_dy[i] = physical_derivative[1]

    return dN_dx, dN_dy


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    # -------------------------------------------------
    # Test with a simple rectangular Q4 element
    # -------------------------------------------------

    coordinates = np.array([
        [0.0, 0.0],   # Node 1
        [2.0, 0.0],   # Node 2
        [2.0, 1.0],   # Node 3
        [0.0, 1.0]    # Node 4
    ])

    xi = 0.0
    eta = 0.0

    # Calculate Jacobian
    J, det_J, inv_J = calculate_jacobian(
        xi, eta, coordinates
    )

    # Calculate physical derivatives
    dN_dx, dN_dy = calculate_physical_derivatives(
        xi, eta, coordinates
    )

    # -------------------------------------------------
    # Print results
    # -------------------------------------------------

    print("Q4 Jacobian Test")
    print("----------------")

    print("\nCoordinates:")
    print(coordinates)

    print("\nAt xi =", xi, "eta =", eta)

    print("\nJacobian J:")
    print(J)

    print("\nDeterminant of J:")
    print(det_J)

    print("\nInverse Jacobian:")
    print(inv_J)

    print("\ndN/dx:")
    print(dN_dx)

    print("\ndN/dy:")
    print(dN_dy)