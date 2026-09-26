import numpy as np

from jacobian import calculate_physical_derivatives


def build_B_matrix(xi, eta, coordinates):
    """
    Construct the Q4 strain-displacement matrix B.

    Parameters
    ----------
    xi : float
        Natural coordinate xi.

    eta : float
        Natural coordinate eta.

    coordinates : ndarray, shape (4, 2)
        Physical coordinates of the four element nodes.

    Returns
    -------
    B : ndarray, shape (3, 8)
        Q4 strain-displacement matrix.
    """

    # Get derivatives of shape functions
    # with respect to physical coordinates x and y.
    dN_dx, dN_dy = calculate_physical_derivatives(
        xi,
        eta,
        coordinates
    )

    # Initialize B matrix
    B = np.zeros((3, 8))

    # Fill B matrix
    for i in range(4):

        # Position of node i in the displacement vector
        col = 2 * i

        # epsilon_x = du/dx
        B[0, col] = dN_dx[i]

        # epsilon_y = dv/dy
        B[1, col + 1] = dN_dy[i]

        # gamma_xy = du/dy + dv/dx
        B[2, col] = dN_dy[i]
        B[2, col + 1] = dN_dx[i]

    return B


if __name__ == "__main__":

    # ------------------------------------------------
    # Test with a simple rectangular Q4 element
    # ------------------------------------------------

    coordinates = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [0.0, 1.0]
    ])

    xi = 0.0
    eta = 0.0

    B = build_B_matrix(
        xi,
        eta,
        coordinates
    )

    print("Q4 B Matrix Test")
    print("-----------------")

    print("\nElement coordinates:")
    print(coordinates)

    print("\nAt xi =", xi, "eta =", eta)

    print("\nB matrix:")
    print(B)

    print("\nB matrix shape:")
    print(B.shape)

def element_stiffness_matrix(
    coordinates,
    thickness,
    D
):
    """
    Calculate the Q4 element stiffness matrix
    using 2x2 Gauss quadrature.

    Parameters
    ----------
    coordinates : ndarray, shape (4, 2)
        Coordinates of the four element nodes.

    thickness : float
        Element thickness.

    D : ndarray, shape (3, 3)
        Plane-stress material matrix.

    Returns
    -------
    Ke : ndarray, shape (8, 8)
        Q4 element stiffness matrix.
    """

    # Import Gauss quadrature
    from gauss import gauss_quadrature_2x2

    # Get Gauss points and weights
    points, weights = gauss_quadrature_2x2()

    # Initialize 8x8 element stiffness matrix
    Ke = np.zeros((8, 8))

    # Loop over the four Gauss points
    for g in range(4):

        xi = points[g, 0]
        eta = points[g, 1]

        weight = weights[g]

        # Calculate B matrix
        B = build_B_matrix(
            xi,
            eta,
            coordinates
        )

        # Calculate Jacobian
        from jacobian import calculate_jacobian

        J, det_J, inv_J = calculate_jacobian(
            xi,
            eta,
            coordinates
        )

        # Gauss point contribution
        Ke += (
            B.T
            @ D
            @ B
            * det_J
            * weight
            * thickness
        )

    return Ke
if __name__ == "__main__":

    # ------------------------------------------------
    # Test Q4 element stiffness matrix
    # ------------------------------------------------

    coordinates = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [0.0, 1.0]
    ])

    thickness = 0.01

    # Example material:
    # E = 70 GPa
    # nu = 0.33

    from material import plane_stress_matrix

    E = 70e9
    nu = 0.33

    D = plane_stress_matrix(E, nu)

    Ke = element_stiffness_matrix(
        coordinates,
        thickness,
        D
    )

    print("Q4 Element Stiffness Matrix Test")
    print("---------------------------------")

    print("\nElement coordinates:")
    print(coordinates)

    print("\nThickness:")
    print(thickness, "m")

    print("\nElement stiffness matrix Ke:")
    print(Ke)

    print("\nKe shape:")
    print(Ke.shape)

    print("\nSymmetry error:")
    print(np.max(np.abs(Ke - Ke.T)))

    # ------------------------------------------------
    # Rigid body mode tests
    # ------------------------------------------------

    rigid_x = np.array([
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0,
        1.0, 0.0
    ])

    rigid_y = np.array([
        0.0, 1.0,
        0.0, 1.0,
        0.0, 1.0,
        0.0, 1.0
    ])

    rigid_rotation = np.array([
        0.0, 0.0,
        0.0, 2.0,
        -1.0, 2.0,
        -1.0, 0.0
    ])

    print("\nRigid body mode tests:")

    print("\nRigid translation X:")
    print(Ke @ rigid_x)

    print("\nRigid translation Y:")
    print(Ke @ rigid_y)

    print("\nRigid rotation:")
    print(Ke @ rigid_rotation)

    # ------------------------------------------------
    # Eigenvalue test
    # ------------------------------------------------

    eigenvalues = np.linalg.eigvalsh(Ke)

    print("\nEigenvalues of Ke:")
    print(eigenvalues)

    print("\nNumber of near-zero eigenvalues:")

    tolerance = 1e-6 * np.max(np.abs(eigenvalues))

    number_zero = np.sum(
        np.abs(eigenvalues) < tolerance
    )

    print(number_zero)

    print("\nEigenvalue tolerance:")
    print(tolerance)