import numpy as np


def plane_stress_matrix(E, nu):
    """
    Create the constitutive matrix D for
    a linear elastic isotropic plane-stress material.

    Parameters
    ----------
    E : float
        Young's modulus.

    nu : float
        Poisson's ratio.

    Returns
    -------
    D : ndarray, shape (3, 3)
        Plane-stress material stiffness matrix.
    """

    # Check material properties
    if E <= 0:
        raise ValueError("Young's modulus E must be positive.")

    if not (-1.0 < nu < 0.5):
        raise ValueError(
            "Poisson's ratio must satisfy -1 < nu < 0.5."
        )

    coefficient = E / (1 - nu**2)

    D = coefficient * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])

    return D


if __name__ == "__main__":

    # Temporary test material
    E = 70e9
    nu = 0.33

    D = plane_stress_matrix(E, nu)

    print("Plane Stress Material Matrix Test")
    print("---------------------------------")

    print("\nYoung's modulus:")
    print(E, "Pa")

    print("\nPoisson's ratio:")
    print(nu)

    print("\nD matrix:")
    print(D)