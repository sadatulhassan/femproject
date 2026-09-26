import numpy as np
from element_q4 import build_B_matrix
from gauss import gauss_quadrature_2x2


def element_strain_stress(coordinates, element_displacements, D):
    """
    Calculate strain and stress at the 2x2 Gauss points
    of one Q4 element.

    Parameters
    ----------
    coordinates : ndarray, shape (4, 2)
        Element nodal coordinates.

    element_displacements : ndarray, shape (8,)
        Element displacement vector:
        [u1,v1,u2,v2,u3,v3,u4,v4]

    D : ndarray, shape (3,3)
        Plane-stress material matrix.

    Returns
    -------
    strains : ndarray, shape (4,3)
        Strain at each Gauss point.

    stresses : ndarray, shape (4,3)
        Stress at each Gauss point.
    """

    points, weights = gauss_quadrature_2x2()

    strains = []
    stresses = []

    for point in points:

        xi = point[0]
        eta = point[1]

        # Construct B matrix at this Gauss point
        B = build_B_matrix(xi, eta, coordinates)

        # Strain = B d
        strain = B @ element_displacements

        # Stress = D strain
        stress = D @ strain

        strains.append(strain)
        stresses.append(stress)

    return np.array(strains), np.array(stresses)


if __name__ == "__main__":

    print("Q4 Post-Processing Test")
    print("-----------------------")

    # Rectangle element
    coordinates = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [0.0, 1.0]
    ])

    # Material
    E = 70e9
    nu = 0.33

    from material import plane_stress_matrix

    D = plane_stress_matrix(E, nu)

    # Example element displacement
    element_displacements = np.array([
        0.0,
        0.0,
        0.001,
        0.0,
        0.001,
        0.0005,
        0.0,
        0.0005
    ])

    strains, stresses = element_strain_stress(
        coordinates,
        element_displacements,
        D
    )

    print("\nStrains at Gauss points:")
    print(strains)

    print("\nStresses at Gauss points (Pa):")
    print(stresses)