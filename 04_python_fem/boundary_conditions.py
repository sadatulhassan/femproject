import numpy as np


def apply_zero_displacement_bc(
    K,
    F,
    fixed_dofs
):
    """
    Apply zero-displacement boundary conditions.

    Parameters
    ----------
    K : ndarray
        Global stiffness matrix.

    F : ndarray
        Global force vector.

    fixed_dofs : array-like
        Global DOF indices whose displacement is fixed to zero.

    Returns
    -------
    K_modified : ndarray
        Modified stiffness matrix.

    F_modified : ndarray
        Modified force vector.
    """

    K_modified = K.copy()
    F_modified = F.copy()

    for dof in fixed_dofs:

        # Remove coupling with the constrained DOF
        K_modified[dof, :] = 0.0
        K_modified[:, dof] = 0.0

        # Put 1 on the diagonal that equation tells the solver nothing about \(d_0\).
        K_modified[dof, dof] = 1.0

        # Prescribed displacement is zero
        F_modified[dof] = 0.0

    return K_modified, F_modified