import numpy as np


def create_load_vector(nodes, total_force, number_of_dofs):

    F = np.zeros(number_of_dofs)

    # ---------------------------------------------
    # RIGHT VERTICAL EDGE
    # ---------------------------------------------

    x_max = np.max(nodes[:, 0])

    tolerance = 1e-6

    right_nodes = np.where(
        np.isclose(
            nodes[:, 0],
            x_max,
            atol=tolerance
        )
    )[0]

    if len(right_nodes) == 0:
        raise ValueError("No right-edge nodes found.")

    # ---------------------------------------------
    # TOTAL TENSILE FORCE = 10000 N
    # ---------------------------------------------

    force_per_node = total_force / len(right_nodes)

    for node in right_nodes:

        # X DOF
        F[2 * node] += force_per_node

    print("\nLOAD INFORMATION")
    print("-" * 60)
    print("Right-edge nodes    :", len(right_nodes))
    print("Right-edge length   : 40 mm")
    print("Total tensile load  :", total_force, "N")
    print("Load per node       :", force_per_node, "N")
    print("Direction           : +X")

    return F

def create_fixed_dofs(nodes):
    """
    Identify the left-edge nodes and fix both
    X and Y displacement DOFs.

    Parameters
    ----------
    nodes : ndarray
        Node coordinates, shape (n_nodes, 2).

    Returns
    -------
    fixed_dofs : list
        Global DOF numbers that are fixed.
    """

    # Find left edge
    x_min = np.min(nodes[:, 0])

    tolerance = 1e-6

    left_nodes = np.where(
        np.isclose(
            nodes[:, 0],
            x_min,
            atol=tolerance
        )
    )[0]

    if len(left_nodes) == 0:
        raise ValueError("No left-edge nodes found.")

    fixed_dofs = []

    for node in left_nodes:

        # X displacement = 0
        fixed_dofs.append(2 * node)

        # Y displacement = 0
        fixed_dofs.append(2 * node + 1)

    print("\nBOUNDARY CONDITIONS")
    print("-" * 60)
    print("Left-edge nodes     :", len(left_nodes))
    print("Fixed DOFs          :", len(fixed_dofs))
    print("Left edge           : FIXED")

    return fixed_dofs


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
        Global DOF indices fixed to zero.

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

        # Remove coupling with constrained DOF
        K_modified[dof, :] = 0.0
        K_modified[:, dof] = 0.0

        # Prescribed displacement = 0
        K_modified[dof, dof] = 1.0

        # Prescribed displacement is zero
        F_modified[dof] = 0.0

    return K_modified, F_modified
def create_fixed_dofs(nodes):
    """
    Boundary conditions for uniaxial tension.

    Left edge:
        Ux = 0 for all left-edge nodes.

    One node on the left edge:
        Uy = 0 to remove rigid-body motion.

    This allows the remaining left-edge nodes
    to move freely in Y due to Poisson's effect.
    """

    x_min = np.min(nodes[:, 0])

    left_edge_nodes = np.where(
        np.isclose(nodes[:, 0], x_min, atol=1e-6)
    )[0]

    fixed_dofs = []

    # -------------------------------------------------
    # Fix X displacement of all left-edge nodes
    # -------------------------------------------------

    for node in left_edge_nodes:
        fixed_dofs.append(2 * node)

    # -------------------------------------------------
    # Fix Y displacement of only ONE left-edge node
    # -------------------------------------------------

    # Choose the first left-edge node
    reference_node = left_edge_nodes[0]

    fixed_dofs.append(2 * reference_node + 1)

    print("BOUNDARY CONDITIONS")
    print("-" * 60)
    print(f"Left-edge nodes     : {len(left_edge_nodes)}")
    print(f"Ux fixed DOFs       : {len(left_edge_nodes)}")
    print(f"Uy fixed node       : {reference_node}")
    print(f"Total fixed DOFs    : {len(fixed_dofs)}")
    print("Left edge Ux        : FIXED")
    print("One left node Uy    : FIXED")
    print("Remaining Uy        : FREE")

    return fixed_dofs