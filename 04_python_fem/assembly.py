import numpy as np


def element_dof_indices(element_nodes):
    """
    Convert element node numbers into global DOF indices.

    Parameters
    ----------
    element_nodes : array-like
        Four global node numbers for a Q4 element.

    Returns
    -------
    dof_indices : ndarray
        Eight global DOF indices.
    """

    dof_indices = []

    for node in element_nodes:

        # Each node has:
        # DOF 2*node     -> u
        # DOF 2*node + 1 -> v

        dof_indices.append(2 * node)
        dof_indices.append(2 * node + 1)

    return np.array(dof_indices, dtype=int)

def assemble_element(
    K_global,
    K_element,
    element_nodes
):
    """
    Add one element stiffness matrix
    into the global stiffness matrix.

    Parameters
    ----------
    K_global : ndarray
        Global stiffness matrix.

    K_element : ndarray
        8x8 element stiffness matrix.

    element_nodes : array-like
        Four global node numbers of the element.

    Returns
    -------
    K_global : ndarray
        Updated global stiffness matrix.
    """

    # Get the global DOF locations
    dof_indices = element_dof_indices(
        element_nodes
    )

    # Add element contributions
    for i in range(8):

        I = dof_indices[i]

        for j in range(8):

            J = dof_indices[j]

            K_global[I, J] += K_element[i, j]

    return K_global

if __name__ == "__main__":

    print("Global Assembly Test")
    print("--------------------")

    # Six nodes
    number_of_nodes = 6

    # Global stiffness matrix
    K_global = np.zeros(
        (2 * number_of_nodes,
         2 * number_of_nodes)
    )

    # Fake element stiffness matrix
    # We use a simple matrix filled with 1s
    # only for testing assembly.
    Ke = np.ones((8, 8))

    # Element 1
    element_1_nodes = np.array([
        0, 1, 4, 3
    ])

    # Element 2
    element_2_nodes = np.array([
        1, 2, 5, 4
    ])

    # Assemble both
    K_global = assemble_element(
        K_global,
        Ke,
        element_1_nodes
    )

    K_global = assemble_element(
        K_global,
        Ke,
        element_2_nodes
    )

    print("\nElement 1 global DOFs:")
    print(element_dof_indices(element_1_nodes))

    print("\nElement 2 global DOFs:")
    print(element_dof_indices(element_2_nodes))

    print("\nGlobal stiffness matrix:")
    print(K_global)

    print("\nGlobal matrix shape:")
    print(K_global.shape)