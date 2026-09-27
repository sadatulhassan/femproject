import numpy as np


def create_rectangular_mesh(length, height, nx, ny):
    """
    Create a structured rectangular Q4 mesh.

    Parameters
    ----------
    length : float
        Length of the rectangle in x-direction.

    height : float
        Height of the rectangle in y-direction.

    nx : int
        Number of elements along x.

    ny : int
        Number of elements along y.

    Returns
    -------
    nodes : ndarray
        Node coordinates, shape (N, 2).

    elements : ndarray
        Q4 element connectivity, shape (M, 4).
    """

    # --------------------------------------------------------
    # 1. Create nodal coordinates
    # --------------------------------------------------------

    x_values = np.linspace(
        0.0,
        length,
        nx + 1
    )

    y_values = np.linspace(
        0.0,
        height,
        ny + 1
    )

    nodes = []

    for j in range(ny + 1):

        for i in range(nx + 1):

            x = x_values[i]
            y = y_values[j]

            nodes.append([x, y])

    nodes = np.array(nodes, dtype=float)


    # --------------------------------------------------------
    # 2. Create element connectivity
    # --------------------------------------------------------

    elements = []

    for j in range(ny):

        for i in range(nx):

            # Bottom-left node
            n1 = j * (nx + 1) + i

            # Bottom-right node
            n2 = n1 + 1

            # Top-right node
            n3 = n2 + (nx + 1)

            # Top-left node
            n4 = n1 + (nx + 1)

            # Our Q4 convention is:
            #
            # n4 -------- n3
            # |            |
            # |            |
            # n1 -------- n2

            elements.append([
                n1,
                n2,
                n3,
                n4
            ])

    elements = np.array(
        elements,
        dtype=int
    )

    return nodes, elements


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("Rectangular Mesh Generator Test")
    print("--------------------------------")

    nodes, elements = create_rectangular_mesh(
        length=2.0,
        height=1.0,
        nx=2,
        ny=1
    )

    print("\nNodes:")
    for i, node in enumerate(nodes):
        print(
            f"Node {i}: "
            f"({node[0]:.2f}, {node[1]:.2f})"
        )

    print("\nElements:")
    for i, element in enumerate(elements):
        print(
            f"Element {i + 1}: "
            f"{element}"
        )

    print("\nNumber of nodes:", len(nodes))
    print("Number of elements:", len(elements))