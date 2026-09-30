
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


# =================================================
# Q4 -> TRIANGLE CONNECTIVITY FOR PLOTTING
# =================================================

def q4_to_triangles(elements):
    """
    Convert Q4 quadrilateral elements into triangles
    for matplotlib contour plotting.

    This conversion is ONLY for visualization.
    It does not change the FEM element formulation.

    Q4 node ordering:

        4 -------- 3
        |          |
        |          |
        1 -------- 2

    Each Q4 is divided into:

        1 ---- 2
        |   /  |
        |  /   |
        4 ---- 3

    Parameters
    ----------
    elements : ndarray, shape (n_elements, 4)
        Q4 element connectivity.

    Returns
    -------
    triangles : ndarray, shape (2*n_elements, 3)
        Triangle connectivity.
    """

    triangles = []

    for element in elements:

        n1, n2, n3, n4 = element

        # First triangle
        triangles.append([n1, n2, n3])

        # Second triangle
        triangles.append([n1, n3, n4])

    return np.array(triangles, dtype=int)


# =================================================
# GENERAL CONTOUR PLOT
# =================================================

def plot_contour(
    nodes,
    elements,
    nodal_values,
    title,
    colorbar_label,
    units="",
    save_path=None
):
    """
    Plot a scalar stress/result contour over a Q4 mesh.

    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
        Node coordinates [x, y].

    elements : ndarray, shape (n_elements, 4)
        Q4 element connectivity.

    nodal_values : ndarray, shape (n_nodes,)
        Scalar value associated with each node.

    title : str
        Plot title.

    colorbar_label : str
        Label for the colorbar.

    units : str, optional
        Units shown in the colorbar.

    save_path : str, optional
        File path for saving the figure.
    """

    nodes = np.asarray(nodes)
    elements = np.asarray(elements)
    nodal_values = np.asarray(nodal_values)

    # -------------------------------------------------
    # Basic validation
    # -------------------------------------------------

    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(
            "nodes must have shape (number_of_nodes, 2)."
        )

    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError(
            "elements must have shape (number_of_elements, 4)."
        )

    if len(nodal_values) != len(nodes):
        raise ValueError(
            "Number of nodal values must equal number of nodes."
        )

    # -------------------------------------------------
    # Convert Q4 elements to triangles
    # -------------------------------------------------

    triangles = q4_to_triangles(elements)

    # -------------------------------------------------
    # Create triangulation
    # -------------------------------------------------

    triangulation = mtri.Triangulation(
        nodes[:, 0],
        nodes[:, 1],
        triangles
    )

    # -------------------------------------------------
    # Create figure
    # -------------------------------------------------

    plt.figure(figsize=(10, 6))

    contour = plt.tricontourf(
        triangulation,
        nodal_values,
        levels=30
    )

    # Mesh edges
    plt.triplot(
        triangulation,
        linewidth=0.25,
        alpha=0.35
    )

    # -------------------------------------------------
    # Colorbar
    # -------------------------------------------------

    cbar = plt.colorbar(contour)

    if units:
        cbar.set_label(
            f"{colorbar_label} ({units})"
        )
    else:
        cbar.set_label(colorbar_label)

    # -------------------------------------------------
    # Labels
    # -------------------------------------------------

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title(title)

    plt.axis("equal")
    plt.tight_layout()

    # -------------------------------------------------
