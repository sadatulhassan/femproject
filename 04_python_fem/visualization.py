
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
    """

    triangles = []

    for element in elements:

        n1, n2, n3, n4 = element

        triangles.append([n1, n2, n3])
        triangles.append([n1, n3, n4])

    return np.array(triangles, dtype=int)


# =================================================
# GAUSS-POINT STRESS -> NODAL STRESS
# =================================================

def gauss_to_nodal_values(
    elements,
    gauss_values,
    number_of_nodes
):
    """
    Recover element Gauss-point values to nodal values
    using simple averaging.

    Parameters
    ----------
    elements : ndarray
        Q4 element connectivity.
        Shape = (number_of_elements, 4)

    gauss_values : ndarray
        Values at the 4 Gauss points of every element.
        Shape = (number_of_elements, 4)

    number_of_nodes : int
        Total number of nodes.

    Returns
    -------
    nodal_values : ndarray
        Recovered value at every node.
        Shape = (number_of_nodes,)

    Notes
    -----
    This is a simple arithmetic averaging recovery.
    It is intended for visualization, not for changing
    the actual FEM stress calculation.
    """

    nodal_sum = np.zeros(number_of_nodes)
    nodal_count = np.zeros(number_of_nodes)

    for element_index, element_nodes in enumerate(elements):

        # Four Gauss-point values
        element_values = gauss_values[element_index]

        # Average the four Gauss-point values
        element_average = np.mean(element_values)

        # Add the element value to each of its nodes
        for node in element_nodes:

            nodal_sum[node] += element_average
            nodal_count[node] += 1.0

    # Avoid division by zero
    nodal_values = np.zeros(number_of_nodes)

    valid_nodes = nodal_count > 0

    nodal_values[valid_nodes] = (
        nodal_sum[valid_nodes]
        / nodal_count[valid_nodes]
    )

    return nodal_values


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
    Plot a scalar FEM result over a Q4 mesh.

    Parameters
    ----------
    nodes : ndarray
        Node coordinates.
        Shape = (number_of_nodes, 2)

    elements : ndarray
        Q4 connectivity.
        Shape = (number_of_elements, 4)

    nodal_values : ndarray
        Scalar value at every node.
        Shape = (number_of_nodes,)

    title : str
        Plot title.

    colorbar_label : str
        Colorbar label.

    units : str
        Physical units.

    save_path : str, optional
        Path for saving the figure.
    """

    nodes = np.asarray(nodes)
    elements = np.asarray(elements)
    nodal_values = np.asarray(nodal_values)

    # -------------------------------------------------
    # BASIC VALIDATION
    # -------------------------------------------------

    if nodes.ndim != 2 or nodes.shape[1] != 2:

        raise ValueError(
            "nodes must have shape "
            "(number_of_nodes, 2)."
        )

    if elements.ndim != 2 or elements.shape[1] != 4:

        raise ValueError(
            "elements must have shape "
            "(number_of_elements, 4)."
        )

    if len(nodal_values) != len(nodes):

        raise ValueError(
            "Number of nodal values must equal "
            "number of nodes."
        )

    # -------------------------------------------------
    # Q4 -> TRIANGLES
    # -------------------------------------------------

    triangles = q4_to_triangles(elements)

    # -------------------------------------------------
    # CREATE TRIANGULATION
    # -------------------------------------------------

    triangulation = mtri.Triangulation(
        nodes[:, 0],
        nodes[:, 1],
        triangles
    )

    # -------------------------------------------------
    # CREATE FIGURE
    # -------------------------------------------------

    plt.figure(figsize=(10, 6))

    contour = plt.tricontourf(
        triangulation,
        nodal_values,
        levels=30
    )

    # -------------------------------------------------
    # MESH EDGES
    # -------------------------------------------------

    plt.triplot(
        triangulation,
        linewidth=0.25,
        alpha=0.35
    )

    # -------------------------------------------------
    # COLORBAR
    # -------------------------------------------------

    cbar = plt.colorbar(contour)

    if units:

        cbar.set_label(
            f"{colorbar_label} ({units})"
        )

    else:

        cbar.set_label(
            colorbar_label
        )

    # -------------------------------------------------
    # LABELS
    # -------------------------------------------------

    plt.xlabel("X coordinate (mm)")
    plt.ylabel("Y coordinate (mm)")
    plt.title(title)

    plt.axis("equal")
    plt.tight_layout()

    # -------------------------------------------------
    # SAVE FIGURE
    # -------------------------------------------------

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"Figure saved to: {save_path}"
        )

    # -------------------------------------------------
    # SHOW FIGURE
    # -------------------------------------------------

    plt.show()


# =================================================
# PLOT VON MISES STRESS
# =================================================

def plot_von_mises(
    nodes,
    elements,
    all_von_mises,
    save_path=None
):
    """
    Plot recovered Von Mises stress.

    Parameters
    ----------
    nodes : ndarray
        Node coordinates.

    elements : ndarray
        Q4 connectivity.

    all_von_mises : ndarray
        Von Mises stress at the four Gauss points
        of every element.

        Shape:
        (number_of_elements, 4)

    save_path : str, optional
        Output figure path.
    """

    # -----------------------------------------------
    # Convert Gauss-point stresses to nodal values
    # -----------------------------------------------

    nodal_vm = gauss_to_nodal_values(
        elements,
        all_von_mises,
        len(nodes)
    )

    # -----------------------------------------------
    # Plot
    # -----------------------------------------------

    plot_contour(
        nodes=nodes,
        elements=elements,
        nodal_values=nodal_vm,
        title="Von Mises Stress",
        colorbar_label="Von Mises Stress",
        units="MPa",
        save_path=save_path
    )


# =================================================
# PLOT DISPLACEMENT MAGNITUDE
# =================================================

def calculate_displacement_magnitude(displacements):
    """
    Calculate displacement magnitude at every node.

    Parameters
    ----------
    displacements : ndarray
        Global displacement vector:

        [Ux1, Uy1, Ux2, Uy2, ...]

    Returns
    -------
    displacement_magnitude : ndarray
        Magnitude of displacement at each node.
    """

    ux = displacements[0::2]
    uy = displacements[1::2]

    displacement_magnitude = np.sqrt(
        ux**2 + uy**2
    )

    return displacement_magnitude


def plot_displacement(
    nodes,
    elements,
    displacements,
    save_path=None
):
    """
    Plot total displacement magnitude.
    """

    displacement_magnitude = (
        calculate_displacement_magnitude(
            displacements
        )
    )

    plot_contour(
        nodes=nodes,
        elements=elements,
        nodal_values=displacement_magnitude,
        title="Displacement Magnitude",
        colorbar_label="Displacement",
        units="mm",
        save_path=save_path
    )


# =================================================
# PLOT SIGMA X
# =================================================

def plot_sigma_x(
    nodes,
    elements,
    all_stresses,
    save_path=None
):
    """
    Plot recovered sigma_x stress.
    """

    sigma_x_gp = all_stresses[:, :, 0]

    nodal_sigma_x = gauss_to_nodal_values(
        elements,
        sigma_x_gp,
        len(nodes)
    )

    plot_contour(
        nodes=nodes,
        elements=elements,
        nodal_values=nodal_sigma_x,
        title="Normal Stress σx",
        colorbar_label="σx",
        units="MPa",
        save_path=save_path
    )


# =================================================
# PLOT SIGMA Y
# =================================================

def plot_sigma_y(
    nodes,
    elements,
    all_stresses,
    save_path=None
):
    """
    Plot recovered sigma_y stress.
    """

    sigma_y_gp = all_stresses[:, :, 1]

    nodal_sigma_y = gauss_to_nodal_values(
        elements,
        sigma_y_gp,
        len(nodes)
    )

    plot_contour(
        nodes=nodes,
        elements=elements,
        nodal_values=nodal_sigma_y,
        title="Normal Stress σy",
        colorbar_label="σy",
        units="MPa",
        save_path=save_path
    )


# =================================================
# PLOT SHEAR STRESS
# =================================================

def plot_tau_xy(
    nodes,
    elements,
    all_stresses,
    save_path=None
):
    """
    Plot recovered shear stress tau_xy.
    """

    tau_xy_gp = all_stresses[:, :, 2]

    nodal_tau_xy = gauss_to_nodal_values(
        elements,
        tau_xy_gp,
        len(nodes)
    )

    plot_contour(
        nodes=nodes,
        elements=elements,
        nodal_values=nodal_tau_xy,
        title="Shear Stress τxy",
        colorbar_label="τxy",
        units="MPa",
        save_path=save_path
    )
