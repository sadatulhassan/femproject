import sys
sys.path.append("/home/132612011/femproject/04_python_fem")

import numpy as np

from mesh_generator import create_rectangular_mesh
from material import plane_stress_matrix
from element_q4 import element_stiffness_matrix
from assembly import assemble_element
from boundary_conditions import apply_zero_displacement_bc
from solver import solve_system


# -------------------------------------------------
# CANTILEVER PLATE PARAMETERS
# -------------------------------------------------

length = 2.0          # m
height = 1.0          # m
thickness = 0.01      # m

E = 70e9              # Pa
nu = 0.33

total_force = 1000.0  # N

D = plane_stress_matrix(E, nu)


# -------------------------------------------------
# CONSISTENT LOAD ON Q4 RIGHT EDGE
# -------------------------------------------------

def right_edge_load(element_length, thickness, total_force):

    fe = np.zeros(8)

    # Uniform traction on this edge
    traction = total_force / (element_length * thickness)

    # 2-point Gauss quadrature
    a = 1.0 / np.sqrt(3.0)

    for eta in [-a, a]:

        # Right edge of Q4 element
        xi = 1.0

        N = np.array([
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta)
        ])

        # Edge Jacobian
        J_edge = element_length / 2.0

        # Downward traction
        tx = 0.0
        ty = -traction

        for i in range(4):

            fe[2*i] += (
                N[i] * tx * J_edge * thickness
            )

            fe[2*i + 1] += (
                N[i] * ty * J_edge * thickness
            )

    return fe


# -------------------------------------------------
# RUN ONE CANTILEVER FEM ANALYSIS
# -------------------------------------------------

def run_cantilever(nx, ny):

    # -------------------------------------------------
    # CREATE MESH
    # -------------------------------------------------

    nodes, elements = create_rectangular_mesh(
        length,
        height,
        nx,
        ny
    )

    print("\n" + "=" * 60)
    print(f"MESH: {nx} x {ny}")
    print("=" * 60)

    print("Number of nodes   :", len(nodes))
    print("Number of elements:", len(elements))


    # -------------------------------------------------
    # GLOBAL STIFFNESS MATRIX
    # -------------------------------------------------

    number_of_dofs = 2 * len(nodes)

    K_global = np.zeros(
        (number_of_dofs, number_of_dofs)
    )

    for element_nodes in elements:

        coordinates = nodes[element_nodes]

        Ke = element_stiffness_matrix(
            coordinates,
            thickness,
            D
        )

        K_global = assemble_element(
            K_global,
            Ke,
            element_nodes
        )


    # -------------------------------------------------
    # GLOBAL FORCE VECTOR
    # -------------------------------------------------

    F = np.zeros(number_of_dofs)

    # Find elements having an edge on x = length
    for element_nodes in elements:

        coordinates = nodes[element_nodes]

        right_edge_local_nodes = np.where(
            np.isclose(
                coordinates[:, 0],
                length
            )
        )[0]

        # A Q4 element on the right boundary
        # has exactly two nodes on x = length
        if len(right_edge_local_nodes) == 2:

            # Length of this element's right edge
            edge_node_1 = right_edge_local_nodes[0]
            edge_node_2 = right_edge_local_nodes[1]

            edge_length = np.linalg.norm(
                coordinates[edge_node_2]
                - coordinates[edge_node_1]
            )

            # Total load carried by this edge segment
            edge_force = total_force / ny

            # Consistent local load vector
            fe = right_edge_load(
                edge_length,
                thickness,
                edge_force
            )

            # Local → global DOF mapping
            dof_indices = []

            for node in element_nodes:

                dof_indices.append(2 * node)
                dof_indices.append(2 * node + 1)

            # Assemble load vector
            for i in range(8):

                F[dof_indices[i]] += fe[i]


    # -------------------------------------------------
    # CHECK TOTAL APPLIED FORCE
    # -------------------------------------------------

    total_Fx = np.sum(F[0::2])
    total_Fy = np.sum(F[1::2])

    print("\nApplied force:")
    print("Total Fx:", total_Fx)
    print("Total Fy:", total_Fy)


    # -------------------------------------------------
    # FIND FIXED LEFT EDGE
    # -------------------------------------------------

    fixed_dofs = []

    for node, (x, y) in enumerate(nodes):

        if np.isclose(x, 0.0):

            fixed_dofs.append(2 * node)
            fixed_dofs.append(2 * node + 1)

    print("\nFixed DOFs:")
    print(fixed_dofs)


    # -------------------------------------------------
    # APPLY BOUNDARY CONDITIONS
    # -------------------------------------------------

    K_bc, F_bc = apply_zero_displacement_bc(
        K_global,
        F,
        fixed_dofs
    )


    # -------------------------------------------------
    # SOLVE FEM SYSTEM
    # -------------------------------------------------

    d = solve_system(
        K_bc,
        F_bc
    )


    # -------------------------------------------------
    # FIND FREE-END NODES
    # -------------------------------------------------

    right_nodes = np.where(
        np.isclose(
            nodes[:, 0],
            length
        )
    )[0]


    # -------------------------------------------------
    # FREE-END VERTICAL DISPLACEMENT
    # -------------------------------------------------

    tip_uy_values = d[
        2 * right_nodes + 1
    ]

    tip_uy = np.mean(
        tip_uy_values
    )


    # -------------------------------------------------
    # RESULTS FOR THIS MESH
    # -------------------------------------------------

    print("\nFree-end nodes:")
    print(right_nodes)

    print("\nFree-end vertical displacements:")
    print(tip_uy_values)

    print("\nAverage free-end vertical displacement:")
    print(f"{tip_uy:.10e} m")

    print(
        f"Maximum displacement: "
        f"{np.max(np.abs(d)):.10e} m"
    )


    return tip_uy, len(nodes), len(elements)


# -------------------------------------------------
# MESH CONVERGENCE STUDY
# -------------------------------------------------

mesh_sizes = [
    (2, 1),
    (4, 2),
    (8, 4),
    (16, 8)
]

results = []


for nx, ny in mesh_sizes:

    tip_uy, number_of_nodes, number_of_elements = (
        run_cantilever(nx, ny)
    )

    results.append([
        nx,
        ny,
        number_of_nodes,
        number_of_elements,
        tip_uy
    ])


# -------------------------------------------------
# CONVERGENCE TABLE
# -------------------------------------------------

print("\n\n")
print("=" * 75)
print("MESH CONVERGENCE RESULTS")
print("=" * 75)

print(
    f"{'Mesh':<12}"
    f"{'Nodes':<12}"
    f"{'Elements':<12}"
    f"{'Tip Uy (m)':<20}"
)

print("-" * 75)

for nx, ny, nodes_count, elements_count, tip_uy in results:

    print(
        f"{nx} x {ny:<7}"
        f"{nodes_count:<12}"
        f"{elements_count:<12}"
        f"{tip_uy:<20.10e}"
    )

    # -------------------------------------------------
# RELATIVE ERROR USING FINEST MESH AS REFERENCE
# -------------------------------------------------

reference = abs(results[-1][4])

print("\n")
print("=" * 75)
print("CONVERGENCE ERROR")
print("=" * 75)

print(
    f"{'Mesh':<12}"
    f"{'Tip Uy (m)':<20}"
    f"{'Relative Error (%)':<20}"
)

print("-" * 75)

for nx, ny, nodes_count, elements_count, tip_uy in results:

    error = (
        abs(abs(tip_uy) - reference)
        / reference
        * 100
    )

    print(
        f"{nx} x {ny:<7}"
        f"{tip_uy:<20.10e}"
        f"{error:<20.6f}"
    )
    # -------------------------------------------------
# CONVERGENCE PLOT
# -------------------------------------------------

import matplotlib.pyplot as plt

elements = [result[3] for result in results]
tip_displacements = [
    abs(result[4]) * 1000
    for result in results
]

plt.figure()

plt.plot(
    elements,
    tip_displacements,
    marker="o"
)

plt.xlabel("Number of Elements")
plt.ylabel("Tip Vertical Displacement (mm)")
plt.title("Mesh Convergence Study")

plt.grid(True)

plt.savefig(
    "/home/132612011/femproject/09_figures/mesh_convergence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()