import numpy as np
import matplotlib.pyplot as plt
from material import plane_stress_matrix
from element_q4 import element_stiffness_matrix
from assembly import assemble_element
from boundary_conditions import apply_zero_displacement_bc
from solver import solve_system
from postprocess import element_strain_stress


# ============================================================
# 1. MESH
# ============================================================

#       3 -------- 4 -------- 5
#       |          |          |
#       | Element 1| Element 2|
#       |          |          |
#       0 -------- 1 -------- 2

nodes = np.array([
    [0.0, 0.0],   # Node 0
    [1.0, 0.0],   # Node 1
    [2.0, 0.0],   # Node 2
    [0.0, 1.0],   # Node 3
    [1.0, 1.0],   # Node 4
    [2.0, 1.0]    # Node 5
])

# Q4 node ordering:
#
# 4 -------- 3
# |          |
# |          |
# 1 -------- 2

elements = np.array([
    [0, 1, 4, 3],   # Element 1
    [1, 2, 5, 4]    # Element 2
])


print("=" * 60)
print("          2-ELEMENT Q4 FEM SIMULATION")
print("=" * 60)


# ============================================================
# 2. MATERIAL PROPERTIES
# ============================================================

E = 70e9          # Young's modulus [Pa]
nu = 0.33         # Poisson's ratio
thickness = 0.01  # Thickness [m]

D = plane_stress_matrix(E, nu)

print("\nMATERIAL")
print("-" * 60)
print(f"Young's modulus E = {E:.3e} Pa")
print(f"Poisson's ratio    = {nu}")
print(f"Thickness          = {thickness} m")

print("\nMaterial matrix D:")
print(D)


# ============================================================
# 3. GLOBAL SYSTEM SIZE
# ============================================================

number_of_nodes = len(nodes)
number_of_dofs = 2 * number_of_nodes

K_global = np.zeros(
    (number_of_dofs, number_of_dofs)
)

print("\nSYSTEM")
print("-" * 60)
print(f"Number of nodes = {number_of_nodes}")
print(f"Number of elements = {len(elements)}")
print(f"Number of DOFs = {number_of_dofs}")


# ============================================================
# 4. ELEMENT CALCULATIONS + ASSEMBLY
# ============================================================

element_stiffness_matrices = []

for element_number, element_nodes in enumerate(elements):

    # Coordinates of the four nodes belonging
    # to this element
    coordinates = nodes[element_nodes]

    # Calculate element stiffness matrix
    Ke = element_stiffness_matrix(
        coordinates,
        thickness,
        D
    )

    # Store it for later post-processing
    element_stiffness_matrices.append(Ke)

    # Assemble into global stiffness matrix
    K_global = assemble_element(
        K_global,
        Ke,
        element_nodes
    )

    print("\nELEMENT")
    print("-" * 60)
    print(f"Element {element_number + 1}")
    print(f"Nodes: {element_nodes}")
    print(f"Ke shape: {Ke.shape}")

    symmetry_error = np.max(
        np.abs(Ke - Ke.T)
    )

    print(f"Ke symmetry error: {symmetry_error:.3e}")


# ============================================================
# 5. GLOBAL STIFFNESS MATRIX
# ============================================================

print("\nGLOBAL STIFFNESS MATRIX")
print("-" * 60)
print("Shape:", K_global.shape)

global_symmetry_error = np.max(
    np.abs(K_global - K_global.T)
)

print(
    f"Global symmetry error: "
    f"{global_symmetry_error:.3e}"
)


# ============================================================
# 6. GLOBAL FORCE VECTOR
# ============================================================

F = np.zeros(number_of_dofs)

# Apply 1000 N downward force at Node 5.
#
# Node 5:
#     DOF 10 -> ux
#     DOF 11 -> uy
#
# Downward = negative y direction

F[11] = -1000.0

print("\nFORCE VECTOR")
print("-" * 60)
print(F)


# ============================================================
# 7. BOUNDARY CONDITIONS
# ============================================================

# Fix Node 0 completely:
#     DOF 0 -> ux = 0
#     DOF 1 -> uy = 0
#
# Fix Node 3 completely:
#     DOF 6 -> ux = 0
#     DOF 7 -> uy = 0

fixed_dofs = [0, 1, 6, 7]

K_modified, F_modified = apply_zero_displacement_bc(
    K_global,
    F,
    fixed_dofs
)

print("\nBOUNDARY CONDITIONS")
print("-" * 60)
print("Fixed DOFs:", fixed_dofs)

print("\nForce vector after BC:")
print(F_modified)


# ============================================================
# 8. SOLVE Kd = F
# ============================================================

d = solve_system(
    K_modified,
    F_modified
)

print("\nNODAL DISPLACEMENTS")
print("-" * 60)

for node in range(number_of_nodes):

    ux = d[2 * node]
    uy = d[2 * node + 1]

    print(
        f"Node {node}: "
        f"ux = {ux:.6e} m, "
        f"uy = {uy:.6e} m"
    )


# ============================================================
# 9. VERIFY THE SOLUTION
# ============================================================

F_check = K_modified @ d

error = np.max(
    np.abs(F_check - F_modified)
)

print("\nSOLVER VERIFICATION")
print("-" * 60)
print("Maximum |Kd - F| =", error)

print("\nKd:")
print(F_check)

print("\nF:")
print(F_modified)


# ============================================================
# 10. POST-PROCESSING
# ============================================================

print("\nSTRAIN AND STRESS")
print("-" * 60)

for element_number, element_nodes in enumerate(elements):

    # Coordinates of this element
    coordinates = nodes[element_nodes]

    # --------------------------------------------------------
    # Extract the 8 displacement DOFs belonging to this element
    # --------------------------------------------------------

    element_dofs = []

    for node in element_nodes:
        element_dofs.append(2 * node)
        element_dofs.append(2 * node + 1)

    element_displacements = d[element_dofs]

    # Calculate strain and stress at Gauss points
    strains, stresses = element_strain_stress(
        coordinates,
        element_displacements,
        D
    )

    print(f"\nElement {element_number + 1}")
    print("Nodes:", element_nodes)

    print("\nElement displacement vector:")
    print(element_displacements)

    print("\nStrains at Gauss points:")
    print(strains)

    print("\nStresses at Gauss points [Pa]:")
    print(stresses)

    print("\nStresses at Gauss points [MPa]:")
    print(stresses / 1e6)


# ============================================================
# 11. END
# ============================================================

print("\n" + "=" * 60)
print("              FEM ANALYSIS COMPLETE")
print("=" * 60)

print("\n" + "=" * 60)
print("              FEM ANALYSIS COMPLETE")
print("=" * 60)

# ============================================================
# 12. PLOT ORIGINAL AND DEFORMED MESH
# ============================================================

deformation_scale = 5000.0

# Convert global displacement vector into (ux, uy)
# for every node
displacements = d.reshape((-1, 2))

# Calculate deformed coordinates
deformed_nodes = (
    nodes + deformation_scale * displacements
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))


# Element edges
element_edges = [
    (0, 1, 4, 3, 0),
    (1, 2, 5, 4, 1)
]


# Plot original mesh
for element in element_edges:

    node_ids = element[:4]

    polygon = np.array([
        nodes[node_ids[0]],
        nodes[node_ids[1]],
        nodes[node_ids[2]],
        nodes[node_ids[3]],
        nodes[node_ids[0]]
    ])

    plt.plot(
        polygon[:, 0],
        polygon[:, 1],
        "--",
        linewidth=1.5,
        label="Original mesh"
        if element[4] == 0 else None
    )


# Plot deformed mesh
for element in element_edges:

    node_ids = element[:4]

    polygon = np.array([
        deformed_nodes[node_ids[0]],
        deformed_nodes[node_ids[1]],
        deformed_nodes[node_ids[2]],
        deformed_nodes[node_ids[3]],
        deformed_nodes[node_ids[0]]
    ])

    plt.plot(
        polygon[:, 0],
        polygon[:, 1],
        "-",
        linewidth=2.0,
        label="Deformed mesh"
        if element[4] == 0 else None
    )


# ------------------------------------------------------------
# Plot node numbers
# ------------------------------------------------------------

for i, coordinate in enumerate(nodes):

    plt.text(
        coordinate[0],
        coordinate[1],
        f"  {i}",
        fontsize=11
    )


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

plt.xlabel("x [m]")
plt.ylabel("y [m]")

plt.title(
    f"Q4 FEM: Original vs Deformed Mesh "
    f"(scale = {deformation_scale:g})"
)

plt.axis("equal")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()

# ============================================================
# REACTION FORCE CHECK
# ============================================================

reactions = K_global @ d - F

print("\nREACTION FORCES")
print("-" * 60)

for dof in fixed_dofs:
    print(
        f"DOF {dof}: "
        f"Reaction = {reactions[dof]:.6f} N"
    )

print("\nTotal vertical reaction:")
vertical_reaction = reactions[1] + reactions[7]

print(f"{vertical_reaction:.6f} N")

print("\nApplied vertical force:")
print(f"{F[11]:.6f} N")

print("\nForce equilibrium:")
print(
    f"Reaction + Applied = "
    f"{vertical_reaction + F[11]:.6e} N"
)