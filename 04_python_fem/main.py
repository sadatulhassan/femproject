import numpy as np

from material import plane_stress_matrix
from element_q4 import element_stiffness_matrix
from assembly import assemble_element
from boundary_conditions import apply_zero_displacement_bc
from solver import solve_system
from postprocess import element_strain_stress
from mesh_generator import create_rectangular_mesh

# ============================================================
# 1. BENCHMARK MESH
# ============================================================

#       3 -------- 4 -------- 5
#       |          |          |
#       | Element 1| Element 2|
#       |          |          |
#       0 -------- 1 -------- 2

length = 2.0
height = 1.0

nx = 8
ny = 4

nodes, elements = create_rectangular_mesh(
    length,
    height,
    nx,
    ny
)


print("=" * 60)
print("          Q4 FEM TENSION BENCHMARK")
print("=" * 60)


# ============================================================
# 2. MATERIAL
# ============================================================

E = 70e9
nu = 0.33
thickness = 0.01

D = plane_stress_matrix(E, nu)


# ============================================================
# 3. GLOBAL STIFFNESS MATRIX
# ============================================================

number_of_nodes = len(nodes)
number_of_dofs = 2 * number_of_nodes

K_global = np.zeros(
    (number_of_dofs, number_of_dofs)
)


# ============================================================
# 4. ELEMENT STIFFNESS + ASSEMBLY
# ============================================================

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


# ============================================================
# 5. APPLY TENSILE FORCE
# ============================================================



# Right edge:
#
# Node 2 -> DOF 4
# Node 5 -> DOF 10
#
# Total force = 1000 N

F = np.zeros(number_of_dofs)

right_edge_nodes = []

tolerance = 1e-12

for node, coordinate in enumerate(nodes):

    x = coordinate[0]

    if abs(x - length) < tolerance:
        right_edge_nodes.append(node)

force_per_node = 1000.0 / len(right_edge_nodes)

for node in right_edge_nodes:

    F[2 * node] = force_per_node


print("Applied forces:")

for node in right_edge_nodes:
    print(f"Node {node}, Fx = {F[2*node]:.2f} N")

print(f"Total Fx  = {np.sum(F[0::2]):.2f} N")


# ============================================================
# 6. BOUNDARY CONDITIONS
# ============================================================

# Node 0:
#   ux = 0
#   uy = 0
#
# Node 3:
#   ux = 0
#
# This prevents rigid-body motion while allowing
# Poisson contraction.

fixed_dofs = []

tolerance = 1e-12

for node, coordinate in enumerate(nodes):

    x = coordinate[0]
    y = coordinate[1]

    # Left edge
    if abs(x) < tolerance:

        # Fix x displacement
        fixed_dofs.append(2 * node)

# Fix y displacement at bottom-left node
for node, coordinate in enumerate(nodes):

    x = coordinate[0]
    y = coordinate[1]

    if abs(x) < tolerance and abs(y) < tolerance:

        fixed_dofs.append(2 * node + 1)

        break

K_modified, F_modified = apply_zero_displacement_bc(
    K_global,
    F,
    fixed_dofs
)


print("\nFixed DOFs:")
print(fixed_dofs)


# ============================================================
# 7. SOLVE
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
        f"ux = {ux:.8e} m, "
        f"uy = {uy:.8e} m"
    )


# ============================================================
# 8. ANALYTICAL SOLUTION
# ============================================================

height = 1.0

area = height * thickness

analytical_stress = 1000.0 / area

analytical_ex = analytical_stress / E

analytical_ey = -nu * analytical_ex


print("\nANALYTICAL SOLUTION")
print("-" * 60)

print(
    f"Area       = {area:.6e} m^2"
)

print(
    f"sigma_x    = {analytical_stress:.6e} Pa"
)

print(
    f"sigma_x    = {analytical_stress / 1e6:.6f} MPa"
)

print(
    f"epsilon_x  = {analytical_ex:.6e}"
)

print(
    f"epsilon_y  = {analytical_ey:.6e}"
)


# ============================================================
# 9. POST-PROCESSING
# ============================================================

print("\nFEM STRAIN AND STRESS")
print("-" * 60)

for element_number, element_nodes in enumerate(elements):

    coordinates = nodes[element_nodes]

    # Get the 8 displacement DOFs for this element
    element_dofs = []

    for node in element_nodes:

        element_dofs.append(2 * node)
        element_dofs.append(2 * node + 1)

    element_displacements = d[element_dofs]

    strains, stresses = element_strain_stress(
        coordinates,
        element_displacements,
        D
    )

    print(f"\nElement {element_number + 1}")

    print("\nStrain at Gauss points:")
    print(strains)

    print("\nStress at Gauss points [MPa]:")
    print(stresses / 1e6)


# ============================================================
# 10. FORCE EQUILIBRIUM
# ============================================================

applied_fx = np.sum(F[0::2])

reactions = K_global @ d - F
reaction_fx = 0.0

for dof in fixed_dofs:
    if dof % 2 == 0:
        reaction_fx += reactions[dof]

print("\nFORCE EQUILIBRIUM")
print("-" * 60)
print(f"Applied Fx = {applied_fx:.6f} N")
print(f"Support Rx = {reaction_fx:.6f} N")
print(f"Sum Fx = {applied_fx + reaction_fx:.6e} N")


# ============================================================
# 11. BENCHMARK COMPARISON
# ============================================================

# Average FEM sigma_x over all Gauss points

fem_sigma_x_values = []

for element_nodes in elements:

    coordinates = nodes[element_nodes]

    element_dofs = []

    for node in element_nodes:
        element_dofs.append(2 * node)
        element_dofs.append(2 * node + 1)

    element_displacements = d[element_dofs]

    strains, stresses = element_strain_stress(
        coordinates,
        element_displacements,
        D
    )

    fem_sigma_x_values.extend(
        stresses[:, 0]
    )


fem_sigma_x = np.mean(
    fem_sigma_x_values
)

percentage_error = (
    abs(fem_sigma_x - analytical_stress)
    / abs(analytical_stress)
    * 100
)


print("\nBENCHMARK COMPARISON")
print("-" * 60)

print(
    f"Analytical sigma_x = "
    f"{analytical_stress / 1e6:.6f} MPa"
)

print(
    f"FEM average sigma_x = "
    f"{fem_sigma_x / 1e6:.6f} MPa"
)

print(
    f"Percentage error = "
    f"{percentage_error:.6f} %"
)


print("\n" + "=" * 60)
print("             BENCHMARK COMPLETE")
print("=" * 60)