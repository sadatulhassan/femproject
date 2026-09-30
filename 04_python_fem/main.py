import numpy as np

# ============================================================
# IMPORTS
# ============================================================
from visualization import (
    plot_von_mises,
    plot_sigma_x,
    plot_sigma_y,
    plot_tau_xy,
    plot_displacement
)
from mesh_io import (
    read_ansys_cdb,
    classify_elements,
    correct_element_orientation
)

from material import plane_stress_matrix

from element_q4 import element_stiffness_matrix

from assembly import assemble_element

from boundary_conditions import (
    create_load_vector,
    create_fixed_dofs,
    apply_zero_displacement_bc
)

from solver import solve_system

from postprocess import (
    calculate_all_element_stresses,
    print_stress_summary
)


# ============================================================
# 1. READ ANSYS MESH
# ============================================================

print("=" * 60)
print("          PYTHON FEM - ANSYS MESH")
print("=" * 60)

nodes, elements = read_ansys_cdb("my_mesh.cdb")

# Separate normal Q4 elements from degenerate elements
q4_elements, degenerate_elements = classify_elements(elements)

# Correct Q4 element orientation
q4_elements = correct_element_orientation(
    nodes,
    q4_elements
)

print("\nMESH INFORMATION")
print("-" * 60)

print("Number of nodes       :", len(nodes))
print("Total elements        :", len(elements))
print("Q4 elements           :", len(q4_elements))
print("Degenerate elements   :", len(degenerate_elements))


# ============================================================
# 2. MATERIAL PROPERTIES
# ============================================================

# Units:
#
# Length       = mm
# Force        = N
# Young's E    = N/mm^2 = MPa
# Thickness    = mm
# Stress       = N/mm^2 = MPa

E = 200000.0          # MPa
nu = 0.30             # Poisson's ratio
thickness = 2.0       # mm

# Plane stress material matrix
D = plane_stress_matrix(E, nu)

print("\nMATERIAL")
print("-" * 60)

print("Young's modulus E     :", E, "MPa")
print("Poisson's ratio       :", nu)
print("Thickness             :", thickness, "mm")


# ============================================================
# 3. GLOBAL SYSTEM SIZE
# ============================================================

number_of_nodes = len(nodes)

# Two DOFs per node:
#
# DOF 2*i     = Ux
# DOF 2*i + 1 = Uy

number_of_dofs = 2 * number_of_nodes

# Global stiffness matrix
K_global = np.zeros(
    (number_of_dofs, number_of_dofs),
    dtype=float
)


# ============================================================
# 4. ASSEMBLE GLOBAL STIFFNESS MATRIX
# ============================================================

print("\nASSEMBLING GLOBAL STIFFNESS MATRIX...")
print("-" * 60)

for element_number, element_nodes in enumerate(q4_elements):

    # Coordinates of the four nodes
    coordinates = nodes[element_nodes]

    # Element stiffness matrix
    Ke = element_stiffness_matrix(
        coordinates,
        thickness,
        D
    )

    # Assemble element matrix into global matrix
    K_global = assemble_element(
        K_global,
        Ke,
        element_nodes
    )

print("Global stiffness matrix assembled.")
print("Matrix size:", K_global.shape)


# ============================================================
# 5. APPLY TENSILE LOAD
# ============================================================

total_force = 10000.0

F = create_load_vector(
    nodes,
    total_force,
    number_of_dofs
)


# ============================================================
# 6. BOUNDARY CONDITIONS
# ============================================================

# Left edge:
#
# Ux = 0 for all left-edge nodes
#
# Uy = 0 for only ONE left-edge node
#
# This allows Poisson contraction in Y while
# preventing rigid-body translation.

fixed_dofs = create_fixed_dofs(
    nodes
)


# ============================================================
# 7. MODIFY SYSTEM FOR BOUNDARY CONDITIONS
# ============================================================

K_modified, F_modified = apply_zero_displacement_bc(
    K_global,
    F,
    fixed_dofs
)


# ============================================================
# 8. SOLVE FOR NODAL DISPLACEMENTS
# ============================================================

print("\nSOLVING FEM SYSTEM...")
print("-" * 60)

d = solve_system(
    K_modified,
    F_modified
)


# ============================================================
# 9. CALCULATE STRAINS AND STRESSES
# ============================================================

print("\nCALCULATING STRAINS AND STRESSES...")
print("-" * 60)

all_strains, all_stresses, all_von_mises = (
    calculate_all_element_stresses(
        nodes,
        q4_elements,
        d,
        D
    )
)


# ============================================================
# 10. DISPLACEMENT RESULTS
# ============================================================

print("\nDISPLACEMENT RESULTS")
print("-" * 60)

# Maximum absolute displacement
max_displacement = np.max(
    np.abs(d)
)

# Maximum absolute X displacement
max_x_displacement = np.max(
    np.abs(d[0::2])
)

# Maximum absolute Y displacement
max_y_displacement = np.max(
    np.abs(d[1::2])
)

print(
    "Maximum |displacement|:",
    max_displacement
)

print(
    "Maximum X displacement:",
    max_x_displacement
)

print(
    "Maximum Y displacement:",
    max_y_displacement
)


# ============================================================
# 11. FIRST 10 NODE DISPLACEMENTS
# ============================================================

print("\nFirst 10 nodal displacements:")
print("-" * 60)

for i in range(min(10, number_of_nodes)):

    ux = d[2 * i]
    uy = d[2 * i + 1]

    print(
        f"Node {i:4d}: "
        f"Ux = {ux:.6e} mm, "
        f"Uy = {uy:.6e} mm"
    )


# ============================================================
# 12. SOLVER VERIFICATION
# ============================================================

residual = K_modified @ d - F_modified

print("\nSOLVER VERIFICATION")
print("-" * 60)

print(
    "Maximum |Kd - F| =",
    np.max(np.abs(residual))
)


# ============================================================
# 13. STRESS RESULTS
# ============================================================

results = print_stress_summary(
    all_stresses,
    all_von_mises
)
# ============================================================
# 14. VISUALIZATION
# ============================================================

print("\nGENERATING FEM VISUALIZATIONS...")
print("-" * 60)

# Von Mises stress
plot_von_mises(
    nodes,
    q4_elements,
    all_von_mises,
    save_path="von_mises_stress.png"
)

# Sigma X
plot_sigma_x(
    nodes,
    q4_elements,
    all_stresses,
    save_path="sigma_x.png"
)

# Sigma Y
plot_sigma_y(
    nodes,
    q4_elements,
    all_stresses,
    save_path="sigma_y.png"
)

# Shear stress
plot_tau_xy(
    nodes,
    q4_elements,
    all_stresses,
    save_path="tau_xy.png"
)

# Displacement magnitude
plot_displacement(
    nodes,
    q4_elements,
    d,
    save_path="displacement.png"
)

# ============================================================
# 15. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("             FEM SOLUTION COMPLETE")
print("=" * 60)