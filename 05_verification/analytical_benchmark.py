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
# ANALYTICAL BENCHMARK PARAMETERS
# -------------------------------------------------

length = 2.0
height = 1.0
thickness = 0.01

E = 70e9
nu = 0.33

kappa = 1.0e-4

D = plane_stress_matrix(E, nu)
# -------------------------------------------------
# EXACT DISPLACEMENT FIELD
# -------------------------------------------------

def exact_displacement(x, y):

    ux = -kappa * x * y

    uy = 0.5 * kappa * (
        x**2 + nu * y**2
    )

    return ux, uy
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
# ANALYTICAL BENCHMARK PARAMETERS
# -------------------------------------------------

length = 2.0
height = 1.0
thickness = 0.01

E = 70e9
nu = 0.33

kappa = 1.0e-4

D = plane_stress_matrix(E, nu)
# -------------------------------------------------
# EXACT DISPLACEMENT FIELD
# -------------------------------------------------

def exact_displacement(x, y):

    ux = -kappa * x * y

    uy = 0.5 * kappa * (
        x**2 + nu * y**2
    )

    return ux, uy
ux, uy = exact_displacement(2.0, 1.0)

print("Exact displacement at (2,1):")
print("ux =", ux, "m")
print("uy =", uy, "m")

# -------------------------------------------------
# EXACT BOUNDARY TRACTION
# -------------------------------------------------

def exact_traction(y, normal_x):

    sigma_x = -E * kappa * y

    tx = sigma_x * normal_x
    ty = 0.0

    return tx, ty
print("\nExact traction at y = 1 m:")

tx_right, ty_right = exact_traction(1.0, 1.0)
print("Right edge:", tx_right, ty_right, "Pa")

tx_left, ty_left = exact_traction(1.0, -1.0)
print("Left edge :", tx_left, ty_left, "Pa")

# -------------------------------------------------
# APPLY PRESCRIBED DISPLACEMENT
# -------------------------------------------------

def apply_prescribed_displacement(K, F, prescribed_dofs, prescribed_values):

    K_modified = K.copy()
    F_modified = F.copy()

    for dof, value in zip(prescribed_dofs, prescribed_values):

        # Adjust RHS for the known displacement
        F_modified -= K_modified[:, dof] * value

        # Remove coupling
        K_modified[dof, :] = 0.0
        K_modified[:, dof] = 0.0

        # Enforce prescribed value
        K_modified[dof, dof] = 1.0
        F_modified[dof] = value

    return K_modified, F_modified
# -------------------------------------------------
# CREATE MESH
# -------------------------------------------------

nx = 4
ny = 2

nodes, elements = create_rectangular_mesh(
    length,
    height,
    nx,
    ny
)

print("\nMesh:")
print("Nodes:", len(nodes))
print("Elements:", len(elements))

# -------------------------------------------------
# LEFT EDGE NODES
# -------------------------------------------------

left_nodes = np.where(
    np.isclose(nodes[:, 0], 0.0)
)[0]

print("\nLeft edge nodes:")
print(left_nodes)

# -------------------------------------------------
# LEFT EDGE NODES
# -------------------------------------------------

left_nodes = np.where(
    np.isclose(nodes[:, 0], 0.0)
)[0]

print("\nLeft edge nodes:")
print(left_nodes)
# -------------------------------------------------
# EXACT DISPLACEMENTS ON LEFT EDGE
# -------------------------------------------------

prescribed_dofs = []
prescribed_values = []

for node in left_nodes:

    x, y = nodes[node]

    ux, uy = exact_displacement(x, y)

    prescribed_dofs.append(2 * node)
    prescribed_values.append(ux)

    prescribed_dofs.append(2 * node + 1)
    prescribed_values.append(uy)

print("\nPrescribed DOFs:")
print(prescribed_dofs)

print("\nPrescribed displacement values:")
print(prescribed_values)