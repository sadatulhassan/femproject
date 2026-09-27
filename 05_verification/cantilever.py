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

D = plane_stress_matrix(E, nu)

# -------------------------------------------------
# CREATE CANTILEVER MESH
# -------------------------------------------------

nx = 2
ny = 1

nodes, elements = create_rectangular_mesh(
    length,
    height,
    nx,
    ny
)

print("Number of nodes:", len(nodes))
print("Number of elements:", len(elements))

print("\nNodes:")
print(nodes)

print("\nElements:")
print(elements)

# -------------------------------------------------
# CONSISTENT LOAD ON RIGHT EDGE
# -------------------------------------------------

def right_edge_load(element_length, thickness, total_force):

    fe = np.zeros(8)

    # Uniform traction
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

            fe[2*i] += N[i] * tx * J_edge * thickness
            fe[2*i + 1] += N[i] * ty * J_edge * thickness

    return fe


# -------------------------------------------------
# TEST THE EDGE LOAD
# -------------------------------------------------

element_length = 1.0
total_force = 1000.0

fe = right_edge_load(
    element_length,
    thickness,
    total_force
)

print("\nConsistent edge load vector:")
print(fe)

print("\nTotal vertical force:")
print(np.sum(fe[1::2]))

# -------------------------------------------------
# ASSEMBLE EDGE LOAD INTO GLOBAL FORCE VECTOR
# -------------------------------------------------

F = np.zeros(2 * len(nodes))

# Rightmost element
right_element = elements[-1]

# Local element load vector
fe = right_edge_load(
    element_length,
    thickness,
    total_force
)

# Convert local DOFs to global DOFs
dof_indices = []

for node in right_element:
    dof_indices.append(2 * node)
    dof_indices.append(2 * node + 1)

# Assemble
for i in range(8):
    F[dof_indices[i]] += fe[i]

print("\nGlobal force vector:")
print(F)

print("\nTotal Fx:", np.sum(F[0::2]))
print("Total Fy:", np.sum(F[1::2]))

# -------------------------------------------------
# APPLY FIXED LEFT EDGE
# -------------------------------------------------

fixed_dofs = [0, 1, 6, 7]

K_dummy = np.eye(len(F))

K_bc, F_bc = apply_zero_displacement_bc(
    K_dummy,
    F,
    fixed_dofs
)

print("\nFixed DOFs:")
print(fixed_dofs)

print("\nForce vector after boundary conditions:")
print(F_bc)

# -------------------------------------------------
# ASSEMBLE GLOBAL STIFFNESS MATRIX
# -------------------------------------------------

K_global = np.zeros((2 * len(nodes), 2 * len(nodes)))

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

print("\nGlobal stiffness matrix:")
print(K_global)

# -------------------------------------------------
# APPLY BOUNDARY CONDITIONS TO K AND F
# -------------------------------------------------

K_bc, F_bc = apply_zero_displacement_bc(
    K_global,
    F,
    fixed_dofs
)

print("\nBoundary conditions applied.")
print("Fixed DOFs:", fixed_dofs)

# -------------------------------------------------
# SOLVE FEM SYSTEM
# -------------------------------------------------

d = solve_system(K_bc, F_bc)

print("\nNodal displacement vector:")
print(d)

print("\nMaximum displacement:")
print(np.max(np.abs(d)))