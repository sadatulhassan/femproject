import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from mesh_io import read_ansys_cdb


# -------------------------------------------------
# READ ANSYS MESH
# -------------------------------------------------

nodes, elements = read_ansys_cdb("my_mesh.cdb")


# -------------------------------------------------
# PLOT MESH
# -------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 7))


for element in elements:

    # Get the four node coordinates
    coords = nodes[element]

    x = coords[:, 0]
    y = coords[:, 1]

    # Close the element
    x = list(x) + [x[0]]
    y = list(y) + [y[0]]

    ax.plot(x, y, linewidth=0.4)


# -------------------------------------------------
# FORMAT PLOT
# -------------------------------------------------

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")

ax.set_title("ANSYS Mesh")

ax.set_aspect("equal")

ax.grid(True)

plt.tight_layout()

plt.show()