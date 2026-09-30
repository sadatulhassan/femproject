import numpy as np


from element_q4 import build_B_matrix
from gauss import gauss_quadrature_2x2


# -------------------------------------------------
# ELEMENT STRAIN AND STRESS
# -------------------------------------------------

def element_strain_stress(coordinates, element_displacements, D):

    points, weights = gauss_quadrature_2x2()

    strains = []
    stresses = []

    for point in points:

        xi = point[0]
        eta = point[1]

        B = build_B_matrix(
            xi,
            eta,
            coordinates
        )

        strain = B @ element_displacements
        stress = D @ strain

        strains.append(strain)
        stresses.append(stress)

    return np.array(strains), np.array(stresses)


# -------------------------------------------------
# VON MISES STRESS
# -------------------------------------------------

def von_mises_stress(stress):

    sigma_x = stress[..., 0]
    sigma_y = stress[..., 1]
    tau_xy = stress[..., 2]

    sigma_vm = np.sqrt(
        sigma_x**2
        - sigma_x * sigma_y
        + sigma_y**2
        + 3.0 * tau_xy**2
    )

    return sigma_vm


# -------------------------------------------------
# CALCULATE STRESSES FOR ALL ELEMENTS
# -------------------------------------------------

def calculate_all_element_stresses(
    nodes,
    elements,
    displacements,
    D
):

    number_of_elements = len(elements)

    all_strains = []
    all_stresses = []
    all_von_mises = []

    for element_nodes in elements:

        coordinates = nodes[element_nodes]

        # Extract element displacement vector
        element_dofs = []

        for node in element_nodes:

            element_dofs.append(2 * node)
            element_dofs.append(2 * node + 1)

        element_displacements = displacements[element_dofs]

        # Calculate stress at 4 Gauss points
        strains, stresses = element_strain_stress(
            coordinates,
            element_displacements,
            D
        )

        vm = von_mises_stress(stresses)

        all_strains.append(strains)
        all_stresses.append(stresses)
        all_von_mises.append(vm)

    return (
        np.array(all_strains),
        np.array(all_stresses),
        np.array(all_von_mises)
    )


# -------------------------------------------------
# MAXIMUM STRESS VALUES
# -------------------------------------------------

def find_maximum_stresses(all_stresses, all_von_mises):

    sigma_x = all_stresses[:, :, 0]
    sigma_y = all_stresses[:, :, 1]
    tau_xy = all_stresses[:, :, 2]

    # Maximum algebraic values
    max_sigma_x = np.max(sigma_x)
    min_sigma_x = np.min(sigma_x)

    max_sigma_y = np.max(sigma_y)
    min_sigma_y = np.min(sigma_y)

    max_tau_xy = np.max(tau_xy)
    min_tau_xy = np.min(tau_xy)

    # Maximum absolute normal stresses
    max_abs_sigma_x = np.max(np.abs(sigma_x))
    max_abs_sigma_y = np.max(np.abs(sigma_y))

    # Von Mises
    max_von_mises = np.max(all_von_mises)

    # Locations
    vm_location = np.unravel_index(
        np.argmax(all_von_mises),
        all_von_mises.shape
    )

    sigma_x_location = np.unravel_index(
        np.argmax(np.abs(sigma_x)),
        sigma_x.shape
    )

    sigma_y_location = np.unravel_index(
        np.argmax(np.abs(sigma_y)),
        sigma_y.shape
    )

    return {
        "max_sigma_x": max_sigma_x,
        "min_sigma_x": min_sigma_x,
        "max_sigma_y": max_sigma_y,
        "min_sigma_y": min_sigma_y,
        "max_tau_xy": max_tau_xy,
        "min_tau_xy": min_tau_xy,
        "max_abs_sigma_x": max_abs_sigma_x,
        "max_abs_sigma_y": max_abs_sigma_y,
        "max_von_mises": max_von_mises,
        "sigma_x_location": sigma_x_location,
        "sigma_y_location": sigma_y_location,
        "von_mises_location": vm_location
    }


# -------------------------------------------------
# PRINT STRESS SUMMARY
# -------------------------------------------------

def print_stress_summary(all_stresses, all_von_mises):

    results = find_maximum_stresses(
        all_stresses,
        all_von_mises
    )

    print("\n")
    print("=" * 70)
    print("STRESS RESULTS")
    print("=" * 70)

    print("\nNORMAL STRESS σx")
    print("-" * 70)

    print(
        f"Maximum σx       : "
        f"{results['max_sigma_x'] / 1e6:.6f} MPa"
    )

    print(
        f"Minimum σx       : "
        f"{results['min_sigma_x'] / 1e6:.6f} MPa"
    )

    print(
        f"Maximum |σx|     : "
        f"{results['max_abs_sigma_x'] / 1e6:.6f} MPa"
    )

    print("\nNORMAL STRESS σy")
    print("-" * 70)

    print(
        f"Maximum σy       : "
        f"{results['max_sigma_y'] / 1e6:.6f} MPa"
    )

    print(
        f"Minimum σy       : "
        f"{results['min_sigma_y'] / 1e6:.6f} MPa"
    )

    print(
        f"Maximum |σy|     : "
        f"{results['max_abs_sigma_y'] / 1e6:.6f} MPa"
    )

    print("\nSHEAR STRESS τxy")
    print("-" * 70)

    print(
        f"Maximum τxy       : "
        f"{results['max_tau_xy'] / 1e6:.6f} MPa"
    )

    print(
        f"Minimum τxy       : "
        f"{results['min_tau_xy'] / 1e6:.6f} MPa"
    )

    print("\nVON MISES STRESS")
    print("-" * 70)

    print(
        f"Maximum Von Mises : "
        f"{results['max_von_mises'] / 1e6:.6f} MPa"
    )

    print("=" * 70)

    return results